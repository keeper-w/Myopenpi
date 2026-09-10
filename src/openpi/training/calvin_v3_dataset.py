"""Zero-copy reader for the CALVIN LeRobot v3 dataset.

The openpi revision in this repository uses the LeRobot v2 Python API, while the
Traly/calvin_abc_d-lerobot dataset is stored in LeRobot v3. This reader keeps the
downloaded parquet and video files unchanged and exposes samples in the format
expected by the openpi data transforms.
"""

import json
import pathlib
from typing import Any, SupportsIndex

import av
import numpy as np
import polars as pl


class CalvinLeRobotV3Dataset:
    """Random-access CALVIN dataset backed by v3 parquet and chunked MP4 files."""

    _TOP_IMAGE_KEY = "observation.images.top"
    _WRIST_IMAGE_KEY = "observation.images.wrist"

    def __init__(
        self,
        root: pathlib.Path | str,
        action_horizon: int,
        *,
        load_images: bool = True,
    ) -> None:
        self._root = pathlib.Path(root).expanduser().resolve()
        self._action_horizon = action_horizon
        self._load_images = load_images
        self._containers: dict[pathlib.Path, Any] = {}

        info_path = self._root / "meta/info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"CALVIN metadata not found: {info_path}")
        self._info = json.loads(info_path.read_text())
        if self._info.get("codebase_version") != "v3.0":
            raise ValueError(f"Expected LeRobot v3.0, got {self._info.get('codebase_version')!r}")
        if self._info.get("robot_type") != "panda":
            raise ValueError(f"Expected Panda robot data, got {self._info.get('robot_type')!r}")

        self._fps = float(self._info["fps"])
        self._image_shapes = {
            key: tuple(self._info["features"][key]["shape"]) for key in (self._TOP_IMAGE_KEY, self._WRIST_IMAGE_KEY)
        }

        data_files = sorted((self._root / "data").glob("chunk-*/*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No frame parquet files found under {self._root / 'data'}")
        frame_columns = [
            "observation.state",
            "action.relative",
            "timestamp",
            "episode_index",
            "index",
            "task_index",
        ]
        frames = pl.concat([pl.read_parquet(path, columns=frame_columns) for path in data_files], rechunk=True)

        self._states = frames["observation.state"].list.to_array(15).to_numpy().astype(np.float32, copy=False)
        self._actions = frames["action.relative"].list.to_array(7).to_numpy().astype(np.float32, copy=False)
        self._timestamps = frames["timestamp"].to_numpy().astype(np.float64, copy=False)
        self._episode_indices = frames["episode_index"].to_numpy().astype(np.int64, copy=False)
        self._task_indices = frames["task_index"].to_numpy().astype(np.int64, copy=False)
        frame_indices = frames["index"].to_numpy()
        expected_indices = np.arange(len(frames), dtype=frame_indices.dtype)
        if not np.array_equal(frame_indices, expected_indices):
            raise ValueError("CALVIN frame indices are not contiguous or frame parquet files are out of order")

        episode_files = sorted((self._root / "meta/episodes").glob("chunk-*/*.parquet"))
        if not episode_files:
            raise FileNotFoundError(f"No episode parquet files found under {self._root / 'meta/episodes'}")
        episodes = pl.concat([pl.read_parquet(path) for path in episode_files], rechunk=True).sort("episode_index")
        episode_ids = episodes["episode_index"].to_numpy()
        if not np.array_equal(episode_ids, np.arange(len(episodes), dtype=episode_ids.dtype)):
            raise ValueError("CALVIN episode indices are not contiguous")
        self._episode_last_indices = episodes["dataset_to_index"].to_numpy().astype(np.int64, copy=False) - 1

        self._video_locations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for key in (self._TOP_IMAGE_KEY, self._WRIST_IMAGE_KEY):
            prefix = f"videos/{key}"
            chunks = episodes[f"{prefix}/chunk_index"].to_numpy().astype(np.int64, copy=False)
            files = episodes[f"{prefix}/file_index"].to_numpy().astype(np.int64, copy=False)
            offsets = episodes[f"{prefix}/from_timestamp"].to_numpy().astype(np.float64, copy=False)
            self._video_locations[key] = (chunks, files, offsets)

        tasks_path = self._root / "meta/tasks.parquet"
        tasks = pl.read_parquet(tasks_path)
        task_text_column = next(column for column in tasks.columns if column != "task_index")
        self._tasks = dict(zip(tasks["task_index"].to_list(), tasks[task_text_column].to_list(), strict=True))

        if len(self._states) != int(self._info["total_frames"]):
            raise ValueError(f"Expected {self._info['total_frames']} frames, loaded {len(self._states)}")

    def __len__(self) -> int:
        return len(self._states)

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        index = index.__index__()
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        episode_index = int(self._episode_indices[index])
        last_index = int(self._episode_last_indices[episode_index])
        action_indices = np.minimum(index + np.arange(self._action_horizon), last_index)

        if self._load_images:
            top_image = self._decode_image(self._TOP_IMAGE_KEY, episode_index, float(self._timestamps[index]))
            wrist_image = self._decode_image(self._WRIST_IMAGE_KEY, episode_index, float(self._timestamps[index]))
        else:
            top_image = np.zeros(self._image_shapes[self._TOP_IMAGE_KEY], dtype=np.uint8)
            wrist_image = np.zeros(self._image_shapes[self._WRIST_IMAGE_KEY], dtype=np.uint8)

        return {
            "observation.images.top": top_image,
            "observation.images.wrist": wrist_image,
            "observation.state": self._states[index],
            "actions": self._actions[action_indices],
            "prompt": self._tasks[int(self._task_indices[index])],
        }

    def _decode_image(self, key: str, episode_index: int, timestamp: float) -> np.ndarray:
        chunks, files, offsets = self._video_locations[key]
        video_path = (
            self._root
            / "videos"
            / key
            / f"chunk-{int(chunks[episode_index]):03d}"
            / f"file-{int(files[episode_index]):03d}.mp4"
        )
        if not video_path.exists():
            raise FileNotFoundError(f"CALVIN video not found: {video_path}")

        container = self._containers.get(video_path)
        if container is None:
            container = av.open(str(video_path))
            self._containers[video_path] = container
        stream = container.streams.video[0]
        target_time = float(offsets[episode_index]) + timestamp
        target_pts = int(target_time / float(stream.time_base))
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)

        best_frame = None
        best_distance = float("inf")
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_time = float(frame.pts * stream.time_base)
            distance = abs(frame_time - target_time)
            if distance < best_distance:
                best_frame = frame
                best_distance = distance
            if frame_time >= target_time:
                break

        tolerance = 0.5 / self._fps + 1e-4
        if best_frame is None or best_distance > tolerance:
            raise RuntimeError(
                f"Could not decode {video_path} at {target_time:.6f}s "
                f"(closest distance={best_distance:.6f}s, tolerance={tolerance:.6f}s)"
            )
        return best_frame.to_ndarray(format="rgb24")

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_containers"] = {}
        return state

    def __del__(self) -> None:
        for container in getattr(self, "_containers", {}).values():
            container.close()
