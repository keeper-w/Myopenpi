"""Validate a CALVIN LeRobot v3 download and prepare openpi normalization assets."""

import json
import pathlib

import numpy as np
import tyro

from openpi.shared import normalize
from openpi.training.calvin_v3_dataset import CalvinLeRobotV3Dataset
from openpi.training import config as _config


def _norm_stats(stats: dict) -> normalize.NormStats:
    return normalize.NormStats(
        mean=np.asarray(stats["mean"], dtype=np.float32),
        std=np.asarray(stats["std"], dtype=np.float32),
        q01=np.asarray(stats["q01"], dtype=np.float32),
        q99=np.asarray(stats["q99"], dtype=np.float32),
    )


def main(
    dataset_dir: pathlib.Path = pathlib.Path("data/lerobot_v3/Traly/calvin_abc_d-lerobot"),
    config_name: str = "pi05_calvin_lora",
    decode_images: bool = True,
) -> None:
    dataset_dir = dataset_dir.expanduser().resolve()
    config = _config.get_config(config_name)
    if not isinstance(config.data, _config.LeRobotV3CalvinDataConfig):
        raise ValueError(f"Config {config_name!r} is not a CALVIN v3 config")

    dataset = CalvinLeRobotV3Dataset(dataset_dir, config.model.action_horizon, load_images=decode_images)
    sample_indices = [0, len(dataset) // 2, len(dataset) - 1]
    for index in sample_indices:
        sample = dataset[index]
        if sample["observation.state"].shape != (15,):
            raise ValueError(f"Unexpected state shape at frame {index}: {sample['observation.state'].shape}")
        if sample["actions"].shape != (config.model.action_horizon, 7):
            raise ValueError(f"Unexpected action shape at frame {index}: {sample['actions'].shape}")
        if decode_images:
            if sample["observation.images.top"].shape != (200, 200, 3):
                raise ValueError(f"Unexpected top image shape at frame {index}")
            if sample["observation.images.wrist"].shape != (84, 84, 3):
                raise ValueError(f"Unexpected wrist image shape at frame {index}")
        if not sample["prompt"]:
            raise ValueError(f"Missing task prompt at frame {index}")

    source_stats = json.loads((dataset_dir / "meta/stats.json").read_text())
    norm_stats = {
        "state": _norm_stats(source_stats["observation.state"]),
        "actions": _norm_stats(source_stats["action.relative"]),
    }
    output_dir = config.assets_dirs / config.data.repo_id
    normalize.save(output_dir, norm_stats)

    print(f"Validated {len(dataset):,} CALVIN frames")
    print("State: 15 dims")
    print(f"Actions: {config.model.action_horizon} x 7 relative controls")
    print("Images: top 200x200 RGB, wrist 84x84 RGB")
    print(f"Normalization stats: {output_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    tyro.cli(main)
