"""Replay successful CALVIN sequences and save verified MP4 visualizations."""

import argparse
import json
from pathlib import Path
import textwrap

import cv2
import numpy as np
from openpi_client import websocket_client_policy
from tqdm import tqdm

import evaluate_openpi as calvin_eval


CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="outputs/calvin_eval/29999/results_finetuned_1000.json",
    )
    parser.add_argument(
        "--comparison",
        default=None,
        help="Comparison JSON; select only fine-tuned 5/5 successes where the original model failed.",
    )
    parser.add_argument("--output-dir", default="outputs/calvin_eval/29999/videos")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--min-completed", default=5, type=int)
    parser.add_argument("--limit", default=5, type=int)
    parser.add_argument(
        "--sequence-indices",
        default=None,
        help="Comma-separated explicit indices; by default select them from the result file.",
    )
    parser.add_argument("--attempts-per-sequence", default=1, type=int)
    parser.add_argument("--replan-steps", default=None, type=int)
    parser.add_argument("--fps", default=30, type=float)
    parser.add_argument("--success-hold-seconds", default=1.0, type=float)
    parser.add_argument("--use-egl", action="store_true")
    parser.add_argument("--show-gui", action="store_true")
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print selected sequence indices and tasks without starting CALVIN or contacting the server.",
    )
    return parser.parse_args()


def load_candidates(args):
    results_path = Path(args.results)
    report = json.loads(results_path.read_text())
    completed = report.get("completed_tasks_per_sequence")
    num_sequences = int(report["num_sequences"])
    if not isinstance(completed, list) or len(completed) != num_sequences:
        raise ValueError(
            "%s does not contain one completed-task count per sequence" % results_path
        )

    sequences = calvin_eval.get_sequences(num_sequences, num_workers=1)
    if args.sequence_indices:
        indices = [int(value.strip()) for value in args.sequence_indices.split(",") if value.strip()]
    elif args.comparison:
        comparison_path = Path(args.comparison)
        comparison = json.loads(comparison_path.read_text())
        if comparison.get("status") != "complete":
            raise ValueError("Comparison is not complete: %s" % comparison_path)
        if int(comparison.get("num_sequences", -1)) != num_sequences:
            raise ValueError("Comparison and fine-tuned report use different sequence counts")
        indices = [int(value) for value in comparison["finetuned_only_full_success_indices"]]
    else:
        indices = [index for index, value in enumerate(completed) if value >= args.min_completed]
    if args.limit > 0:
        indices = indices[: args.limit]
    if not indices:
        raise ValueError("No sequence matches --min-completed=%d" % args.min_completed)
    for index in indices:
        if not 0 <= index < num_sequences:
            raise IndexError("Sequence index %d is outside [0, %d)" % (index, num_sequences))
        if completed[index] < args.min_completed:
            raise ValueError(
                "Selected sequence %d completed only %d tasks in the fine-tuned report"
                % (index, completed[index])
            )
    return report, sequences, indices


class CompositeVideoRecorder:
    def __init__(self, path, fps, sequence_index, tasks, success_hold_seconds):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.path), codec, fps, (CANVAS_WIDTH, CANVAS_HEIGHT)
        )
        if not self.writer.isOpened():
            raise RuntimeError("Could not open MP4 writer for %s" % self.path)
        self.fps = fps
        self.sequence_index = sequence_index
        self.tasks = list(tasks)
        self.task_position = 0
        self.last_frame = None
        self.success_hold_frames = max(1, int(round(fps * success_hold_seconds)))

    def set_task(self, position):
        self.task_position = position

    @staticmethod
    def _put_wrapped(canvas, text, x, y, width=35, scale=0.62, color=(225, 225, 225)):
        for line in textwrap.wrap(str(text), width=width) or [""]:
            cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            y += 29
        return y

    def _compose(self, obs, prompt, subtask, step, succeeded):
        static_rgb = np.asarray(obs["rgb_obs"]["rgb_static"], dtype=np.uint8)
        wrist_rgb = np.asarray(obs["rgb_obs"]["rgb_gripper"], dtype=np.uint8)
        static_bgr = cv2.cvtColor(static_rgb, cv2.COLOR_RGB2BGR)
        wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)

        canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), (24, 27, 33), dtype=np.uint8)
        canvas[:, :720] = cv2.resize(static_bgr, (720, 720), interpolation=cv2.INTER_CUBIC)
        wrist = cv2.resize(wrist_bgr, (320, 320), interpolation=cv2.INTER_CUBIC)
        canvas[70:390, 840:1160] = wrist
        cv2.rectangle(canvas, (838, 68), (1162, 392), (235, 235, 235), 2)

        cv2.putText(
            canvas,
            "OpenPI  x  CALVIN-D",
            (770, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "WRIST CAMERA",
            (840, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (170, 180, 195),
            1,
            cv2.LINE_AA,
        )
        y = 465
        y = self._put_wrapped(
            canvas,
            "Sequence %d   Task %d/5" % (self.sequence_index, self.task_position),
            770,
            y,
            scale=0.7,
            color=(110, 220, 255),
        )
        y = self._put_wrapped(canvas, subtask, 770, y + 5, scale=0.62, color=(120, 235, 160))
        y = self._put_wrapped(canvas, prompt, 770, y + 8, scale=0.58)
        status = "TASK SUCCESS" if succeeded else "RUNNING  |  step %d/%d" % (step + 1, calvin_eval.EP_LEN)
        status_color = (80, 230, 120) if succeeded else (180, 190, 205)
        cv2.putText(
            canvas,
            status,
            (770, 690),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            status_color,
            2,
            cv2.LINE_AA,
        )
        return canvas

    def write(self, obs, prompt, subtask, step, succeeded):
        frame = self._compose(obs, prompt, subtask, step, succeeded)
        self.writer.write(frame)
        self.last_frame = frame
        if succeeded:
            for _ in range(self.success_hold_frames):
                self.writer.write(frame)

    def hold_final(self, text, success):
        if self.last_frame is None:
            return
        frame = self.last_frame.copy()
        color = (70, 220, 110) if success else (80, 100, 235)
        cv2.rectangle(frame, (0, 630), (720, 720), (15, 18, 22), -1)
        cv2.putText(
            frame,
            text,
            (35, 685),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.15,
            color,
            3,
            cv2.LINE_AA,
        )
        for _ in range(self.success_hold_frames * 2):
            self.writer.write(frame)

    def close(self):
        self.writer.release()


def replay_sequence(env, client, task_oracle, annotations, initial_state, tasks, replan_steps, recorder):
    robot_obs, scene_obs = calvin_eval.get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    completed = 0
    task_records = []
    for position, subtask in enumerate(tasks, start=1):
        prompt = str(annotations[subtask][0])
        recorder.set_task(position)
        success = calvin_eval.rollout_subtask(
            env,
            client,
            task_oracle,
            prompt,
            subtask,
            replan_steps,
            frame_callback=recorder.write,
        )
        task_records.append({"task": subtask, "prompt": prompt, "success": bool(success)})
        if not success:
            break
        completed += 1
    recorder.hold_final(
        "FULL CHAIN SUCCESS  5/5" if completed == 5 else "REPLAY COMPLETED  %d/5" % completed,
        completed == 5,
    )
    return completed, task_records


def main(args):
    report, sequences, indices = load_candidates(args)
    selected = [
        {
            "index": index,
            "original_completed": report["completed_tasks_per_sequence"][index],
            "tasks": list(sequences[index][1]),
        }
        for index in indices
    ]
    print(json.dumps({"results": args.results, "selected": selected}, indent=2))
    if args.list_only:
        return

    replan_steps = args.replan_steps or int(report.get("replan_steps", 5))
    if not 1 <= replan_steps <= 10:
        raise ValueError("--replan-steps must be between 1 and 10")
    output_dir = Path(args.output_dir)
    success_dir = output_dir / "successes"
    rejected_dir = output_dir / "rejected"
    success_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    env = calvin_eval.make_calvin_d_env(args.show_gui, args.use_egl)
    task_oracle, annotations = calvin_eval.load_task_assets()
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    manifest = {
        "source_results": str(Path(args.results).resolve()),
        "source_comparison": str(Path(args.comparison).resolve()) if args.comparison else None,
        "source_num_sequences": int(report["num_sequences"]),
        "selection_min_completed": args.min_completed,
        "replan_steps": replan_steps,
        "videos": [],
    }
    try:
        for index in tqdm(indices, desc="Recording successful CALVIN replays"):
            initial_state, tasks = sequences[index]
            for attempt in range(1, args.attempts_per_sequence + 1):
                partial = output_dir / ("sequence_%04d_attempt_%02d.partial.mp4" % (index, attempt))
                recorder = CompositeVideoRecorder(
                    partial, args.fps, index, tasks, args.success_hold_seconds
                )
                try:
                    completed, task_records = replay_sequence(
                        env,
                        client,
                        task_oracle,
                        annotations,
                        initial_state,
                        tasks,
                        replan_steps,
                        recorder,
                    )
                finally:
                    recorder.close()
                verified = completed == 5
                destination_dir = success_dir if verified else rejected_dir
                destination = destination_dir / (
                    "sequence_%04d_attempt_%02d_%dof5.mp4" % (index, attempt, completed)
                )
                partial.rename(destination)
                manifest["videos"].append(
                    {
                        "sequence_index": index,
                        "attempt": attempt,
                        "original_completed": report["completed_tasks_per_sequence"][index],
                        "replay_completed": completed,
                        "verified_full_success": verified,
                        "tasks": task_records,
                        "video": str(destination.resolve()),
                    }
                )
                manifest_path = output_dir / "manifest.json"
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                if verified:
                    break
    finally:
        calvin_eval.close_env(env)

    print("Saved video manifest to %s" % (output_dir / "manifest.json"))


if __name__ == "__main__":
    main(parse_args())
