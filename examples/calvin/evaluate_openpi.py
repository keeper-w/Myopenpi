"""Evaluate an OpenPI policy in the official CALVIN environment-D protocol.

This script belongs in the CALVIN (Python 3.8) client environment.  The OpenPI
model runs separately behind ``scripts/serve_policy.py``.
"""

import argparse
import collections
import contextlib
import json
import logging
from math import pi
from pathlib import Path
import sys
import types

import calvin_agent
import calvin_env
import hydra
import numpy as np
from omegaconf import OmegaConf
from openpi_client import websocket_client_policy
import pyhash
from tqdm import tqdm


@contextlib.contextmanager
def temp_seed(seed):
    """Match CALVIN's temporary NumPy seed helper without importing its Torch models."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


# multistep_sequences only needs temp_seed from evaluation.utils. Supplying that
# small interface avoids importing CALVIN's MCIL/PyTorch training stack into the
# lightweight simulator client.
_lightweight_utils = types.ModuleType("calvin_agent.evaluation.utils")
_lightweight_utils.temp_seed = temp_seed
sys.modules["calvin_agent.evaluation.utils"] = _lightweight_utils

from calvin_agent.evaluation.multistep_sequences import get_sequences  # noqa: E402


EP_LEN = 360
_FNV1_32 = pyhash.fnv1_32()


def get_env_state_for_initial_condition(initial_condition):
    """Official CALVIN reset-state construction, kept free of Torch imports."""
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    seed = _FNV1_32(str(initial_condition.values()))
    with temp_seed(seed):
        np.random.shuffle(block_table)
        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)
    return robot_obs, scene_obs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--num-sequences", default=10, type=int)
    parser.add_argument(
        "--replan-steps",
        default=5,
        type=int,
        help="Execute this many actions from each predicted 10-step chunk.",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--observation-profile",
        choices=("calvin", "pi05_libero"),
        default="calvin",
        help="Use 15-D CALVIN state for the fine-tuned model or native 8-D LIBERO state for the original baseline.",
    )
    parser.add_argument("--show-gui", action="store_true")
    parser.add_argument("--use-egl", action="store_true")
    parser.add_argument(
        "--check-environment",
        action="store_true",
        help="Initialize CALVIN D and validate observations without connecting to a policy server.",
    )
    parser.add_argument("--save-every", default=1, type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run from --output instead of starting at sequence zero.",
    )
    parser.add_argument("--output", default="outputs/calvin_eval/29999/results.json")
    return parser.parse_args()


def make_calvin_d_env(show_gui=False, use_egl=False):
    """Compose the official static+gripper camera environment-D config."""
    conf_dir = Path(calvin_env.__file__).resolve().parents[1] / "conf"
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=str(conf_dir), job_name="openpi_calvin_eval"):
        cfg = hydra.compose(
            config_name="config_data_collection.yaml",
            overrides=["cameras=static_and_gripper", "scene=calvin_scene_D"],
        )
    return hydra.utils.instantiate(
        cfg.env,
        show_gui=show_gui,
        use_egl=use_egl,
        use_vr=False,
        use_scene_info=True,
    )


def load_task_assets():
    conf_dir = Path(calvin_agent.__file__).resolve().parents[1] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    return task_oracle, annotations


def model_observation(obs, prompt, observation_profile="calvin"):
    rgb = obs["rgb_obs"]
    static = np.ascontiguousarray(rgb["rgb_static"], dtype=np.uint8)
    gripper = np.ascontiguousarray(rgb["rgb_gripper"], dtype=np.uint8)
    calvin_state = np.ascontiguousarray(obs["robot_obs"], dtype=np.float32)
    if static.shape != (200, 200, 3):
        raise ValueError("Expected CALVIN static RGB shape (200, 200, 3), got %s" % (static.shape,))
    if gripper.shape != (84, 84, 3):
        raise ValueError("Expected CALVIN gripper RGB shape (84, 84, 3), got %s" % (gripper.shape,))
    if calvin_state.shape != (15,):
        raise ValueError("Expected CALVIN robot state shape (15,), got %s" % (calvin_state.shape,))
    if observation_profile == "calvin":
        state = calvin_state
    elif observation_profile == "pi05_libero":
        # Native pi05-LIBERO input: EE xyz + Euler xyz + two finger positions.
        # CALVIN reports total gripper opening width, so split it equally across fingers.
        state = np.concatenate((calvin_state[:6], np.repeat(calvin_state[6] / 2.0, 2))).astype(np.float32)
    else:
        raise ValueError("Unknown observation profile %r" % observation_profile)
    return {
        "observation/image": static,
        "observation/wrist_image": gripper,
        "observation/state": state,
        "prompt": str(prompt),
    }


def sanitize_action(action):
    action = np.asarray(action, dtype=np.float32).copy()
    if action.shape != (7,):
        raise ValueError("Expected a 7-D CALVIN action, got %s" % (action.shape,))
    action[:6] = np.clip(action[:6], -1.0, 1.0)
    action[6] = 1.0 if action[6] >= 0.0 else -1.0
    return action


def rollout_subtask(
    env,
    client,
    task_oracle,
    prompt,
    subtask,
    replan_steps,
    frame_callback=None,
    observation_profile="calvin",
):
    start_info = env.get_info()
    obs = env.get_obs()
    action_plan = collections.deque()
    for step in range(EP_LEN):
        if not action_plan:
            prediction = client.infer(model_observation(obs, prompt, observation_profile))
            chunk = np.asarray(prediction["actions"])
            if chunk.ndim != 2 or chunk.shape[1] < 7 or len(chunk) < replan_steps:
                raise ValueError("Unexpected action chunk shape %s" % (chunk.shape,))
            action_plan.extend(chunk[:replan_steps, :7])
        action = sanitize_action(action_plan.popleft())
        obs, _, _, current_info = env.step(action)
        completed = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if frame_callback is not None:
            frame_callback(obs, prompt, subtask, step, bool(completed))
        if completed:
            return True
    return False


def chain_success_rates(results):
    return [sum(value >= length for value in results) / len(results) for length in range(1, 6)]


def close_env(env):
    """Close once; the pinned CALVIN destructor otherwise disconnects twice."""
    env.close()
    env.close = lambda: None


def build_report(
    results,
    per_task,
    sequence_records,
    num_sequences,
    replan_steps,
    seed,
    status,
    observation_profile,
):
    rates = chain_success_rates(results) if results else [0.0] * 5
    success_indices_by_depth = {
        str(depth): [index for index, completed in enumerate(results) if completed >= depth]
        for depth in range(1, 6)
    }
    return {
        "status": status,
        "num_sequences": num_sequences,
        "evaluated_sequences": len(results),
        "replan_steps": replan_steps,
        "seed": seed,
        "observation_profile": observation_profile,
        "average_successful_sequence_length": float(np.mean(results)) if results else 0.0,
        "chain_success_rates": {str(i): rates[i - 1] for i in range(1, 6)},
        "full_success_indices": success_indices_by_depth["5"],
        "success_indices_by_depth": success_indices_by_depth,
        "per_task": dict(sorted(per_task.items())),
        "completed_tasks_per_sequence": results,
        "sequence_records": sequence_records,
    }


def save_report(output, report):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def evaluate(args):
    if not 1 <= args.num_sequences <= 1000:
        raise ValueError("--num-sequences must be between 1 and 1000")
    if not 1 <= args.replan_steps <= 10:
        raise ValueError("--replan-steps must be between 1 and the trained action horizon (10)")
    if args.save_every < 1:
        raise ValueError("--save-every must be at least 1")

    np.random.seed(args.seed)
    sequences = get_sequences(args.num_sequences, num_workers=1)

    if args.check_environment:
        env = make_calvin_d_env(args.show_gui, args.use_egl)
        task_oracle, annotations = load_task_assets()
        sequence = get_sequences(1, num_workers=1)[0]
        robot_obs, scene_obs = get_env_state_for_initial_condition(sequence[0])
        obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        sample = model_observation(obs, annotations[sequence[1][0]][0], args.observation_profile)
        env.step(np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32))
        print(
            json.dumps(
                {
                    "environment": "CALVIN D",
                    "static_rgb": list(sample["observation/image"].shape),
                    "gripper_rgb": list(sample["observation/wrist_image"].shape),
                    "robot_state": list(sample["observation/state"].shape),
                    "observation_profile": args.observation_profile,
                    "task_oracle": type(task_oracle).__name__,
                    "first_chain": list(sequence[1]),
                    "status": "OK",
                },
                indent=2,
            )
        )
        close_env(env)
        return

    results = []
    per_task = collections.defaultdict(lambda: {"success": 0, "total": 0})
    sequence_records = []
    output = Path(args.output)
    if args.resume and output.exists():
        previous = json.loads(output.read_text())
        if int(previous.get("num_sequences", -1)) != args.num_sequences:
            raise ValueError("Cannot resume: --num-sequences differs from the saved report")
        if int(previous.get("replan_steps", -1)) != args.replan_steps:
            raise ValueError("Cannot resume: --replan-steps differs from the saved report")
        if int(previous.get("seed", args.seed)) != args.seed:
            raise ValueError("Cannot resume: --seed differs from the saved report")
        if previous.get("observation_profile", args.observation_profile) != args.observation_profile:
            raise ValueError("Cannot resume: --observation-profile differs from the saved report")
        results = [int(value) for value in previous.get("completed_tasks_per_sequence", [])]
        for task, counts in previous.get("per_task", {}).items():
            per_task[task] = {"success": int(counts["success"]), "total": int(counts["total"])}
        sequence_records = list(previous.get("sequence_records", []))
        if not sequence_records and results:
            sequence_records = [
                {
                    "index": index,
                    "tasks": [str(task) for task in sequences[index][1]],
                    "completed": completed,
                    "full_success": completed == 5,
                }
                for index, completed in enumerate(results)
            ]
        if len(results) == args.num_sequences:
            final_report = build_report(
                results,
                per_task,
                sequence_records,
                args.num_sequences,
                args.replan_steps,
                args.seed,
                "complete",
                args.observation_profile,
            )
            save_report(output, final_report)
            print("Evaluation is already complete: %s" % output)
            return

    env = make_calvin_d_env(args.show_gui, args.use_egl)
    task_oracle, annotations = load_task_assets()
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    start_index = len(results)
    progress = tqdm(
        range(start_index, len(sequences)),
        initial=start_index,
        total=len(sequences),
        desc="CALVIN D",
    )
    try:
        for sequence_index in progress:
            initial_state, sequence = sequences[sequence_index]
            robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            completed_count = 0
            for subtask in sequence:
                prompt = annotations[subtask][0]
                success = rollout_subtask(
                    env,
                    client,
                    task_oracle,
                    prompt,
                    subtask,
                    args.replan_steps,
                    observation_profile=args.observation_profile,
                )
                per_task[subtask]["total"] += 1
                per_task[subtask]["success"] += int(success)
                if not success:
                    break
                completed_count += 1
            results.append(completed_count)
            sequence_records.append(
                {
                    "index": sequence_index,
                    "tasks": [str(task) for task in sequence],
                    "completed": completed_count,
                    "full_success": completed_count == 5,
                }
            )
            rates = chain_success_rates(results)
            progress.set_postfix(avg="%.3f" % np.mean(results), sr5="%.1f%%" % (100 * rates[4]))
            if len(results) % args.save_every == 0:
                save_report(
                    output,
                    build_report(
                        results,
                        per_task,
                        sequence_records,
                        args.num_sequences,
                        args.replan_steps,
                        args.seed,
                        "running",
                        args.observation_profile,
                    ),
                )
    finally:
        close_env(env)

    report = build_report(
        results,
        per_task,
        sequence_records,
        args.num_sequences,
        args.replan_steps,
        args.seed,
        "complete",
        args.observation_profile,
    )
    save_report(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Saved results to %s" % output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate(parse_args())
