# Fine-tune pi0.5 on CALVIN ABC-D

This setup reads the downloaded LeRobot v3 dataset in place. It does not copy or
re-encode the 2.1 GB of parquet and MP4 data.

Expected dataset location:

```text
data/lerobot_v3/Traly/calvin_abc_d-lerobot
```

The `pi05_calvin_lora` config uses:

- released `pi05_libero` parameters as the initialization;
- LoRA adapters for the PaliGemma and action-expert language models;
- top and wrist RGB cameras;
- the full 15-dimensional CALVIN robot state;
- 10-step chunks of 7-dimensional relative end-effector actions;
- task text from `meta/tasks.parquet`;
- all 17,870 demonstrations from CALVIN environments A/B/C for training;
- batch size 8 for a single 48 GB RTX A6000.

Here, `ABC-D` means **ABC -> D**: the downloaded demonstrations are the ABC
training data, while environment D is the external simulation evaluation target.

## Prepare and validate

```bash
conda activate openpi_env
cd /home/dell/wzm/openpi

export OPENPI_DATA_HOME=/home/dell/wzm/openpi/.openpi_cache

python scripts/prepare_calvin_for_openpi.py
```

This validates all 1,071,743 ABC frames and writes OpenPI normalization
statistics to:

```text
assets/pi05_calvin_lora/Traly/calvin_abc_d-lerobot/norm_stats.json
```

The source dataset already contains global statistics over all ABC state and
relative-action rows. The preparation script converts those statistics and only
decodes three image samples for integrity checks.

## Train

Stop any running policy server first so the training process can use the A6000.

```bash
conda activate openpi_env
cd /home/dell/wzm/openpi

export OPENPI_DATA_HOME=/home/dell/wzm/openpi/.openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

bash examples/calvin/run_calvin_training.sh --overwrite
```

Checkpoints are written to:

```text
checkpoints/pi05_calvin_lora/calvin_abc_d_lora/
```

Every console message, including initialization, step loss, gradient norm,
checkpoint messages, warnings, and failures, is also written to a timestamped
file under:

```text
logs/pi05_calvin_lora/calvin_abc_d_lora/train_YYYYMMDD_HHMMSS.log
```

To continue from the latest checkpoint without deleting it:

```bash
bash examples/calvin/run_calvin_training.sh --resume
```

You can select another experiment/log directory without editing the script:

```bash
EXP_NAME=calvin_abc_lora_v2 \
LOG_DIR=/home/dell/wzm/openpi/logs/calvin_v2 \
bash examples/calvin/run_calvin_training.sh --overwrite
```

The first step includes JAX compilation and can take several minutes. If batch
size 8 is too large on the current system, override it with `--batch-size=4`.

## Important

This change prepares the training path. CALVIN and LIBERO have different state
definitions, camera geometry, control frequency, scenes, and task distributions.
A model fine-tuned with this config should be evaluated with a CALVIN environment
adapter that supplies the same 15-dimensional state and relative action space;
the existing LIBERO client is not a drop-in CALVIN evaluator.

## Evaluate on CALVIN environment D

The final checkpoint is:

```text
checkpoints/pi05_calvin_lora/calvin_abc_d_lora/29999
```

The evaluation uses two processes and two environments. The OpenPI server loads
the JAX checkpoint on the GPU. A separate Python 3.8 CALVIN environment renders
the static and gripper cameras and sends observations through the lightweight
OpenPI client.

The environment has been prepared locally at:

```text
third_party/calvin
examples/calvin/.venv
```

The repository is pinned by its own Git metadata to CALVIN commit
`fa03f01f19c65920e18cf37398a9ce859274af76`, `calvin_env` commit
`1431a46bd36bde5903fb6345e68b5ccc30def666`, and the recursively pinned tacto
submodule. The lightweight Python 3.8 environment contains PyBullet, the CALVIN
simulator, task oracle, Hydra configs, and the OpenPI WebSocket client. It does
not duplicate JAX, Torch, or the model checkpoint.

The evaluation client composes CALVIN's environment-D config directly, so no
CALVIN demonstration archive is required to run the official generated
1,000-sequence protocol.

The simulator-only check does not use a GPU or policy server:

```bash
cd /home/dell/wzm/openpi
bash examples/calvin/run_calvin_eval.sh --check-environment
```

When the GPU is available, start the policy server in terminal 1:

```bash
conda activate openpi_env
cd /home/dell/wzm/openpi
bash examples/calvin/run_calvin_policy_server.sh
```

Run a one-sequence smoke test from terminal 2:

```bash
cd /home/dell/wzm/openpi
bash examples/calvin/run_calvin_eval.sh \
  --host 127.0.0.1 \
  --num-sequences 1 \
  --show-gui
```

Then run 10 headless sequences as a sanity check:

```bash
bash examples/calvin/run_calvin_eval.sh --num-sequences 10
```

For the official protocol, run all 1,000 five-instruction chains:

```bash
bash examples/calvin/run_calvin_full_eval.sh
```

The full runner saves after every sequence to:

```text
outputs/calvin_eval/29999/results_finetuned_1000.json
```

If it is interrupted, restart the policy server and continue without repeating
completed sequence indices:

```bash
bash examples/calvin/run_calvin_full_eval.sh --resume
```

### Paired 500-sequence comparison with original pi0.5-LIBERO

The appropriate original baseline is the released `pi05_libero` checkpoint,
because it is the exact Franka-compatible checkpoint used to initialize this
CALVIN LoRA run. Evaluate the two models sequentially so they do not compete
for GPU memory.

For a controlled before/after comparison, both runs below use the first 500
official D-scene chains generated with seed 0, replanning every 5 steps, the
same CALVIN camera/state adapter, and the CALVIN action normalization stored in
checkpoint `29999`. The original run changes only the model weights back to
the released `pi05_libero` checkpoint.

Terminal 1, start the fine-tuned server:

```bash
conda activate openpi_env
cd /home/dell/wzm/openpi
bash examples/calvin/run_calvin_policy_server.sh
```

Terminal 2, evaluate 500 sequences:

```bash
cd /home/dell/wzm/openpi
bash examples/calvin/run_calvin_finetuned_500.sh
```

If interrupted, repeat the command with `--resume`. After it completes, stop
terminal 1 with `Ctrl-C`. Then start the controlled original model server:

```bash
conda activate openpi_env
cd /home/dell/wzm/openpi
bash examples/calvin/run_pi05_libero_calvin_policy_server.sh
```

In terminal 2, evaluate the same 500 sequences:

```bash
cd /home/dell/wzm/openpi
bash examples/calvin/run_calvin_pi05_libero_500.sh
```

This command also supports `--resume`. When both runs are complete, generate
the paired comparison:

```bash
bash examples/calvin/run_calvin_compare_500.sh
```

The raw and comparison reports are:

```text
outputs/calvin_eval/29999/results_finetuned_500.json
outputs/calvin_eval/pi05_libero/results_original_500.json
outputs/calvin_eval/comparison_finetuned_vs_pi05_libero_500.json
outputs/calvin_eval/comparison_finetuned_vs_pi05_libero_500.md
```

It reports the average completed task count, success rates for completing at
least 1 through 5 tasks, percentage-point improvements, and paired indices for
both-success, fine-tuned-only success, original-only success, and neither.

The report contains success rates for completing at least 1 through 5
instructions, average successful sequence length, and per-task success counts.
The default replanning interval is 5 environment steps; keep it fixed when
comparing checkpoints.

Do not use `run_pi05_libero_policy_server.sh` for this controlled comparison:
that older runner loads native LIBERO normalization and pairs with the 8-D
LIBERO observation profile. It remains available only for a separate
cross-domain/native-preprocessing reference, not for isolating the effect of
CALVIN fine-tuning.

### Select successful sequences and record MP4 files

Inspect the full-success candidates without starting a simulator or server:

```bash
bash examples/calvin/run_calvin_record_successes.sh \
  --results outputs/calvin_eval/29999/results_finetuned_500.json \
  --comparison outputs/calvin_eval/comparison_finetuned_vs_pi05_libero_500.json \
  --list-only \
  --limit 0
```

After starting the policy server, replay and record the first five candidates:

```bash
bash examples/calvin/run_calvin_record_successes.sh \
  --results outputs/calvin_eval/29999/results_finetuned_500.json \
  --comparison outputs/calvin_eval/comparison_finetuned_vs_pi05_libero_500.json \
  --limit 5 \
  --attempts-per-sequence 3 \
  --output-dir outputs/calvin_eval/29999/videos
```

The script rechecks every replay with the official task oracle. Replays that
again complete all five tasks are written under `videos/successes`; unsuccessful
reruns are retained under `videos/rejected`. `videos/manifest.json` records the
source sequence index, tasks, prompts, replay score, and MP4 path. Each video is
a 1280x720 composite of the static view, wrist camera, task name, instruction,
and live success status.
