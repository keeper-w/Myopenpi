# OpenPI pi0.5 CALVIN configuration bundle

This archive contains only the local OpenPI/CALVIN integration code and launch
configuration. It intentionally excludes model checkpoints, datasets, Python
environments, caches, CALVIN source code, logs, evaluation results, and videos.

## Expected base repository

Apply this bundle to an OpenPI checkout based on commit:

```text
215abfb217dbac7d5f1273282331b9b1866c0479
```

The archive contains complete files, not a Git patch. Back up local changes on
the destination machine before extracting it.

## Install

From the OpenPI repository root on the destination machine:

```bash
tar -xzf openpi_calvin_config_bundle_20260905.tar.gz
```

The expected external paths are:

```text
data/lerobot_v3/Traly/calvin_abc_d-lerobot/
checkpoints/pi05_calvin_lora/calvin_abc_d_lora/29999/
third_party/calvin/
```

For comparison against the original model, also provide:

```text
.openpi_cache/openpi-assets/checkpoints/pi05_libero/
```

Paths can be changed with the environment variables documented in the runner
scripts. The fine-tuned checkpoint already contains its own normalization
statistics under `29999/assets/` for inference. This bundle also includes the
project-level `assets/pi05_calvin_lora/.../norm_stats.json` copy needed by the
training data loader.

## What this bundle configures

The `pi05_calvin_lora` training configuration uses:

```text
initial checkpoint: pi05_libero
model: pi0.5
PaliGemma: gemma_2b_lora (rank 16, alpha 16)
action expert: gemma_300m_lora (rank 32, alpha 32)
action horizon: 10
internal action dimension: 32
CALVIN action dimension: 7 relative end-effector controls
discrete_state_input: false
batch size: 8
data workers: 4
training steps: 30000
EMA: disabled
```

The CALVIN reader consumes top RGB, wrist RGB, 15-D robot state, language
prompt, and 7-D relative actions from the local LeRobot v3 dataset. With the
current `discrete_state_input=false` setting, robot state is carried through the
pipeline but is not an effective conditioning input to pi0.5. Do not change
this flag only at evaluation time for the existing checkpoint.

## Smoke checks

Check the simulator without starting a policy server:

```bash
bash examples/calvin/run_calvin_eval.sh --check-environment
```

Start the fine-tuned policy server:

```bash
conda activate openpi_env
bash examples/calvin/run_calvin_policy_server.sh
```

In the CALVIN client terminal, run one sequence:

```bash
bash examples/calvin/run_calvin_eval.sh --num-sequences 1
```

Run or resume the controlled 500-sequence fine-tuned evaluation:

```bash
bash examples/calvin/run_calvin_finetuned_500.sh
bash examples/calvin/run_calvin_finetuned_500.sh --resume
```

Start and evaluate the original pi0.5-LIBERO baseline after stopping the
fine-tuned server. This controlled server reuses the checkpoint `29999` CALVIN
normalization while loading the original `pi05_libero` weights:

```bash
bash examples/calvin/run_pi05_libero_calvin_policy_server.sh
bash examples/calvin/run_calvin_pi05_libero_500.sh
```

Compare the two completed 500-sequence reports:

```bash
bash examples/calvin/run_calvin_compare_500.sh
```

List fine-tuned-only full-success video candidates:

```bash
bash examples/calvin/run_calvin_record_successes.sh \
  --results outputs/calvin_eval/29999/results_finetuned_500.json \
  --comparison outputs/calvin_eval/comparison_finetuned_vs_pi05_libero_500.json \
  --list-only --limit 0
```

## Training

Validate the downloaded dataset and prepare normalization assets:

```bash
conda activate openpi_env
python scripts/prepare_calvin_for_openpi.py
```

Start a new training run:

```bash
bash examples/calvin/run_calvin_training.sh --overwrite
```

To continue a checkpoint at step 29999, preserve the full numbered checkpoint,
including `train_state/`, and set a new total greater than 30000:

```bash
bash examples/calvin/run_calvin_training.sh \
  --resume \
  --num-train-steps=40000
```

## Included files

```text
CALVIN_CONFIG_BUNDLE_README.md
src/openpi/training/config.py
src/openpi/training/data_loader.py
src/openpi/training/calvin_v3_dataset.py
scripts/compute_norm_stats.py
scripts/prepare_calvin_for_openpi.py
assets/pi05_calvin_lora/Traly/calvin_abc_d-lerobot/norm_stats.json
examples/calvin/README_FINETUNE.md
examples/calvin/compare_calvin_results.py
examples/calvin/evaluate_openpi.py
examples/calvin/record_successful_rollouts.py
examples/calvin/requirements_eval.txt
examples/calvin/run_calvin_baseline_full_eval.sh
examples/calvin/run_calvin_compare.sh
examples/calvin/run_calvin_compare_500.sh
examples/calvin/run_calvin_eval.sh
examples/calvin/run_calvin_finetuned_500.sh
examples/calvin/run_calvin_full_eval.sh
examples/calvin/run_calvin_policy_server.sh
examples/calvin/run_calvin_pi05_libero_500.sh
examples/calvin/run_calvin_record_successes.sh
examples/calvin/run_calvin_training.sh
examples/calvin/run_pi05_libero_calvin_policy_server.sh
examples/calvin/run_pi05_libero_policy_server.sh
examples/calvin/serve_pi05_libero_calvin.py
```
