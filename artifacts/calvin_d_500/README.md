# CALVIN-D 500-sequence evaluation artifacts

This directory contains the shareable results from the paired CALVIN-D
evaluation. It does not contain model checkpoints, the CALVIN dataset, Python
environments, or simulator source code.

## Experimental setup

- 500 official CALVIN-D five-task sequences, generated with `seed=0`
- `replan_steps=5`
- Same CALVIN camera/state/action preprocessing for both policies
- Fine-tuned checkpoint: `pi05_calvin_lora/calvin_abc_d_lora/29999`
- Baseline checkpoint: released `pi05_libero`
- Fine-tuned model: 30,000-step LoRA SFT on CALVIN ABC demonstrations

## Results

| Metric | Fine-tuned | Original pi05_libero | Improvement |
|---|---:|---:|---:|
| Average completed tasks | 1.926 | 0.002 | +1.924 |
| Success >= 1 task | 71.6% | 0.2% | +71.4 pp |
| Success >= 2 tasks | 48.0% | 0.0% | +48.0 pp |
| Success >= 3 tasks | 32.6% | 0.0% | +32.6 pp |
| Success >= 4 tasks | 23.6% | 0.0% | +23.6 pp |
| Success = 5 tasks | 16.8% (84/500) | 0.0% (0/500) | +16.8 pp |

The MP4 files below are replay-verified full successes from the fine-tuned
policy. They are a curated subset of the 84 full successes in the 500-sequence
evaluation, not an exhaustive export.

## Videos

| Sequence index | Tasks | Video |
|---:|---|---|
| 0 | lift blue block from slider → place in slider → turn on lightbulb → open drawer → push pink block left | `success_sequence_0000.mp4` |
| 16 | lift red block from slider → place in drawer → move slider left → turn on LED → close drawer | `success_sequence_0016.mp4` |
| 17 | push red block left → turn on lightbulb → open drawer → lift blue block from table → place in drawer | `success_sequence_0017.mp4` |
| 20 | lift red block from slider → place in drawer → turn on LED → move slider right → lift red block from drawer | `success_sequence_0020.mp4` |

The source JSON and Markdown reports are included alongside this directory in
the repository's `outputs/`-independent artifact package:

- `comparison_finetuned_vs_pi05_libero_500.json`
- `comparison_finetuned_vs_pi05_libero_500.md`
- `results_finetuned_500.json`
- `results_original_500.json`

The full source code and reproduction commands are documented in
`examples/calvin/README_FINETUNE.md`.
