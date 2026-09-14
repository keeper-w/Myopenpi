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

The MP4 files below are replay-verified full successes from an earlier
100-sequence fine-tuned-policy run. They are included as visual demonstrations;
they are not claimed to be the same successful indices as the 500-sequence
aggregate report. The 500-sequence aggregate result remains 84/500 full
successes, as recorded in the JSON report.

## Paired videos from the 100-sequence showcase run

| Sequence index | Tasks | Fine-tuned success | Original failure |
|---:|---|---|---|
| 0 | lift blue block from slider → place in slider → turn on lightbulb → open drawer → push pink block left | [MP4](videos/success_sequence_0000.mp4) | [MP4](videos/original_failures_100/rejected/sequence_0000_attempt_01_0of5.mp4) |
| 16 | lift red block from slider → place in drawer → move slider left → turn on LED → close drawer | [MP4](videos/success_sequence_0016.mp4) | [MP4](videos/original_failures_100/rejected/sequence_0016_attempt_01_0of5.mp4) |
| 17 | push red block left → turn on lightbulb → open drawer → lift blue block from table → place in drawer | [MP4](videos/success_sequence_0017.mp4) | [MP4](videos/original_failures_100/rejected/sequence_0017_attempt_01_0of5.mp4) |
| 20 | lift red block from slider → place in drawer → turn on LED → move slider right → lift red block from drawer | [MP4](videos/success_sequence_0020.mp4) | [MP4](videos/original_failures_100/rejected/sequence_0020_attempt_01_0of5.mp4) |

The paired original-model failures are under
`videos/original_failures_100/rejected/`. These videos use the same 100-sequence
generation protocol and the same sequence indices as the four showcase success
videos.

The source JSON and Markdown reports are included in this directory:

- `comparison_finetuned_vs_pi05_libero_500.json`
- `comparison_finetuned_vs_pi05_libero_500.md`
- `results_finetuned_500.json`
- `results_original_500.json`

The full source code and reproduction commands are documented in
`examples/calvin/README_FINETUNE.md`.
