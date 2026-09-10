"""Compare paired CALVIN results for a fine-tuned and original pi0.5 policy."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finetuned",
        default="outputs/calvin_eval/29999/results_finetuned_1000.json",
    )
    parser.add_argument(
        "--original",
        default="outputs/calvin_eval/pi05_libero/results_original_1000.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/calvin_eval/comparison_finetuned_vs_pi05_libero.json",
    )
    return parser.parse_args()


def load_complete(path):
    path = Path(path)
    report = json.loads(path.read_text())
    completed = report.get("completed_tasks_per_sequence")
    if report.get("status") != "complete":
        raise ValueError("Evaluation is not complete: %s" % path)
    if not isinstance(completed, list) or len(completed) != int(report["num_sequences"]):
        raise ValueError("Invalid per-sequence results: %s" % path)
    return path, report, [int(value) for value in completed]


def compare(args):
    finetuned_path, finetuned, ft = load_complete(args.finetuned)
    original_path, original, base = load_complete(args.original)
    if len(ft) != len(base):
        raise ValueError("The two evaluations use different numbers of sequences")
    if int(finetuned["replan_steps"]) != int(original["replan_steps"]):
        raise ValueError("The two evaluations use different replanning intervals")
    if int(finetuned.get("seed", 0)) != int(original.get("seed", 0)):
        raise ValueError("The two evaluations use different sequence seeds")
    if finetuned.get("observation_profile") != original.get("observation_profile"):
        raise ValueError("The two evaluations use different observation profiles")

    ft_only = [index for index, (a, b) in enumerate(zip(ft, base)) if a == 5 and b < 5]
    base_only = [index for index, (a, b) in enumerate(zip(ft, base)) if a < 5 and b == 5]
    both = [index for index, (a, b) in enumerate(zip(ft, base)) if a == 5 and b == 5]
    neither = [index for index, (a, b) in enumerate(zip(ft, base)) if a < 5 and b < 5]
    chain_comparison = {}
    for depth in range(1, 6):
        key = str(depth)
        ft_rate = float(finetuned["chain_success_rates"][key])
        base_rate = float(original["chain_success_rates"][key])
        chain_comparison[key] = {
            "finetuned": ft_rate,
            "original_pi05_libero": base_rate,
            "delta_percentage_points": (ft_rate - base_rate) * 100.0,
        }

    comparison = {
        "status": "complete",
        "num_sequences": len(ft),
        "replan_steps": int(finetuned["replan_steps"]),
        "seed": int(finetuned.get("seed", 0)),
        "observation_profile": finetuned.get("observation_profile"),
        "finetuned_results": str(finetuned_path.resolve()),
        "original_results": str(original_path.resolve()),
        "average_successful_sequence_length": {
            "finetuned": float(finetuned["average_successful_sequence_length"]),
            "original_pi05_libero": float(original["average_successful_sequence_length"]),
            "delta": float(finetuned["average_successful_sequence_length"])
            - float(original["average_successful_sequence_length"]),
        },
        "chain_success_rates": chain_comparison,
        "paired_full_chain_outcomes": {
            "both_success_count": len(both),
            "finetuned_only_success_count": len(ft_only),
            "original_only_success_count": len(base_only),
            "neither_success_count": len(neither),
        },
        "finetuned_only_full_success_indices": ft_only,
        "both_full_success_indices": both,
        "original_only_full_success_indices": base_only,
        "neither_full_success_indices": neither,
        "per_sequence_completed_delta": [a - b for a, b in zip(ft, base)],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")

    markdown = output.with_suffix(".md")
    lines = [
        "# CALVIN-D: fine-tuned vs original pi0.5-LIBERO",
        "",
        "| Metric | Fine-tuned | Original | Delta |",
        "|---|---:|---:|---:|",
        "| Average completed tasks | %.3f | %.3f | %+.3f |"
        % (
            comparison["average_successful_sequence_length"]["finetuned"],
            comparison["average_successful_sequence_length"]["original_pi05_libero"],
            comparison["average_successful_sequence_length"]["delta"],
        ),
    ]
    for depth in range(1, 6):
        item = chain_comparison[str(depth)]
        lines.append(
            "| Success >= %d tasks | %.1f%% | %.1f%% | %+.1f pp |"
            % (
                depth,
                item["finetuned"] * 100,
                item["original_pi05_libero"] * 100,
                item["delta_percentage_points"],
            )
        )
    lines.extend(
        [
            "",
            "Fine-tuned-only full successes: %d" % len(ft_only),
            "",
            "Indices: `%s`" % ",".join(map(str, ft_only)),
            "",
        ]
    )
    markdown.write_text("\n".join(lines))
    print(json.dumps(comparison, indent=2, sort_keys=True))
    print("Saved comparison to %s and %s" % (output, markdown))


if __name__ == "__main__":
    compare(parse_args())
