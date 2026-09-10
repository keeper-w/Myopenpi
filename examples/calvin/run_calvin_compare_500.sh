#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
exec bash examples/calvin/run_calvin_compare.sh \
  --finetuned outputs/calvin_eval/29999/results_finetuned_500.json \
  --original outputs/calvin_eval/pi05_libero/results_original_500.json \
  --output outputs/calvin_eval/comparison_finetuned_vs_pi05_libero_500.json \
  "$@"
