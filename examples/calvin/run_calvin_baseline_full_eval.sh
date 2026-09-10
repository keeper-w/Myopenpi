#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT="${CALVIN_BASELINE_RESULTS:-${REPO_ROOT}/outputs/calvin_eval/pi05_libero/results_original_1000.json}"

cd "${REPO_ROOT}"
exec bash examples/calvin/run_calvin_eval.sh \
  --num-sequences 1000 \
  --replan-steps 5 \
  --observation-profile pi05_libero \
  --save-every 1 \
  --output "${OUTPUT}" \
  "$@"
