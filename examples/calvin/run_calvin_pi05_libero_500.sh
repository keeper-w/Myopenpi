#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT="${CALVIN_PI05_LIBERO_500_RESULTS:-${REPO_ROOT}/outputs/calvin_eval/pi05_libero/results_original_500.json}"

cd "${REPO_ROOT}"
exec bash examples/calvin/run_calvin_eval.sh \
  --num-sequences 500 \
  --replan-steps 5 \
  --seed 0 \
  --observation-profile calvin \
  --save-every 1 \
  --output "${OUTPUT}" \
  "$@"
