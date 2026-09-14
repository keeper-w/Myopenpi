#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS="${CALVIN_ORIGINAL_RESULTS:-${REPO_ROOT}/artifacts/calvin_d_500/results_original_500.json}"
COMPARISON="${CALVIN_COMPARISON:-${REPO_ROOT}/artifacts/calvin_d_500/comparison_finetuned_vs_pi05_libero_500.json}"
OUTPUT_DIR="${CALVIN_ORIGINAL_FAILURE_VIDEO_DIR:-${REPO_ROOT}/artifacts/calvin_d_500/videos/original_failures}"
SEQUENCE_INDICES="${CALVIN_ORIGINAL_SEQUENCE_INDICES:-0,16,17,20}"

cd "${REPO_ROOT}"
exec bash examples/calvin/run_calvin_record_successes.sh \
  --results "${RESULTS}" \
  --comparison "${COMPARISON}" \
  --output-dir "${OUTPUT_DIR}" \
  --sequence-indices "${SEQUENCE_INDICES}" \
  --min-completed 0 \
  --max-completed 0 \
  --limit 4 \
  "$@"
