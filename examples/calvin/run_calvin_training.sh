#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_NAME="${CONFIG_NAME:-pi05_calvin_lora}"
EXP_NAME="${EXP_NAME:-calvin_abc_d_lora}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/${CONFIG_NAME}/${EXP_NAME}}"
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/train_${RUN_TIMESTAMP}.log"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${REPO_ROOT}/.openpi_cache}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}"

{
  echo "CALVIN pi0.5 training"
  echo "  config:      ${CONFIG_NAME}"
  echo "  experiment:  ${EXP_NAME}"
  echo "  dataset:     full ABC (17,870 episodes)"
  echo "  official eval target: CALVIN environment D"
  echo "  checkpoint:  ${REPO_ROOT}/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
  echo "  log:         ${LOG_FILE}"
  echo "  started:     $(date --iso-8601=seconds)"
  echo "  git commit:  $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
  fi
} | tee -a "${LOG_FILE}"

set +e
python -u scripts/train.py "${CONFIG_NAME}" \
  --exp-name="${EXP_NAME}" \
  "$@" 2>&1 | tee -a "${LOG_FILE}"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

{
  echo "  finished:    $(date --iso-8601=seconds)"
  echo "  exit status: ${TRAIN_STATUS}"
} | tee -a "${LOG_FILE}"

exit "${TRAIN_STATUS}"
