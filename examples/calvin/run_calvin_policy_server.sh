#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CHECKPOINT="${CALVIN_CHECKPOINT:-${REPO_ROOT}/checkpoints/pi05_calvin_lora/calvin_abc_d_lora/29999}"

if [[ ! -d "${CHECKPOINT}/params" ]]; then
  echo "Invalid CALVIN checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${REPO_ROOT}/.openpi_cache}"

exec python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_calvin_lora \
  --policy.dir="${CHECKPOINT}" \
  "$@"
