#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CHECKPOINT="${PI05_LIBERO_CHECKPOINT:-${REPO_ROOT}/.openpi_cache/openpi-assets/checkpoints/pi05_libero}"
NORM_CHECKPOINT="${CALVIN_CHECKPOINT:-${REPO_ROOT}/checkpoints/pi05_calvin_lora/calvin_abc_d_lora/29999}"

cd "${REPO_ROOT}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${REPO_ROOT}/.openpi_cache}"

exec python examples/calvin/serve_pi05_libero_calvin.py \
  --checkpoint "${CHECKPOINT}" \
  --calvin-norm-checkpoint "${NORM_CHECKPOINT}" \
  "$@"
