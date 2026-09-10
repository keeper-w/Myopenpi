#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CHECKPOINT="${PI05_LIBERO_CHECKPOINT:-${REPO_ROOT}/.openpi_cache/openpi-assets/checkpoints/pi05_libero}"

if [[ ! -d "${CHECKPOINT}/params" ]]; then
  echo "Missing original pi05-LIBERO checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${REPO_ROOT}/.openpi_cache}"

exec python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir="${CHECKPOINT}" \
  "$@"
