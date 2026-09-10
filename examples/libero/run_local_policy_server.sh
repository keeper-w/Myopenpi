#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

SERVER_PYTHON="${OPENPI_SERVER_PYTHON:-python}"
CHECKPOINT="${OPENPI_LIBERO_CHECKPOINT:-gs://openpi-assets/checkpoints/pi05_libero}"
DATA_HOME="${OPENPI_DATA_HOME:-${REPO_ROOT}/.openpi_cache}"

if ! "${SERVER_PYTHON}" -c "import jax; import openpi; import lerobot.common.datasets.lerobot_dataset" >/dev/null 2>&1; then
    echo "The policy-server environment is incomplete." >&2
    echo "Activate openpi_env and install the LeRobot revision pinned by this repository." >&2
    echo "See examples/libero/README_LOCAL_FRANKA.md for the exact command." >&2
    exit 1
fi

if [[ "${CHECKPOINT}" != gs://* ]]; then
    if [[ ! -d "${CHECKPOINT}" ]]; then
        echo "Checkpoint directory does not exist: ${CHECKPOINT}" >&2
        exit 1
    fi
    if [[ ! -d "${CHECKPOINT}/params" && ! -f "${CHECKPOINT}/model.safetensors" ]]; then
        echo "Checkpoint is missing params/ or model.safetensors: ${CHECKPOINT}" >&2
        exit 1
    fi
    if ! find "${CHECKPOINT}/assets" -name norm_stats.json -print -quit 2>/dev/null | grep -q .; then
        echo "Checkpoint is missing assets/**/norm_stats.json: ${CHECKPOINT}" >&2
        exit 1
    fi
fi

mkdir -p "${DATA_HOME}"
cd "${REPO_ROOT}"

echo "Starting pi0.5-LIBERO policy server"
echo "  checkpoint: ${CHECKPOINT}"
echo "  cache:      ${DATA_HOME}"
echo "  address:    0.0.0.0:${OPENPI_POLICY_PORT:-8000}"

exec env OPENPI_DATA_HOME="${DATA_HOME}" "${SERVER_PYTHON}" -u scripts/serve_policy.py \
    --port "${OPENPI_POLICY_PORT:-8000}" \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "${CHECKPOINT}" \
    "$@"
