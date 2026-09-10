#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LIBERO_ROOT="${REPO_ROOT}/third_party/libero"
CLIENT_PYTHON="${OPENPI_LIBERO_PYTHON:-python}"
CONFIG_DIR="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/data/libero/config}"
DATASETS_DIR="${LIBERO_DATASETS_DIR:-${REPO_ROOT}/data/libero/datasets}"

if [[ ! -d "${LIBERO_ROOT}/libero/libero" ]]; then
    echo "LIBERO submodule is not initialized at ${LIBERO_ROOT}." >&2
    echo "Run: git submodule update --init third_party/libero" >&2
    exit 1
fi

mkdir -p "${CONFIG_DIR}" "${DATASETS_DIR}" "${REPO_ROOT}/data/libero/videos"
printf '%s\n' \
    "benchmark_root: ${LIBERO_ROOT}/libero/libero" \
    "bddl_files: ${LIBERO_ROOT}/libero/libero/bddl_files" \
    "init_states: ${LIBERO_ROOT}/libero/libero/init_files" \
    "datasets: ${DATASETS_DIR}" \
    "assets: ${LIBERO_ROOT}/libero/libero/assets" \
    >"${CONFIG_DIR}/config.yaml"

export LIBERO_CONFIG_PATH="${CONFIG_DIR}"
export PYTHONPATH="${REPO_ROOT}/packages/openpi-client/src:${LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if ! "${CLIENT_PYTHON}" -c "import libero; import mujoco; import openpi_client; import robosuite" >/dev/null 2>&1; then
    echo "The LIBERO client environment is incomplete." >&2
    echo "Activate openpi_libero and follow examples/libero/README_LOCAL_FRANKA.md." >&2
    exit 1
fi

cd "${REPO_ROOT}"

echo "Starting Franka LIBERO client"
echo "  suite:      ${LIBERO_TASK_SUITE:-libero_spatial}"
echo "  trials:     ${LIBERO_NUM_TRIALS:-1} per task"
echo "  policy:     ${OPENPI_POLICY_HOST:-127.0.0.1}:${OPENPI_POLICY_PORT:-8000}"
echo "  renderer:   ${MUJOCO_GL}"

exec "${CLIENT_PYTHON}" examples/libero/main.py \
    --args.host "${OPENPI_POLICY_HOST:-127.0.0.1}" \
    --args.port "${OPENPI_POLICY_PORT:-8000}" \
    --args.task-suite-name "${LIBERO_TASK_SUITE:-libero_spatial}" \
    --args.num-trials-per-task "${LIBERO_NUM_TRIALS:-1}" \
    "$@"
