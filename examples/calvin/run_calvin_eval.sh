#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing CALVIN evaluation environment: ${SCRIPT_DIR}/.venv" >&2
  exit 1
fi

cd "${REPO_ROOT}"
exec "${PYTHON}" examples/calvin/evaluate_openpi.py "$@"
