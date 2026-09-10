#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

cd "${REPO_ROOT}"
exec "${PYTHON}" examples/calvin/compare_calvin_results.py "$@"
