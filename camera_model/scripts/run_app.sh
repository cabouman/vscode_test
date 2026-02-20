#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
APP_FILE="${ROOT_DIR}/camera_model/app.py"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Virtual environment not found at ${VENV_DIR}."
  echo "Run install first:"
  echo "  bash ${ROOT_DIR}/camera_model/scripts/install.sh"
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
exec streamlit run "${APP_FILE}"
