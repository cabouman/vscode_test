#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
REQ_FILE="${ROOT_DIR}/camera_model/requirements.txt"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is not installed or not on PATH."
  exit 1
fi

echo "Project root: ${ROOT_DIR}"
echo "Creating virtual environment in ${VENV_DIR} (if missing)..."
python3 -m venv "${VENV_DIR}"

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing dependencies from ${REQ_FILE}..."
python -m pip install -r "${REQ_FILE}"

echo
echo "Install complete."
echo "To run the app:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  streamlit run ${ROOT_DIR}/camera_model/app.py"
