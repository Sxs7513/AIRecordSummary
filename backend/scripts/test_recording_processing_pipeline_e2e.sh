#!/usr/bin/env bash

set -euo pipefail

BACKEND_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
ROOT_DIR="$(CDPATH= cd -- "${BACKEND_DIR}/.." && pwd)"
PYTHON_BIN="${BACKEND_DIR}/.venv/bin/python"

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [audio-file]" >&2
  exit 2
fi

INPUT_AUDIO_FILE="${1:-${BACKEND_DIR}/tests/files/test1.mp3}"
AUDIO_FILE="$(cd "$(dirname -- "${INPUT_AUDIO_FILE}")" && pwd)/$(basename -- "${INPUT_AUDIO_FILE}")"
if [[ ! -f "${AUDIO_FILE}" ]]; then
  echo "Audio file does not exist: ${AUDIO_FILE}" >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Backend Python environment was not found: ${PYTHON_BIN}" >&2
  echo "Run scripts/install_audio_dependencies.sh first." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 14) else 1)'; then
  echo "${PYTHON_BIN} must use Python 3.14. Recreate backend/.venv with Python 3.14 first." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c 'import importlib.util; modules = ("qwen_asr", "pyannote.audio", "pycorrector", "llama_cpp", "sentence_transformers"); raise SystemExit(0 if all(importlib.util.find_spec(module) is not None for module in modules) else 1)'; then
  echo "Audio model dependencies are missing from backend/.venv." >&2
  echo "Run scripts/install_audio_dependencies.sh first." >&2
  exit 1
fi

cd "${BACKEND_DIR}"
exec env \
  RUN_PIPELINE_E2E=1 \
  AUDIO_E2E_FILE="${AUDIO_FILE}" \
  PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m pytest -o log_cli=true -o log_cli_level=INFO \
  tests/integration/test_recording_processing_e2e.py::test_real_audio_recording_processing_pipeline_e2e
