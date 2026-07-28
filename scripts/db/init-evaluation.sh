#!/usr/bin/env sh

set -eu

REPOSITORY_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
BACKEND_ROOT="$REPOSITORY_ROOT/backend"
PYTHON_BIN="$BACKEND_ROOT/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[db:init-evaluation] Python virtual environment was not found: $PYTHON_BIN" >&2
  echo "[db:init-evaluation] Create it first: cd backend && python3.14 -m venv .venv && .venv/bin/python -m pip install -e '.[dev]'" >&2
  exit 1
fi

cd "$BACKEND_ROOT"
exec env PYTHONPATH=packages "$PYTHON_BIN" scripts/initialize_evaluation_database.py
