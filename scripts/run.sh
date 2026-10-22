#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  echo 'Create the Python environment using README.md first.' >&2
  exit 1
fi
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 "$@"
