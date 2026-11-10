#!/usr/bin/env bash
set -euo pipefail
# Run inside your chosen Python environment. Existing CUDA 2.6 builds are retained.
cd -- "$(dirname -- "$0")"
if [[ "${1:-}" == "--cpu" ]]; then
  python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
elif [[ $# -ne 0 ]]; then
  echo 'Usage: bash setup.sh [--cpu]' >&2
  exit 2
fi
python -m pip install -r requirements.txt
python bootstrap.py
