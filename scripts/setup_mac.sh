#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ "$(uname -s)" == "Darwin" ]] || { echo "This setup script is for macOS."; exit 1; }
if [[ ! -x .venv-mac/bin/python ]]; then
  python3 -m venv .venv-mac
fi
.venv-mac/bin/python -m pip install --upgrade pip
.venv-mac/bin/python -m pip install -r requirements-mac.txt
bash scripts/download_g1_mjcf.sh
echo "Mac runtime ready. Run: make mac-demo"
