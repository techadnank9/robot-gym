#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ "$(uname -s)" == "Darwin" ]] || { echo "The native MuJoCo viewer launcher currently targets macOS."; exit 1; }
[[ -x .venv-mac/bin/mjpython ]] || { echo "Run 'make mac-setup' first."; exit 1; }

exec .venv-mac/bin/mjpython -m pathvla.library_mujoco \
  --scan-dir "${LIBRARY_SCAN_DIR:-assets/library_scan}" \
  "$@"
