#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" == "Darwin" ]]; then
  [[ -x .venv-mac/bin/mjpython ]] || {
    echo "Run 'make mac-setup' first; Demo 2 requires .venv-mac/bin/mjpython on macOS." >&2
    exit 1
  }
  exec .venv-mac/bin/mjpython -m demo_2.full_demo "$@"
fi

exec python3 -m demo_2.full_demo "$@"
