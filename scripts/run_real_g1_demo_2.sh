#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" == "Darwin" ]]; then
  for argument in "$@"; do
    if [[ "$argument" == "mujoco" ]]; then
      [[ -x .venv-mac/bin/mjpython ]] || {
        echo "Run 'make mac-setup' first; MuJoCo GUI requires .venv-mac/bin/mjpython on macOS." >&2
        exit 1
      }
      exec .venv-mac/bin/mjpython -m demo_2 "$@"
    fi
  done
fi

exec python3 -m demo_2 "$@"
