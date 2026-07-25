#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" == "Darwin" ]]; then
  [[ -x .venv-mac/bin/python ]] || {
    echo "Run 'make mac-setup' and install demo_2/requirements-sil.txt first." >&2
    exit 1
  }
  exec .venv-mac/bin/python -m demo_2.sil_evidence "$@"
fi

exec python3 -m demo_2.sil_evidence "$@"
