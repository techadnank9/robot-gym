#!/usr/bin/env bash
set -euo pipefail

if [[ -x "/isaac-sim/python.sh" ]]; then
  exec /isaac-sim/python.sh "$@"
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 "$@"
fi

if command -v python >/dev/null 2>&1; then
  exec python "$@"
fi

echo "No Isaac-compatible Python launcher found. Expected /isaac-sim/python.sh, python3, or python." >&2
exit 1
