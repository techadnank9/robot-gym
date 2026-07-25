#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

exec bash scripts/isaac_python.sh -m uvicorn apps.vla_server:app --host "${PATHVLA_VLA_HOST:-0.0.0.0}" --port "${PATHVLA_VLA_PORT:-5555}"
