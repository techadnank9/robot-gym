#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

exec bash scripts/isaac_python.sh -m streamlit run apps/dashboard.py --server.port "${PATHVLA_DASHBOARD_PORT:-8501}" --server.address 0.0.0.0
