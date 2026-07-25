#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ "$(uname -s)" == "Darwin" ]] || { echo "The native Replit worker requires macOS."; exit 1; }
[[ -x .venv-mac/bin/mjpython ]] || { echo "Run 'make mac-setup' first."; exit 1; }
[[ -n "${GEMINI_API_KEY:-}" ]] || { echo "GEMINI_API_KEY is required; no planner fallback is enabled."; exit 1; }
[[ -n "${REPLIT_CONTROL_URL:-}" ]] || { echo "REPLIT_CONTROL_URL is required."; exit 1; }
[[ -n "${REPLIT_WORKER_TOKEN:-}" ]] || { echo "REPLIT_WORKER_TOKEN is required."; exit 1; }

exec .venv-mac/bin/mjpython -m pathvla.replit_mac_worker
