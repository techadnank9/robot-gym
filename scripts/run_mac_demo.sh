#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ "$(uname -s)" == "Darwin" ]] || { echo "Native Mac demo requires macOS."; exit 1; }
[[ -x .venv-mac/bin/mjpython ]] || { echo "Run 'make mac-setup' first."; exit 1; }
[[ -n "${GEMINI_API_KEY:-}" ]] || { echo "GEMINI_API_KEY is required; no planner fallback is enabled."; exit 1; }

INSTRUCTION="${1:-Sort every red item into the red bucket and every blue item into the blue bucket.}"
exec .venv-mac/bin/mjpython -m pathvla.mujoco_sorting_demo \
  --instruction "$INSTRUCTION" \
  --model "${GEMINI_ROBOTICS_MODEL:-gemini-robotics-er-1.6-preview}" \
  --record-video
