#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ -n "${GEMINI_API_KEY:-}" ]] || { echo "GEMINI_API_KEY is required; no planner fallback is enabled."; exit 1; }

DEFAULT_USD="$ROOT/isaac_ext/pathvla_unitree/assets/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"
if [[ -z "${UNITREE_G1_USD_PATH:-}" ]]; then
  [[ -s "$DEFAULT_USD" ]] || bash scripts/download_g1_usd.sh
  export UNITREE_G1_USD_PATH="/workspace/pathvla-unitree-isaac-live/isaac_ext/pathvla_unitree/assets/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"
fi

bash scripts/check_gpu.sh
bash scripts/check_isaac.sh

INSTRUCTION="${1:-Sort every red item into the red bucket and every blue item into the blue bucket.}"

exec docker compose -f docker/docker-compose.yaml run --rm --service-ports pathvla-isaac \
  bash scripts/isaac_python.sh -m isaac_ext.pathvla_unitree.tasks.sorting_demo \
  --instruction "$INSTRUCTION" \
  --model "${GEMINI_ROBOTICS_MODEL:-gemini-robotics-er-1.6-preview}" \
  --live webrtc \
  --record-video
