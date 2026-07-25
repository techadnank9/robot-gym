#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCENE="${1:-room}"
INSTRUCTION="${2:-Go to the red bin, avoid the chair, inspect the table, then return home.}"
ALLOW_PROXY="${3:-0}"
ALLOW_KINEMATIC="${4:-0}"
ALLOW_RULE_PLANNER="${5:-0}"

if [[ "$ALLOW_RULE_PLANNER" != "1" ]]; then
  [[ -n "${VLA_ENDPOINT:-}" ]] || { echo "VLA_ENDPOINT is required for live demo strict mode."; exit 1; }
fi
if [[ "$ALLOW_PROXY" != "1" ]]; then
  [[ -n "${UNITREE_G1_USD_PATH:-}" ]] || { echo "UNITREE_G1_USD_PATH is required unless ALLOW_PROXY=1."; exit 1; }
fi

bash scripts/check_gpu.sh
bash scripts/check_isaac.sh

CMD=(
  bash scripts/isaac_python.sh
  -m isaac_ext.pathvla_unitree.tasks.room_nav_env
  --scene "$SCENE"
  --instruction "$INSTRUCTION"
  --robot unitree_g1
  --live webrtc
  --record-video
)

if [[ "$ALLOW_RULE_PLANNER" == "1" ]]; then
  CMD+=(--no-require-vla --allow-rule-planner)
else
  CMD+=(--require-vla)
fi

if [[ "$ALLOW_PROXY" == "1" ]]; then
  CMD+=(--allow-proxy)
fi
if [[ "$ALLOW_KINEMATIC" == "1" ]]; then
  CMD+=(--allow-kinematic-control)
fi

exec docker compose -f docker/docker-compose.yaml run --rm --service-ports pathvla-isaac "${CMD[@]}"
