#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${ROBOT_GYM_PYTHON:-$ROOT/.venv-runpod/bin/python}"
if [[ "$PYTHON_BIN" != */* ]]; then
  PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
fi
[[ -x "$PYTHON_BIN" ]] || {
  echo "Run 'bash scripts/setup_runpod.sh' first." >&2
  exit 1
}
[[ -f demo_2/vendor/unitree_rl_gym/deploy/pre_train/g1/motion.pt ]] || {
  echo "Pinned Unitree policy is missing; run scripts/setup_runpod.sh." >&2
  exit 1
}

export PYTHONPATH="$ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

MODE="${1:-play}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

HOST="${DEMO5_HOST:-0.0.0.0}"
HTTP_PORT="${DEMO5_HTTP_PORT:-8085}"
WS_PORT="${DEMO5_WEBSOCKET_PORT:-8765}"
COMMON=(
  --headless
  --realtime
  --host "$HOST"
  --http-port "$HTTP_PORT"
  --websocket-port "$WS_PORT"
  --grasp-mode "${DEMO5_GRASP_MODE:-easy}"
  --render-profile "${DEMO5_RENDER_PROFILE:-performance}"
)

if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
  echo "Open: https://${RUNPOD_POD_ID}-${HTTP_PORT}.proxy.runpod.net"
else
  echo "Open: http://127.0.0.1:${HTTP_PORT}"
fi

case "$MODE" in
  play)
    exec "$PYTHON_BIN" -m demo_5 \
      "${COMMON[@]}" \
      --p1 human --p1-input keyboard \
      --p2 policy --p2-adapter "${DEMO5_OPPONENT_ADAPTER:-scripted}" \
      "$@"
    ;;
  practice)
    exec "$PYTHON_BIN" -m demo_5 \
      "${COMMON[@]}" \
      --p1 human --p1-input keyboard \
      --p2 human --p2-input idle \
      "$@"
    ;;
  human-vs-human|hvh)
    exec "$PYTHON_BIN" -m demo_5 \
      "${COMMON[@]}" \
      --p1 human --p1-input keyboard \
      --p2 human --p2-input keyboard \
      --keyboard-ready-timeout "${DEMO5_TWO_PLAYER_READY_TIMEOUT:-300}" \
      "$@"
    ;;
  gemini)
    [[ -n "${DEMO3_P2_GEMINI_API_KEY:-${GEMINI_API_KEY:-}}" ]] || {
      echo "Set DEMO3_P2_GEMINI_API_KEY or GEMINI_API_KEY first." >&2
      exit 1
    }
    exec "$PYTHON_BIN" -m demo_5 \
      "${COMMON[@]}" \
      --p1 human --p1-input keyboard \
      --p2 policy --p2-adapter gemini-er \
      "$@"
    ;;
  ai-vs-ai)
    exec "$PYTHON_BIN" -m demo_5 \
      "${COMMON[@]}" \
      --p1 policy --p1-adapter "${DEMO5_P1_ADAPTER:-scripted}" \
      --p2 policy --p2-adapter "${DEMO5_P2_ADAPTER:-scripted}" \
      "$@"
    ;;
  validate)
    exec env MUJOCO_GL="$MUJOCO_GL" PYTHONPATH="$ROOT" \
      "$PYTHON_BIN" scripts/validate_runpod.py
    ;;
  match)
    exec "$PYTHON_BIN" -m demo_5 "${COMMON[@]}" "$@"
    ;;
  *)
    echo "Usage: scripts/run_g1_demo_5_runpod.sh {play|practice|human-vs-human|hvh|gemini|ai-vs-ai|validate|match} [options]" >&2
    exit 2
    ;;
esac
