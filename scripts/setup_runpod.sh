#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROBOT_GYM_RUNPOD_VENV:-$ROOT/.venv-runpod}"
VALIDATE=1

if [[ "${1:-}" == "--no-validate" ]]; then
  VALIDATE=0
elif [[ "$#" -gt 0 ]]; then
  echo "Usage: scripts/setup_runpod.sh [--no-validate]" >&2
  exit 2
fi

cd "$ROOT"

if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" -eq 0 ]]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglfw3 \
    libosmesa6 \
    python3-venv
  rm -rf /var/lib/apt/lists/*
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r requirements-runpod.txt

bash scripts/download_g1_mjcf.sh
bash scripts/setup_demo_2_sil.sh

if [[ "$VALIDATE" -eq 1 ]]; then
  MUJOCO_GL="${MUJOCO_GL:-egl}" \
    PYTHONPATH="$ROOT" \
    "$VENV/bin/python" scripts/validate_runpod.py
fi

echo
echo "RunPod runtime ready."
echo "Start a browser-playable match with:"
echo "  scripts/run_g1_demo_5_runpod.sh play"
echo "Or start a two-browser human match with:"
echo "  scripts/run_g1_demo_5_runpod.sh human-vs-human"
