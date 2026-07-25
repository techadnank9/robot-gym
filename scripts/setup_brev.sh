#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[setup] Preparing PathVLA Unitree Isaac Live on a Brev-style GPU VM"

command -v python3 >/dev/null || { echo "python3 is required"; exit 1; }
command -v docker >/dev/null || { echo "docker is required"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required"; exit 1; }

mkdir -p outputs
python3 -m pip install -r requirements-dev.txt

cat <<EOF
[setup] Base Python dependencies installed.
[setup] Next:
  1. export ISAAC_BASE_IMAGE=<official-compatible-isaac-image>
  2. export VLA_ENDPOINT=https://your-vla-server/infer
  3. export UNITREE_G1_USD_PATH=/path/to/unitree_g1.usd
  4. export BREV_PUBLIC_HOST=<public-hostname>
  5. run: make check-gpu && make check-isaac && make check-livestream
EOF
