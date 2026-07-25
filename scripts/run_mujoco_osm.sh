#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ "$(uname -s)" == "Darwin" ]] || { echo "The native MuJoCo viewer launcher currently targets macOS."; exit 1; }
[[ -x .venv-mac/bin/mjpython ]] || { echo "Run 'make mac-setup' first."; exit 1; }

exec .venv-mac/bin/mjpython -m pathvla.osm_mujoco \
  --config "${OSM_SCENE_CONFIG:-config/osm_sf_golden_gate_bridge.yaml}" \
  --output-dir "${OSM_OUTPUT_DIR:-outputs/mujoco_sf_golden_gate_bridge}" \
  "$@"
