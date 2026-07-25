#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

docker compose -f docker/docker-compose.yaml run --rm --service-ports pathvla-isaac bash -lc '
set -euo pipefail
export PYTHONPATH=/workspace/pathvla-unitree-isaac-live:${PYTHONPATH:-}
bash scripts/isaac_python.sh - <<'"'"'PY'"'"'
from isaac_ext.pathvla_unitree.tasks.livestream import configure_livestream
from isaac_ext.pathvla_unitree.tasks.room_nav_env import launch_simulation_app
from isaac_ext.pathvla_unitree.tasks.room_nav_env_cfg import load_livestream_config
from pathvla.schemas import RunMode

cfg = load_livestream_config()
print("[livestream] Required extensions:", ", ".join(cfg.livestream.required_extensions))
print("[livestream] Ports:")
print(f"  signaling: {cfg.livestream.signaling_port}")
print(f"  http: {cfg.livestream.http_port}")
print(f"  udp: {cfg.livestream.udp_port_range}")

app = launch_simulation_app(headless=False)
try:
    info = configure_livestream(RunMode.WEBRTC, cfg, logger=type("L", (), {"info": print})())
    print("[livestream] Mac connection instructions:")
    print(info["instructions"])
finally:
    app.close()
print("[livestream] Livestream checks passed")
PY
'
