#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python3 - <<'PY'
import json
import subprocess
from pathlib import Path

tasks = json.loads(Path("eval/tasks.json").read_text(encoding="utf-8"))
for task in tasks:
    cmd = [
        "python3",
        "-m",
        "isaac_ext.pathvla_unitree.tasks.room_nav_env",
        "--scene",
        task["scene"],
        "--instruction",
        task["instruction"],
        "--robot",
        "unitree_g1",
        "--live",
        "none",
        "--record-video",
        "--require-vla",
    ]
    if task.get("allow_proxy"):
        cmd.append("--allow-proxy")
    if task.get("allow_kinematic_control"):
        cmd.append("--allow-kinematic-control")
    if task.get("allow_rule_planner"):
        cmd.append("--allow-rule-planner")
    print("[eval] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
PY
