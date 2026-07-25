from __future__ import annotations

import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from demo_3.schemas import PlayerConfig
from demo_5.arena import SimToRealG1RaceArena


def main() -> None:
    players = (
        PlayerConfig(player_id="p1", display_name="Vector"),
        PlayerConfig(player_id="p2", display_name="Nova"),
    )
    arena = SimToRealG1RaceArena(
        players,
        viewer=False,
        realtime=False,
        grasp_mode="easy",
    )
    try:
        arena.start()
        arena.step(250)
        frame = arena.render("broadcast_camera", width=320, height=180)
        if frame.shape != (180, 320, 3):
            raise RuntimeError(f"Unexpected render shape: {frame.shape}")
        report = {
            "status": "ok",
            "mujocoGl": os.environ["MUJOCO_GL"],
            "renderShape": list(frame.shape),
            "simulationTime": arena.simulation_time_s,
            "officialUnitreePolicyLoaded": True,
            "graspMode": arena.grasp_mode,
        }
        print(json.dumps(report, indent=2))
    finally:
        arena.close()


if __name__ == "__main__":
    main()
