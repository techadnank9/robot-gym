from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathvla.run_registry import get_output_root
from pathvla.schemas import SceneSnapshotModel
from pathvla.vla_client import HTTPVLAClient, load_vla_endpoint_config


def main() -> int:
    output_dir = get_output_root() / "vla_endpoint_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = SceneSnapshotModel(
        scene_name="room",
        objects=[
            {"name": "red_bin", "pose": [2.0, 1.0, 0.3], "type": "container", "avoidance_radius": 0.5},
            {"name": "chair", "pose": [0.6, 0.6, 0.5], "type": "obstacle", "avoidance_radius": 0.7},
            {"name": "table", "pose": [-1.2, 1.7, 0.4], "type": "inspectable", "avoidance_radius": 0.9},
            {"name": "bottle", "pose": [-1.1, 1.7, 0.9], "type": "pickup", "avoidance_radius": 0.2},
            {"name": "home_marker", "pose": [0.0, -2.5, 0.02], "type": "home", "avoidance_radius": 0.3},
        ],
        robot={"name": "unitree_g1", "pose": [0.0, 0.0, 0.0]},
        bounds={"x": [-4.0, 4.0], "y": [-4.0, 4.0], "z_floor": [0.0, 0.0]},
    )
    client = HTTPVLAClient(load_vla_endpoint_config())
    plan = client.infer(
        instruction="Go to the red bin, avoid the chair, inspect the table, then return home.",
        scene=scene,
        output_dir=output_dir,
    )
    print(json.dumps(plan.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
