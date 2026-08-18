"""
Stand a Unitree G1 in the scanned library and report its articulation.

    antioch run --no-stream src/probe_g1.py

The locomotion policy is indexed by joint name in MuJoCo order, so the DOF
names and order Isaac reports here are what the port has to map onto. Nothing
is controlled yet: this only establishes the interface.
"""

from __future__ import annotations

import antioch

ROOM_ASSET = "library-real"
ROOM_PRIM = "/World/library"
G1_PRIM = "/World/g1"

# MuJoCo leg order the policy was trained with (left leg, then right)
POLICY_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]


def main() -> None:
    antioch.boot()

    import numpy as np
    from isaacsim.core.api import World as WorldSingleton
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.storage.native import get_assets_root_path

    world = WorldSingleton()

    antioch.load_asset(ROOM_ASSET, prim_path=ROOM_PRIM)
    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 300.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 500.0})

    root = get_assets_root_path()
    g1_usd = f"{root}/Isaac/Robots/Unitree/G1/g1.usd"
    print(f"G1_USD={g1_usd}", flush=True)
    add_reference_to_stage(usd_path=g1_usd, prim_path=G1_PRIM)

    from isaacsim.core.prims import Articulation

    # Stand it on the open floor, clear of the tables
    robot = Articulation(prim_paths_expr=G1_PRIM, name="g1", positions=np.array([[2.0, 0.0, 0.8]]))
    world.scene.add(robot)
    world.reset()

    set_camera_view(eye=[5.0, -3.5, 2.0], target=[2.0, 0.0, 0.8], camera_prim_path="/OmniverseKit_Persp")

    names = list(robot.dof_names)
    print(f"NUM_DOF={robot.num_dof}", flush=True)
    print(f"DOF_NAMES={names}", flush=True)

    missing = [j for j in POLICY_JOINTS if j not in names]
    print(f"MISSING_POLICY_JOINTS={missing}", flush=True)
    if not missing:
        index = [names.index(j) for j in POLICY_JOINTS]
        print(f"POLICY_DOF_INDEX={index}", flush=True)

    # Let it settle under gravity so the reported pose is a physical one
    for _ in range(120):
        world.step(render=False)

    pos, quat = robot.get_world_poses()
    print(f"BASE_POS={np.asarray(pos)[0].tolist()}", flush=True)
    print(f"BASE_QUAT={np.asarray(quat)[0].tolist()}", flush=True)
    print(f"JOINT_POS={np.asarray(robot.get_joint_positions())[0].tolist()}", flush=True)


if __name__ == "__main__":
    main()
