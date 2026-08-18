"""
Measure where the G1's right palm actually sits in each arm posture.

    antioch run --no-stream src/probe_reach.py

The book has to sit where the hand can be, not where a guess puts it. This
reports the palm in the body frame for each pose in ARM_POSES, so the shelf
height and the in-hand offset come from measurement.
"""

from __future__ import annotations

import antioch
import numpy as np

from g1_control import (
    ARM_POSES,
    CONTROL_DECIMATION,
    PALM_LINK,
    PHYSICS_DT,
    G1Locomotion,
    build_library_scene,
    quat_to_rotation,
)


def main() -> None:
    antioch.boot()

    from isaacsim.core.api import World as WorldSingleton
    from isaacsim.core.prims import XFormPrim

    world = WorldSingleton(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT * CONTROL_DECIMATION)
    robot, _ = build_library_scene(world, spawn=(2.0, 0.0), height=0.95, room_collision=False)
    world.reset()

    controller = G1Locomotion(robot)
    controller.configure_gains()
    # XFormPrim reads the rendered transform. RigidPrim on a link inside an
    # articulation gets treated as its own body and reports free fall.
    palm = XFormPrim(prim_paths_expr=PALM_LINK, name="palm")

    for pose in ("rest", "reach", "carry"):
        controller.set_arm(pose, hand_closed=(pose == "carry"))
        robot.set_joint_positions(controller.initial_pose()[None, :])
        # Pin the base while the arm settles: this measures arm kinematics, and
        # a robot that topples mid-measurement reports the palm in free fall
        hold_position = np.array([[2.0, 0.0, 0.95]], dtype=np.float32)
        hold_orientation = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        for _ in range(400):
            robot.set_world_poses(positions=hold_position, orientations=hold_orientation)
            robot.set_velocities(np.zeros((1, 6), dtype=np.float32))
            controller.step(PHYSICS_DT)
            world.step(render=False)

        base_position, _, _ = controller.base_state()
        _, base_quat = robot.get_world_poses()
        rotation = quat_to_rotation(np.asarray(base_quat, dtype=np.float32)[0])

        palm_position = np.asarray(palm.get_world_poses()[0], dtype=np.float32)[0]
        local = rotation.T @ (palm_position - base_position)
        print(
            f"POSE {pose:6s} palm_world=({palm_position[0]:6.2f},{palm_position[1]:6.2f},{palm_position[2]:5.2f}) "
            f"palm_body=(fwd {local[0]:5.2f}, left {local[1]:5.2f}, up {local[2]:5.2f})",
            flush=True,
        )

    print("PROBE_REACH_DONE", flush=True)


if __name__ == "__main__":
    main()
