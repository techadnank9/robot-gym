"""
Live-stream the G1 walking in the scanned library.

    antioch run src/g1_live.py
    antioch run src/g1_live.py -- --vx 0.3 --chase

Runs until stopped. The G1 falls fairly often at this stage, so a fall
respawns it and the stream keeps going rather than ending on the first
stumble — the point is to watch it, not to score it.
"""

from __future__ import annotations

import argparse
import time

import antioch
import numpy as np

from g1_control import (
    CONTROL_DECIMATION,
    G1_PRIM,
    PHYSICS_DT,
    G1Locomotion,
    build_library_scene,
)

SPAWN_XY = (2.0, 0.0)
SPAWN_Z = 0.95
FALL_HEIGHT = 0.45
FALL_TILT = 1.05
RECOVER_PAUSE_S = 0.6


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-stream the G1 in the library")
    parser.add_argument("--vx", type=float, default=0.30, help="Forward command in m/s")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral command in m/s")
    parser.add_argument("--yaw", type=float, default=0.15, help="Yaw rate in rad/s; a slow turn keeps it in the room")
    parser.add_argument("--settle", type=float, default=2.0, help="Seconds to stand before walking")
    parser.add_argument("--seconds", type=float, default=0.0, help="Stop after this long; 0 runs until stopped")
    parser.add_argument("--chase", action="store_true", help="Keep the camera following the robot")
    arguments = parser.parse_args()

    antioch.boot()

    from isaacsim.core.api import World as WorldSingleton
    from isaacsim.core.utils.viewports import set_camera_view

    world = WorldSingleton(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT * CONTROL_DECIMATION)

    robot, colliders = build_library_scene(world, spawn=SPAWN_XY, height=SPAWN_Z)
    print(f"room colliders: {colliders} meshes", flush=True)

    world.reset()
    controller = G1Locomotion(robot)
    controller.configure_gains()

    spawn_position = np.array([[SPAWN_XY[0], SPAWN_XY[1], SPAWN_Z]], dtype=np.float32)
    spawn_orientation = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    def respawn() -> None:
        robot.set_world_poses(positions=spawn_position, orientations=spawn_orientation)
        robot.set_velocities(np.zeros((1, 6), dtype=np.float32))
        robot.set_joint_positions(controller.initial_pose()[None, :])
        robot.set_joint_velocities(np.zeros((1, controller.num_dof), dtype=np.float32))
        controller.reset_state()
        controller.set_command(0.0, 0.0, 0.0)

    respawn()
    set_camera_view(
        eye=[SPAWN_XY[0] + 4.0, SPAWN_XY[1] - 4.0, 2.4],
        target=[SPAWN_XY[0], SPAWN_XY[1], 0.9],
        camera_prim_path="/OmniverseKit_Persp",
    )

    limit = arguments.seconds if arguments.seconds > 0 else None
    started = time.monotonic()
    deadline = started + limit if limit else None

    settle_steps = int(arguments.settle / PHYSICS_DT)
    recover_steps = int(RECOVER_PAUSE_S / PHYSICS_DT)
    step_since_spawn = 0
    falls = 0
    best_walk = 0.0
    walk_start = np.array(SPAWN_XY, dtype=np.float32)
    next_report = 5.0

    print("streaming the G1 in the library; open the livestream from Mission Control", flush=True)

    while deadline is None or time.monotonic() < deadline:
        if step_since_spawn == settle_steps:
            controller.set_command(arguments.vx, arguments.vy, arguments.yaw)
            pos, _, _ = controller.base_state()
            walk_start = pos[:2].copy()

        controller.step(PHYSICS_DT)
        world.step(render=False)
        if step_since_spawn % CONTROL_DECIMATION == 0:
            world.render()
        step_since_spawn += 1

        pos, height, tilt = controller.base_state()

        if arguments.chase and step_since_spawn % CONTROL_DECIMATION == 0:
            set_camera_view(
                eye=[float(pos[0]) + 3.0, float(pos[1]) - 3.0, 2.0],
                target=[float(pos[0]), float(pos[1]), 0.9],
                camera_prim_path="/OmniverseKit_Persp",
            )

        if step_since_spawn > settle_steps and (height < FALL_HEIGHT or tilt > FALL_TILT):
            walked = float(np.linalg.norm(pos[:2] - walk_start))
            best_walk = max(best_walk, walked)
            falls += 1
            print(
                f"fall #{falls} after {walked:.2f} m (best {best_walk:.2f} m); respawning",
                flush=True,
            )
            # Hold the collapsed pose briefly so the stream shows what happened
            for _ in range(recover_steps):
                world.step(render=True)
            respawn()
            step_since_spawn = 0
            continue

        elapsed = time.monotonic() - started
        if elapsed >= next_report:
            print(
                f"[{elapsed:6.0f}s] pos=({pos[0]:6.2f},{pos[1]:6.2f}) h={height:4.2f} "
                f"tilt={tilt:4.2f} falls={falls} best={best_walk:.2f}m",
                flush=True,
            )
            next_report += 5.0

    print(f"stopped after {falls} falls; furthest walk {best_walk:.2f} m", flush=True)


if __name__ == "__main__":
    main()
