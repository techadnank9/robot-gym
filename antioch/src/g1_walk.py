"""
Walk the G1 across the scanned library.

    antioch run src/g1_walk.py                        stream it
    antioch run --no-stream src/g1_walk.py -- --seconds 20
    antioch run src/g1_walk.py -- --vx 0.4 --seconds 60

This is the locomotion milestone: the room gets a collider, the G1 stands on
it, and the pinned policy drives the legs from a velocity command. Reaching
and grasping come after this walks reliably.
"""

from __future__ import annotations

import argparse
import time

import antioch

from g1_control import CONTROL_DECIMATION, PHYSICS_DT, G1Locomotion, build_library_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk the G1 in the library")
    parser.add_argument("--seconds", type=float, default=30.0, help="Wall-clock seconds to simulate; 0 runs until stopped")
    parser.add_argument("--vx", type=float, default=0.3, help="Forward command in m/s")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral command in m/s")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw rate command in rad/s")
    parser.add_argument("--settle", type=float, default=1.0, help="Seconds to stand before commanding motion")
    parser.add_argument(
        "--hold",
        action="store_true",
        help="Skip the policy and hold the default stance, to separate a gains problem from a policy problem",
    )
    parser.add_argument("--spawn-height", type=float, default=None, help="Override the spawn height in metres")
    parser.add_argument("--report-hz", type=float, default=1.0, help="Diagnostic print rate")
    parser.add_argument("--no-ground", action="store_true", help="Walk on the scan surface alone, without the flat plane")
    parser.add_argument("--no-room-collision", action="store_true", help="Make the room visual-only, to separate policy falls from obstacle collisions")
    arguments = parser.parse_args()

    antioch.boot()

    from isaacsim.core.api import World as WorldSingleton
    from isaacsim.core.utils.viewports import set_camera_view

    world = WorldSingleton(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT * CONTROL_DECIMATION)

    spawn_kwargs = {} if arguments.spawn_height is None else {"height": arguments.spawn_height}
    robot, colliders = build_library_scene(
        world, ground=not arguments.no_ground, room_collision=not arguments.no_room_collision, **spawn_kwargs
    )
    print(f"room colliders: {colliders} meshes", flush=True)

    # The articulation only answers for its joints once physics is initialized
    world.reset()
    controller = G1Locomotion(robot)
    controller.configure_gains()
    print(f"gains: {controller.report_gains()}", flush=True)
    robot.set_joint_positions(controller.initial_pose()[None, :])

    start_pos, _, _ = controller.base_state()
    set_camera_view(
        eye=[float(start_pos[0]) + 3.5, float(start_pos[1]) - 3.5, 2.2],
        target=[float(start_pos[0]), float(start_pos[1]), 0.9],
        camera_prim_path="/OmniverseKit_Persp",
    )

    settle_steps = int(arguments.settle / PHYSICS_DT)
    limit = arguments.seconds if arguments.seconds > 0 else None
    started = time.monotonic()
    deadline = started + limit if limit else None

    step = 0
    fallen = False
    next_report = 1.0 / arguments.report_hz

    while deadline is None or time.monotonic() < deadline:
        if step == settle_steps and not arguments.hold:
            controller.set_command(arguments.vx, arguments.vy, arguments.yaw)
            print(f"commanding vx={arguments.vx} vy={arguments.vy} yaw={arguments.yaw}", flush=True)

        if arguments.hold:
            controller.hold_default()
        else:
            controller.step(PHYSICS_DT)
        world.step(render=False)
        if step % CONTROL_DECIMATION == 0:
            world.render()
        step += 1

        sim_time = step * PHYSICS_DT
        report_interval = 1.0 / arguments.report_hz
        if sim_time >= next_report:
            pos, height, tilt = controller.base_state()
            print(
                f"t={sim_time:6.1f}s  pos=({pos[0]:6.2f},{pos[1]:6.2f})  h={height:4.2f}m  "
                f"tilt={tilt:4.2f}rad  yaw={controller.heading():5.2f}rad",
                flush=True,
            )
            next_report += report_interval
            # robot-gym calls the G1 fallen below 0.48 m or past 60 degrees
            if height < 0.48 or tilt > 1.05:
                print(f"FALLEN at t={sim_time:.1f}s  height={height:.2f}  tilt={tilt:.2f}", flush=True)
                fallen = True
                break

    pos, height, tilt = controller.base_state()
    travelled = float(((pos[0] - start_pos[0]) ** 2 + (pos[1] - start_pos[1]) ** 2) ** 0.5)
    print(
        f"done: {'FELL' if fallen else 'upright'}  travelled={travelled:.2f}m  "
        f"final=({pos[0]:.2f},{pos[1]:.2f},{height:.2f})  tilt={tilt:.2f}rad",
        flush=True,
    )


if __name__ == "__main__":
    main()
