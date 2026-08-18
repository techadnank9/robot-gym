"""
Send the G1 to the library shelf to fetch a book, with Gemini Robotics-ER 1.6
choosing what to do next.

    antioch run --timeout 14400 src/fetch_book.py
    antioch run --timeout 14400 src/fetch_book.py -- --adapter scripted
    antioch run --no-stream src/fetch_book.py -- --seconds 120

Three layers, each running at its own rate:

    ER 1.6      head camera + world state  ->  one skill      every few seconds
    skills      geometry                   ->  vx, vy, yaw    50 Hz
    motion.pt   blind balance reflex       ->  joint targets  50 Hz / 500 Hz physics

The shelf position came from ER 1.6 pointing at a rendered view of the scan,
ray-cast onto the mesh: see SHELF_POINT in g1_control.

Grasping is the kinematic attach that robot-gym calls its "easy" mode: within
GRASP_RADIUS the book is locked to the hand. That is an assist, not a
mechanical grasp — the fingers are not doing the work, and the run reports it
as assisted so a result is never mistaken for dexterity.
"""

from __future__ import annotations

import argparse
import io
import math
import threading
import time

import antioch
import numpy as np

from g1_control import (
    CONTROL_DECIMATION,
    PALM_LINK,
    PHYSICS_DT,
    SHELF_NORMAL,
    SHELF_POINT,
    G1Locomotion,
    build_library_scene,
    quat_to_rotation,
)

# Where the book sits relative to the palm frame once held
IN_HAND_OFFSET = np.array([0.04, 0.0, -0.02], dtype=np.float32)
from gemini_er import build_adapter

BOOK_PRIM = "/World/book"
LEDGE_PRIM = "/World/book_ledge"
CAMERA_PRIM = "/World/g1/torso_link/head_camera"

BOOK_SIZE = (0.05, 0.14, 0.20)  # spine, depth, height in metres

# probe_reach measures the palm at 0.39 m forward and 0.20 m above the base in
# the reach pose. Walking base height is ~0.76 m, so the hand passes through
# ~0.96 m: the book goes there, not on the 1.39 m shelf the model pointed at.
BOOK_HEIGHT = 0.96
LEDGE_THICKNESS = 0.03

GRASP_RADIUS = 0.85
GOAL_RADIUS = 0.80
REACH_STOP = 0.55

# Stand off the shelf face along its outward normal
APPROACH_OFFSET = 0.85

CAMERA_RESOLUTION = (640, 400)
# Free-tier quota for this model is 5 requests/minute, so decisions are ~13 s
# apart. robot-gym's 12 s floor is the same constraint. Between decisions the
# guard below keeps the robot sane rather than acting on a stale skill.
DECISION_INTERVAL_S = 13.0

FALL_HEIGHT = 0.45
FALL_TILT = 1.05


def approach_target() -> np.ndarray:
    """Where the robot should stand to face the shelf."""

    normal = SHELF_NORMAL / np.linalg.norm(SHELF_NORMAL)
    point = SHELF_POINT[:2] + normal[:2] * APPROACH_OFFSET
    return point.astype(np.float32)


def build_props(world: object, goal_xy: np.ndarray) -> tuple[object, object]:
    """Put a book on the shelf and mark the drop-off point."""

    from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
    from isaacsim.core.utils.prims import create_prim
    from pxr import UsdPhysics
    import omni.usd

    # A thin static ledge under the book: the scanned shelf surface is
    # photogrammetry, so its exact height under the book is not dependable,
    # and a book that falls through the shelf on frame one ends the task.
    normal = SHELF_NORMAL / np.linalg.norm(SHELF_NORMAL)
    ledge_xy = SHELF_POINT[:2] + normal[:2] * 0.18
    ledge_z = BOOK_HEIGHT - BOOK_SIZE[2] / 2 - LEDGE_THICKNESS
    create_prim(
        LEDGE_PRIM,
        "Cube",
        position=(float(ledge_xy[0]), float(ledge_xy[1]), ledge_z),
        scale=(0.22, 0.22, LEDGE_THICKNESS),
    )
    stage = omni.usd.get_context().get_stage()
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(LEDGE_PRIM))

    book = world.scene.add(
        DynamicCuboid(
            prim_path=BOOK_PRIM,
            name="book",
            position=np.array([float(ledge_xy[0]), float(ledge_xy[1]), BOOK_HEIGHT]),
            scale=np.array(BOOK_SIZE),
            color=np.array([0.85, 0.15, 0.15]),
            mass=0.4,
        )
    )
    goal = world.scene.add(
        VisualCuboid(
            prim_path="/World/goal",
            name="goal",
            position=np.array([float(goal_xy[0]), float(goal_xy[1]), 0.02]),
            scale=np.array([0.7, 0.7, 0.02]),
            color=np.array([0.1, 0.8, 0.3]),
        )
    )
    return book, goal


def build_camera() -> object:
    """A head camera on the torso, looking forward — ER 1.6's only eye."""

    from isaacsim.sensors.camera import Camera
    import isaacsim.core.utils.numpy.rotations as rot_utils

    camera = Camera(
        prim_path=CAMERA_PRIM,
        translation=np.array([0.12, 0.0, 0.42]),
        resolution=CAMERA_RESOLUTION,
        orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 10.0, 0.0]), degrees=True),
    )
    return camera


def jpeg_from_camera(camera: object) -> bytes:
    """Grab one RGB frame as JPEG, or empty bytes if the sensor has none yet."""

    from PIL import Image

    frame = camera.get_rgba()
    if frame is None or getattr(frame, "size", 0) == 0:
        return b""
    rgb = np.asarray(frame)[:, :, :3].astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()


def guard_skill(skill: str, carrying: bool, fallen: bool, book_distance: float, goal_distance: float) -> str:
    """
    Keep a stale or impossible skill from driving the robot.

    The deciding layer runs every ~13 s; the world changes in between, and a
    skill that was right when it was chosen can be wrong by the time it is
    still the only one available. This mirrors robot-gym's guard table: the
    model proposes, the grounded state disposes.
    """

    if fallen:
        return "recover"
    if carrying:
        # Already holding it: grasping again is a no-op that stands still while
        # the policy drifts, which is how the book ends up walked away from
        if skill in ("grasp", "navigate_book", "wait"):
            return "navigate_goal"
        if skill == "release" and goal_distance > GOAL_RADIUS:
            return "navigate_goal"
        return skill
    if skill in ("release", "navigate_goal"):
        return "navigate_book"
    if skill == "grasp" and book_distance > GRASP_RADIUS:
        return "navigate_book"
    return skill


def skill_to_command(
    skill: str, position: np.ndarray, heading: float, book_xy: np.ndarray, goal_xy: np.ndarray
) -> tuple[float, float, float]:
    """
    Turn a chosen skill into a body velocity command.

    Mirrors robot-gym's navigate law: drive forward only in proportion to how
    well the body already points at the target, and steer with yaw. Walking
    sideways at a target the robot is not facing is how a humanoid falls over.
    """

    if skill in ("wait", "grasp", "release", "recover"):
        return 0.0, 0.0, 0.0

    target = book_xy if skill == "navigate_book" else goal_xy
    delta = target - position[:2]
    distance = float(np.linalg.norm(delta))
    bearing = math.atan2(float(delta[1]), float(delta[0]))
    error = math.atan2(math.sin(bearing - heading), math.cos(bearing - heading))

    stop = REACH_STOP if skill == "navigate_book" else GOAL_RADIUS * 0.6
    forward = min(0.40, max(0.0, distance - stop)) * max(0.0, math.cos(error))
    yaw_rate = float(np.clip(1.6 * error, -0.82, 0.82))
    return forward, 0.0, yaw_rate


class DecisionWorker:
    """
    Run the deciding layer off the physics thread.

    A model call takes seconds; the gait runs at 500 Hz. Calling the adapter
    inline stalls physics and freezes the livestream for the whole round trip,
    which is what "connection interrupted" looks like from the browser. The
    loop keeps stepping and simply uses the most recent decision until a new
    one lands.
    """

    def __init__(self, adapter: object, interval_s: float) -> None:
        self._adapter = adapter
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._pending: dict | None = None
        self._frame = b""
        self._decision = None
        self._busy = False
        self._calls = 0
        self._last_dispatch = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, state: dict, frame: bytes) -> bool:
        """Offer fresh state; returns True when it starts a new call."""

        now = time.monotonic()
        with self._lock:
            if self._busy or now - self._last_dispatch < self._interval_s:
                return False
            self._pending = state
            self._frame = frame
            self._busy = True
            self._last_dispatch = now
        return True

    def latest(self):
        with self._lock:
            return self._decision

    @property
    def calls(self) -> int:
        return self._calls

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                state = self._pending if self._busy else None
                frame = self._frame
            if state is None:
                time.sleep(0.02)
                continue
            try:
                decision = self._adapter.decide(state, [frame] if frame else [])
            except Exception as error:  # noqa: BLE001 - never kill the sim thread
                print(f"[decide] adapter raised {type(error).__name__}: {error}", flush=True)
                decision = None
            with self._lock:
                if decision is not None:
                    self._decision = decision
                    self._calls += 1
                self._pending = None
                self._busy = False

    def close(self) -> None:
        self._stop.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a book from the library shelf")
    parser.add_argument("--adapter", choices=("gemini", "scripted"), default="gemini")
    parser.add_argument("--seconds", type=float, default=0.0, help="0 runs until stopped")
    parser.add_argument("--spawn-x", type=float, default=2.0)
    parser.add_argument("--spawn-y", type=float, default=0.0)
    parser.add_argument("--spawn-height", type=float, default=0.95)
    parser.add_argument(
        "--room-collision",
        action="store_true",
        help="Collide with the scan as well as the ground plane; off by default because the "
        "scan floor and the plane overlap and fight for every footfall",
    )
    arguments = parser.parse_args()

    adapter = build_adapter(arguments.adapter)
    print(f"deciding layer: {adapter.model_name}", flush=True)

    antioch.boot()

    from isaacsim.core.api import World as WorldSingleton
    from isaacsim.core.utils.viewports import set_camera_view

    world = WorldSingleton(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT * CONTROL_DECIMATION)

    spawn_xy = (arguments.spawn_x, arguments.spawn_y)
    robot, colliders = build_library_scene(
        world, spawn=spawn_xy, height=arguments.spawn_height, room_collision=arguments.room_collision
    )

    stand = approach_target()
    # Hand the book over where the run began
    goal_xy = np.array([spawn_xy[0], spawn_xy[1]], dtype=np.float32)
    book, _goal = build_props(world, goal_xy)
    camera = build_camera()

    print(f"room colliders: {colliders}", flush=True)
    print(f"shelf at {SHELF_POINT.tolist()}  stand at {stand.tolist()}  goal at {goal_xy.tolist()}", flush=True)

    from isaacsim.core.prims import XFormPrim

    # Read the rendered palm transform: a RigidPrim on a link inside an
    # articulation is treated as its own body and reports free fall
    palm = XFormPrim(prim_paths_expr=PALM_LINK, name="palm")

    world.reset()
    camera.initialize()
    controller = G1Locomotion(robot)
    controller.configure_gains()

    spawn_position = np.array([[spawn_xy[0], spawn_xy[1], arguments.spawn_height]], dtype=np.float32)
    spawn_orientation = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    shelf_normal = SHELF_NORMAL / np.linalg.norm(SHELF_NORMAL)
    book_home = np.array(
        [
            float(SHELF_POINT[0] + shelf_normal[0] * 0.18),
            float(SHELF_POINT[1] + shelf_normal[1] * 0.18),
            BOOK_HEIGHT,
        ],
        dtype=np.float32,
    )

    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    book_prim = stage.GetPrimAtPath(BOOK_PRIM)

    def set_book_collision(enabled: bool) -> None:
        """
        Toggle the book's collider.

        A held book is teleported to the palm every tick. Left collidable it
        interpenetrates the hand and the contact solver throws the robot across
        the room, which reads as a fall a fraction of a second after a
        successful grasp. robot-gym disables payload collision for the same
        reason while a payload is attached.
        """

        attribute = book_prim.GetAttribute("physics:collisionEnabled")
        if not attribute:
            attribute = UsdPhysics.CollisionAPI(book_prim).CreateCollisionEnabledAttr()
        attribute.Set(bool(enabled))

    def park_book(position: np.ndarray) -> None:
        """Place the book and kill its momentum; DynamicCuboid is a single prim."""

        book.set_world_pose(position=position)
        book.set_linear_velocity(np.zeros(3, dtype=np.float32))
        book.set_angular_velocity(np.zeros(3, dtype=np.float32))

    def respawn() -> None:
        robot.set_world_poses(positions=spawn_position, orientations=spawn_orientation)
        robot.set_velocities(np.zeros((1, 6), dtype=np.float32))
        robot.set_joint_positions(controller.initial_pose()[None, :])
        robot.set_joint_velocities(np.zeros((1, controller.num_dof), dtype=np.float32))
        controller.reset_state()
        controller.set_command(0.0, 0.0, 0.0)

    respawn()
    park_book(book_home)
    set_camera_view(
        eye=[spawn_xy[0] + 3.0, spawn_xy[1] - 3.0, 2.2],
        target=[float(SHELF_POINT[0]), float(SHELF_POINT[1]), 1.0],
        camera_prim_path="/OmniverseKit_Persp",
    )

    limit = arguments.seconds if arguments.seconds > 0 else None
    started = time.monotonic()
    deadline = started + limit if limit else None

    worker = DecisionWorker(adapter, DECISION_INTERVAL_S)
    raw_skill = "wait"
    skill = "wait"
    last_heartbeat = time.monotonic()
    carrying = False
    delivered = False
    falls = 0
    attempts = 0
    step = 0
    next_decision = 0.0
    settle_steps = int(2.5 / PHYSICS_DT)

    print("fetch task running; open the livestream from Mission Control", flush=True)

    while deadline is None or time.monotonic() < deadline:
        position, height, tilt = controller.base_state()
        heading = controller.heading()
        book_position = np.asarray(book.get_world_pose()[0], dtype=np.float32)

        fallen = step > settle_steps and (height < FALL_HEIGHT or tilt > FALL_TILT)
        book_distance = float(np.linalg.norm(position[:2] - book_position[:2]))
        goal_distance = float(np.linalg.norm(position[:2] - goal_xy))

        if step > settle_steps:
            state = {
                "robot_xy": [round(float(position[0]), 2), round(float(position[1]), 2)],
                "heading_rad": round(heading, 2),
                "book_xy": [round(float(book_position[0]), 2), round(float(book_position[1]), 2)],
                "book_distance_m": round(book_distance, 2),
                "goal_distance_m": round(goal_distance, 2),
                "reach_radius_m": GRASP_RADIUS,
                "goal_radius_m": GOAL_RADIUS,
                "carrying": carrying,
                "fallen": bool(fallen),
                "delivered": delivered,
            }
            if worker.submit(state, jpeg_from_camera(camera)):
                pass
            decision = worker.latest()
            if decision is not None and decision.skill != raw_skill:
                raw_skill = decision.skill
                print(
                    f"[decide t={step * PHYSICS_DT:6.1f}s] {raw_skill:14s} "
                    f"book={book_distance:4.2f}m goal={goal_distance:4.2f}m carrying={carrying} "
                    f"| {decision.rationale[:90]}",
                    flush=True,
                )

        skill = guard_skill(raw_skill, carrying, fallen, book_distance, goal_distance)

        # The arm reaches when the book is close and holds it once carried, so
        # the hand is where the book is rather than the book meeting a pose
        if carrying:
            controller.set_arm("carry", hand_closed=True)
        elif book_distance <= GRASP_RADIUS + 0.5:
            controller.set_arm("reach", hand_closed=False)
        else:
            controller.set_arm("rest", hand_closed=False)

        # Skills with an effect on the world, applied here rather than in the
        # velocity mapping so the assist is explicit
        if skill == "grasp" and not carrying and book_distance <= GRASP_RADIUS:
            carrying = True
            attempts += 1
            set_book_collision(False)
            print(f"[grasp] assisted attach at {book_distance:.2f} m", flush=True)
        elif skill == "release" and carrying:
            carrying = False
            set_book_collision(True)
            park_book(np.array([goal_xy[0], goal_xy[1], 0.20], dtype=np.float32))
            if goal_distance <= GOAL_RADIUS:
                delivered = True
                print(f"[release] DELIVERED at {goal_distance:.2f} m from goal", flush=True)
            else:
                print(f"[release] dropped early, {goal_distance:.2f} m from goal", flush=True)

        if fallen:
            falls += 1
            print(f"[fall #{falls}] respawning", flush=True)
            for _ in range(int(0.5 / PHYSICS_DT)):
                world.step(render=True)
            respawn()
            if carrying:
                carrying = False
                set_book_collision(True)
                park_book(book_home)
            step = 0
            raw_skill = "wait"
            skill = "wait"
            continue

        target_book = book_position[:2] if not carrying else stand
        vx, vy, yaw_rate = skill_to_command(skill, position, heading, target_book, goal_xy)
        controller.set_command(vx, vy, yaw_rate)
        controller.step(PHYSICS_DT)

        # Hold the book at the palm so it travels with the hand
        if carrying:
            palm_position, palm_quat = palm.get_world_poses()
            palm_position = np.asarray(palm_position, dtype=np.float32)[0]
            palm_rotation = quat_to_rotation(np.asarray(palm_quat, dtype=np.float32)[0])
            park_book(palm_position + palm_rotation @ IN_HAND_OFFSET)

        world.step(render=False)
        if step % CONTROL_DECIMATION == 0:
            world.render()
        step += 1

        if time.monotonic() - last_heartbeat >= 10.0:
            print(
                f"[hb t={step * PHYSICS_DT:6.1f}s] skill={skill:14s} pos=({position[0]:5.2f},{position[1]:5.2f}) "
                f"h={height:4.2f} book={book_distance:4.2f}m goal={goal_distance:4.2f}m "
                f"carrying={carrying} calls={worker.calls} falls={falls}",
                flush=True,
            )
            last_heartbeat = time.monotonic()

        if delivered:
            print(
                f"TASK COMPLETE: book delivered after {falls} falls, {attempts} grasp attempts "
                f"(assisted attach, not a mechanical grasp)",
                flush=True,
            )
            delivered = False
            park_book(book_home)
            respawn()
            step = 0

    print(f"stopped: {falls} falls, {attempts} grasp attempts", flush=True)


if __name__ == "__main__":
    main()
