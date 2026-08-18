"""
Unitree G1 locomotion in the scanned library.

The policy is the pinned `unitree_rl_gym` release used by robot-gym
(`policies/g1_motion.pt`): a TorchScript module taking a 47-value observation
and returning 12 leg joint deltas at 50 Hz. This module ports its MuJoCo
deployment contract onto Isaac's articulation API.

Two things do not carry across from MuJoCo and are handled here:

* **Joint order.** The policy is indexed in MuJoCo's left-leg-then-right-leg
  order; Isaac interleaves the legs. `POLICY_JOINTS` is resolved by name.
* **Collision.** A referenced scan is visual geometry only. Without an
  explicit collider the robot falls through the floor, so the room's meshes
  get a static triangle-mesh collider before physics starts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOM_ASSET = "library-real"
ROOM_PRIM = "/World/library"
G1_PRIM = "/World/g1"

POLICY_PATH = Path("policies/g1_motion.pt")

# MuJoCo leg order the policy was trained with: left leg, then right
POLICY_JOINTS = (
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
)

# deploy_mujoco/configs/g1.yaml, kept verbatim so the port stays comparable
KPS = np.array([100, 100, 100, 150, 40, 40] * 2, dtype=np.float32)
KDS = np.array([2, 2, 2, 4, 2, 2] * 2, dtype=np.float32)
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2, dtype=np.float32)

ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
NUM_ACTIONS = 12
NUM_OBS = 47

PHYSICS_DT = 0.002
CONTROL_DECIMATION = 10  # 500 Hz physics, 50 Hz policy
GAIT_PERIOD_S = 0.8

# robot-gym clamps operator intent to this envelope before it reaches the policy
COMMAND_LIMITS = np.array([0.65, 0.35, 1.10], dtype=np.float32)

# Everything the policy does not drive is held at its default pose
POSTURE_STIFFNESS = 200.0
POSTURE_DAMPING = 10.0

SPAWN_HEIGHT = 0.80

# Where Gemini Robotics-ER 1.6 pointed, ray-cast onto the scan: the shelf face
# and its outward normal. The robot approaches along the normal.
SHELF_POINT = np.array([-2.21, 1.54, 1.39], dtype=np.float32)
SHELF_NORMAL = np.array([0.93, -0.09, 0.37], dtype=np.float32)

# Contact and solver settings. Humanoid gait is decided at the feet, and the
# PhysX defaults are tuned for general scenes rather than a 500 Hz biped.
GROUND_STATIC_FRICTION = 1.0
GROUND_DYNAMIC_FRICTION = 0.9
SOLVER_POSITION_ITERATIONS = 16
SOLVER_VELOCITY_ITERATIONS = 4

# deploy_real/configs limits, so the PD cannot ask for torque the hardware
# would refuse. Without these the implicit drive can apply anything.
LEG_EFFORT_LIMITS = np.array([88, 139, 88, 139, 50, 50] * 2, dtype=np.float32)

PALM_LINK = "/World/g1/right_wrist_yaw_link/right_hand_palm_link"

RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
RIGHT_HAND_JOINTS = (
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
)

# Arm postures, in RIGHT_ARM_JOINTS order. The shoulder-pitch sign puts the
# hand in front of the chest rather than behind it; ARM_POSES are measured by
# probe_reach, which reports the palm position in the body frame.
ARM_POSES = {
    "rest": np.zeros(7, dtype=np.float32),
    "reach": np.array([-0.9, -0.25, 0.0, 0.9, 0.0, 0.0, 0.0], dtype=np.float32),
    "carry": np.array([-0.7, -0.20, 0.0, 1.5, 0.0, 0.0, 0.0], dtype=np.float32),
}
HAND_OPEN = np.zeros(7, dtype=np.float32)
HAND_CLOSED = np.array([0.9, 0.9, 0.9, 0.9, 0.7, 0.7, 0.7], dtype=np.float32)


def add_static_collision(prim_path: str) -> int:
    """
    Give every mesh under ``prim_path`` a static triangle-mesh collider.

    A referenced scan arrives as renderable geometry with no physics. Triangle
    meshes are valid colliders for static bodies (the restriction to convex
    shapes applies to dynamic ones), so the scan can be collided against
    as-scanned without decomposition.

    :return: How many meshes were given a collider.
    """

    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    count = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        # "none" means no convex approximation: keep the scanned triangles
        mesh_collision.CreateApproximationAttr().Set(UsdPhysics.Tokens.none)
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
        count += 1
    return count


def quat_to_rotation(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix for an ``(w, x, y, z)`` quaternion."""

    w, x, y, z = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


class G1Locomotion:
    """
    Drive a G1 articulation from velocity commands using the pinned policy.

    The caller owns the world and the stepping loop; this owns the policy, the
    observation it expects, and the joint targets it produces.
    """

    def __init__(self, robot: object, policy_path: Path | str = POLICY_PATH) -> None:
        import torch

        self.robot = robot
        self.policy = torch.jit.load(str(policy_path))
        self.policy.eval()

        names = list(robot.dof_names)
        missing = [j for j in POLICY_JOINTS if j not in names]
        if missing:
            raise RuntimeError(f"G1 articulation is missing policy joints: {missing}")
        self.leg_index = np.array([names.index(j) for j in POLICY_JOINTS], dtype=np.int32)
        self.num_dof = len(names)
        self.arm_index = np.array(
            [names.index(j) for j in RIGHT_ARM_JOINTS if j in names], dtype=np.int32
        )
        self.hand_index = np.array(
            [names.index(j) for j in RIGHT_HAND_JOINTS if j in names], dtype=np.int32
        )
        # Everything the policy does not drive is held here, so a posture
        # change is one assignment rather than a second controller
        self.posture = np.zeros(self.num_dof, dtype=np.float32)

        self.command = np.zeros(3, dtype=np.float32)
        self.previous_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.target_leg_q = DEFAULT_ANGLES.copy()
        self.counter = 0
        self.time_s = 0.0

    def configure_gains(self) -> None:
        """Match the deployment PD gains; hold every other joint at its default."""

        stiffness = np.full(self.num_dof, POSTURE_STIFFNESS, dtype=np.float32)
        damping = np.full(self.num_dof, POSTURE_DAMPING, dtype=np.float32)
        stiffness[self.leg_index] = KPS
        damping[self.leg_index] = KDS
        self.robot.set_gains(kps=stiffness[None, :], kds=damping[None, :])

        # Cap leg torque at the hardware limits the deployment config declares
        try:
            efforts = np.full(self.num_dof, 300.0, dtype=np.float32)
            efforts[self.leg_index] = LEG_EFFORT_LIMITS
            self.robot.set_max_efforts(efforts[None, :])
        except Exception as error:  # noqa: BLE001 - report rather than fail the run
            print(f"[g1] could not set effort limits: {error}", flush=True)

    def initial_pose(self) -> np.ndarray:
        """Full-body joint vector with the legs in the policy's default stance."""

        pose = self.posture.copy()
        pose[self.leg_index] = DEFAULT_ANGLES
        return pose

    def set_arm(self, pose: str, hand_closed: bool = False) -> None:
        """Hold the right arm in a named posture and open or close the fingers."""

        target = ARM_POSES.get(pose, ARM_POSES["rest"])
        if self.arm_index.size:
            self.posture[self.arm_index] = target[: self.arm_index.size]
        if self.hand_index.size:
            grip = HAND_CLOSED if hand_closed else HAND_OPEN
            self.posture[self.hand_index] = grip[: self.hand_index.size]

    def set_command(self, vx: float, vy: float, yaw_rate: float) -> None:
        """Set the commanded body velocity, clamped to the deployment envelope."""

        self.command = np.clip(
            np.array([vx, vy, yaw_rate], dtype=np.float32), -COMMAND_LIMITS, COMMAND_LIMITS
        )

    def observation(self) -> np.ndarray:
        """Assemble the 47-value observation in the order the policy expects."""

        _, quat = self.robot.get_world_poses()
        quat = np.asarray(quat, dtype=np.float32)[0]
        # get_velocities returns one (M, 6) array: linear in [:3], angular in [3:]
        velocities = np.asarray(self.robot.get_velocities(), dtype=np.float32)[0]
        ang_vel_world = velocities[3:6]

        rotation = quat_to_rotation(quat)
        # The policy reads angular velocity and gravity in the base frame
        ang_vel_body = rotation.T @ ang_vel_world
        projected_gravity = rotation.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)

        joint_q = np.asarray(self.robot.get_joint_positions(), dtype=np.float32)[0][self.leg_index]
        joint_dq = np.asarray(self.robot.get_joint_velocities(), dtype=np.float32)[0][self.leg_index]

        phase = (self.time_s % GAIT_PERIOD_S) / GAIT_PERIOD_S

        obs = np.zeros(NUM_OBS, dtype=np.float32)
        obs[0:3] = ang_vel_body * ANG_VEL_SCALE
        obs[3:6] = projected_gravity
        obs[6:9] = self.command * CMD_SCALE
        obs[9:21] = (joint_q - DEFAULT_ANGLES) * DOF_POS_SCALE
        obs[21:33] = joint_dq * DOF_VEL_SCALE
        obs[33:45] = self.previous_action
        obs[45] = np.sin(2.0 * np.pi * phase)
        obs[46] = np.cos(2.0 * np.pi * phase)
        return obs

    def step(self, dt: float) -> None:
        """
        Advance one physics tick, running the policy every ``CONTROL_DECIMATION``.

        Joint targets are held between policy ticks, which is what makes the
        50 Hz policy behave the same way against a 500 Hz physics step.
        """

        import torch

        self.time_s += dt
        if self.counter % CONTROL_DECIMATION == 0:
            with torch.no_grad():
                obs = torch.from_numpy(self.observation()).unsqueeze(0)
                action = self.policy(obs).detach().numpy().squeeze(0)
            self.previous_action = action.astype(np.float32)
            self.target_leg_q = action * ACTION_SCALE + DEFAULT_ANGLES

        targets = self.posture.copy()
        targets[self.leg_index] = self.target_leg_q
        self.robot.set_joint_position_targets(targets[None, :])
        self.counter += 1

    def reset_state(self) -> None:
        """
        Clear the policy's memory so a respawn starts from a clean gait.

        The policy is recurrent (LSTM, hidden size 64) and keeps its state in
        module buffers across calls. Leaving it alone across a respawn feeds
        the memory of falling into the first step of the next attempt —
        measured at up to 1.94 of action difference for an identical
        observation, which is ~0.48 rad of joint target once scaled.
        """

        self.previous_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.target_leg_q = DEFAULT_ANGLES.copy()
        self.counter = 0
        self.time_s = 0.0
        for buffer_name in ("hidden_state", "cell_state"):
            buffer = getattr(self.policy, buffer_name, None)
            if buffer is not None:
                buffer.zero_()

    def hold_default(self) -> None:
        """Command the default stance and nothing else, for isolating gain problems."""

        targets = np.zeros(self.num_dof, dtype=np.float32)
        targets[self.leg_index] = DEFAULT_ANGLES
        self.robot.set_joint_position_targets(targets[None, :])

    def report_gains(self) -> str:
        """Read the gains back, since a silent no-op here looks like a policy failure."""

        kps, kds = self.robot.get_gains()
        kps = np.asarray(kps, dtype=np.float32)[0][self.leg_index]
        kds = np.asarray(kds, dtype=np.float32)[0][self.leg_index]
        return f"leg kps={kps.tolist()} kds={kds.tolist()}"

    def base_state(self) -> tuple[np.ndarray, float, float]:
        """Return base position, height, and tilt from upright in radians."""

        pos, quat = self.robot.get_world_poses()
        pos = np.asarray(pos, dtype=np.float32)[0]
        quat = np.asarray(quat, dtype=np.float32)[0]
        up = quat_to_rotation(quat) @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        tilt = float(np.arccos(np.clip(up[2], -1.0, 1.0)))
        return pos, float(pos[2]), tilt

    def heading(self) -> float:
        """World yaw of the robot's forward axis, in radians."""

        _, quat = self.robot.get_world_poses()
        forward = quat_to_rotation(np.asarray(quat, dtype=np.float32)[0]) @ np.array(
            [1.0, 0.0, 0.0], dtype=np.float32
        )
        return float(np.arctan2(forward[1], forward[0]))


def build_library_scene(
    world: object,
    spawn: tuple[float, float] = (2.0, 0.0),
    height: float = SPAWN_HEIGHT,
    ground: bool = True,
    room_collision: bool = True,
) -> tuple[object, int]:
    """
    Load the room and add a G1 standing on it.

    The articulation is not queryable until the world has been reset, so the
    controller is built by the caller after that — see :class:`G1Locomotion`.

    :return: ``(robot, collider_count)``
    """

    import antioch
    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.storage.native import get_assets_root_path

    antioch.load_asset(ROOM_ASSET, prim_path=ROOM_PRIM)
    colliders = add_static_collision(ROOM_PRIM) if room_collision else 0

    # The scan's floor is a photogrammetry surface: locally bumpy, and pierced
    # by stray geometry up to ~0.2 m below the floor plane. Walking on it
    # directly turns every footfall into a contact lottery, so a flat plane at
    # z=0 carries the feet and the scan supplies walls, furniture, and looks.
    if ground:
        world.scene.add_ground_plane(
            z_position=0.0,
            restitution=0.0,
            static_friction=GROUND_STATIC_FRICTION,
            dynamic_friction=GROUND_DYNAMIC_FRICTION,
        )

    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 300.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 500.0})

    root = get_assets_root_path()
    add_reference_to_stage(usd_path=f"{root}/Isaac/Robots/Unitree/G1/g1.usd", prim_path=G1_PRIM)

    robot = Articulation(
        prim_paths_expr=G1_PRIM,
        name="g1",
        positions=np.array([[spawn[0], spawn[1], height]], dtype=np.float32),
    )
    world.scene.add(robot)
    return robot, colliders
