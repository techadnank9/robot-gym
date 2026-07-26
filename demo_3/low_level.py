from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from demo_2.policy_sil import gravity_orientation
from demo_3.model import LEG_JOINT_NAMES, TORQUE_LIMITS, project_root


@dataclass
class PlayerPolicyState:
    action: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))
    target: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))
    observation: np.ndarray = field(default_factory=lambda: np.zeros(47, dtype=np.float32))


class DualUnitreeLocomotion:
    """Runs one pinned official Unitree policy instance for each shared-world G1."""

    def __init__(
        self,
        model: Any,
        data: Any,
        mujoco_module: Any,
        *,
        command_limits: tuple[float, float, float] = (0.65, 0.35, 1.10),
    ) -> None:
        self.model = model
        self.data = data
        self.mujoco = mujoco_module
        self.command_limits = np.asarray(command_limits, dtype=np.float32)
        if (
            self.command_limits.shape != (3,)
            or not np.all(np.isfinite(self.command_limits))
            or np.any(self.command_limits <= 0)
        ):
            raise ValueError("command_limits must contain three positive finite values")
        root = project_root() / "demo_2" / "vendor" / "unitree_rl_gym"
        config_path = root / "deploy" / "deploy_mujoco" / "configs" / "g1.yaml"
        policy_path = root / "deploy" / "pre_train" / "g1" / "motion.pt"
        if not config_path.is_file() or not policy_path.is_file():
            raise FileNotFoundError(
                "Pinned Unitree policy is missing. Run scripts/setup_demo_2_sil.sh first."
            )
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Demo 3 policy locomotion requires PyTorch") from exc
        self.torch = torch
        first = torch.jit.load(str(policy_path))
        second = copy.deepcopy(first)
        for policy in (first, second):
            policy.eval()
            if hasattr(policy, "reset_memory"):
                policy.reset_memory()
        self.policies = {"p1": first, "p2": second}
        default = np.asarray(self.config["default_angles"], dtype=np.float32)
        self.states = {
            "p1": PlayerPolicyState(target=default.copy()),
            "p2": PlayerPolicyState(target=default.copy()),
        }
        self.commands = {
            "p1": np.zeros(3, dtype=np.float32),
            "p2": np.zeros(3, dtype=np.float32),
        }
        self._leg_qpos: dict[str, np.ndarray] = {}
        self._leg_dof: dict[str, np.ndarray] = {}
        self._base_qpos: dict[str, int] = {}
        self._base_dof: dict[str, int] = {}
        self._leg_actuator: dict[str, np.ndarray] = {}
        for player_id in ("p1", "p2"):
            joints = [model.joint(f"{player_id}_{name}") for name in LEG_JOINT_NAMES]
            self._leg_qpos[player_id] = np.asarray([joint.qposadr[0] for joint in joints])
            self._leg_dof[player_id] = np.asarray([joint.dofadr[0] for joint in joints])
            self._base_qpos[player_id] = int(model.joint(f"{player_id}_floating_base_joint").qposadr[0])
            self._base_dof[player_id] = int(model.joint(f"{player_id}_floating_base_joint").dofadr[0])
            self._leg_actuator[player_id] = np.asarray(
                [model.actuator(f"{player_id}_{name}").id for name in LEG_JOINT_NAMES]
            )

    @property
    def decimation(self) -> int:
        return int(self.config["control_decimation"])

    def set_command(self, player_id: str, vx: float, vy: float, yaw_rate: float) -> None:
        command = np.asarray([vx, vy, yaw_rate], dtype=np.float32)
        if not np.all(np.isfinite(command)):
            raise ValueError("locomotion command must be finite")
        command = np.clip(command, -self.command_limits, self.command_limits)
        self.commands[player_id] = command

    def apply_torques(self) -> None:
        kps = np.asarray(self.config["kps"], dtype=np.float32)
        kds = np.asarray(self.config["kds"], dtype=np.float32)
        limits = np.asarray(TORQUE_LIMITS, dtype=np.float32)
        for player_id, state in self.states.items():
            q = self.data.qpos[self._leg_qpos[player_id]]
            dq = self.data.qvel[self._leg_dof[player_id]]
            torque = np.clip((state.target - q) * kps - dq * kds, -limits, limits)
            self.data.ctrl[self._leg_actuator[player_id]] = torque

    def update(self, simulation_time_s: float) -> None:
        for player_id in ("p1", "p2"):
            self._update_player(player_id, simulation_time_s)

    def _update_player(self, player_id: str, simulation_time_s: float) -> None:
        state = self.states[player_id]
        q = self.data.qpos[self._leg_qpos[player_id]]
        dq = self.data.qvel[self._leg_dof[player_id]]
        base_qpos = self._base_qpos[player_id]
        base_dof = self._base_dof[player_id]
        quat = self.data.qpos[base_qpos + 3 : base_qpos + 7]
        period = 0.8
        phase = simulation_time_s % period / period
        obs = state.observation
        obs[:3] = self.data.qvel[base_dof + 3 : base_dof + 6] * float(
            self.config["ang_vel_scale"]
        )
        obs[3:6] = gravity_orientation(quat)
        obs[6:9] = self.commands[player_id] * np.asarray(
            self.config["cmd_scale"], dtype=np.float32
        )
        default = np.asarray(self.config["default_angles"], dtype=np.float32)
        obs[9:21] = (q - default) * float(self.config["dof_pos_scale"])
        obs[21:33] = dq * float(self.config["dof_vel_scale"])
        obs[33:45] = state.action
        obs[45:47] = [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)]
        with self.torch.no_grad():
            output = self.policies[player_id](self.torch.from_numpy(obs).unsqueeze(0))
        state.action = output.detach().cpu().numpy().squeeze().astype(np.float32)
        state.target = (
            state.action * float(self.config["action_scale"]) + default
        )
