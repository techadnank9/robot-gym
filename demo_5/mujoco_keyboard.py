from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Literal

from demo_3.schemas import Skill, TeleopFrame


class MujocoKeyboard:
    """Teleop frames driven by MuJoCo's native viewer key callback.

    MuJoCo exposes key presses but not releases to the Python callback. Motion
    is therefore latched until another direction is selected or Space is
    pressed. This avoids depending on macOS key-repeat behavior.
    """

    _GLFW_RIGHT = 262
    _GLFW_LEFT = 263
    _GLFW_DOWN = 264
    _GLFW_UP = 265
    _MOTION_KEYS = {
        _GLFW_UP: ("move_y", 1.0),
        _GLFW_DOWN: ("move_y", -1.0),
        _GLFW_RIGHT: ("move_x", 1.0),
        _GLFW_LEFT: ("move_x", -1.0),
        ord("E"): ("yaw", 1.0),
        ord("Q"): ("yaw", -1.0),
    }
    _SKILL_KEYS = {
        ord("G"): (Skill.GRASP, 1.0),
        ord("C"): (Skill.NAVIGATE_GOAL, 1.0),
        ord("R"): (Skill.RELEASE, 0.0),
        ord("U"): (Skill.RECOVER, None),
    }

    def __init__(
        self,
        player_id: Literal["p1", "p2"],
        *,
        motion_timeout_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if motion_timeout_s is not None and motion_timeout_s <= 0:
            raise ValueError("motion_timeout_s must be positive")
        self.player_id = player_id
        self.motion_timeout_s = motion_timeout_s
        self._clock = clock
        self._lock = threading.Lock()
        self._sequence = 0
        self._motion = {"move_x": 0.0, "move_y": 0.0, "yaw": 0.0}
        self._motion_updated_s = 0.0
        self._skill = Skill.WAIT
        self._hand_close = 0.0
        self._reset_requested = False
        self._notice: str | None = None
        self._last_notice: str | None = None

    def on_key(self, keycode: int) -> None:
        """Receive one GLFW keycode from ``mujoco.viewer.launch_passive``."""

        # GLFW's printable key constants match uppercase ASCII.
        if ord("a") <= keycode <= ord("z"):
            keycode -= ord("a") - ord("A")
        now = self._clock()
        with self._lock:
            motion = self._MOTION_KEYS.get(keycode)
            if motion is not None:
                axis, value = motion
                self._motion[axis] = value
                self._motion_updated_s = now
                self._set_notice(
                    {
                        self._GLFW_UP: "forward",
                        self._GLFW_DOWN: "backward",
                        self._GLFW_LEFT: "left",
                        self._GLFW_RIGHT: "right",
                        ord("Q"): "turn left",
                        ord("E"): "turn right",
                    }[keycode]
                )
                return
            if keycode == ord(" "):
                self._clear_motion()
                self._set_notice("stop")
                return
            if keycode == ord("X"):
                self._clear_motion()
                self._reset_requested = True
                self._set_notice("payload reset requested")
                return
            skill = self._SKILL_KEYS.get(keycode)
            if skill is not None:
                self._skill = skill[0]
                if skill[1] is not None:
                    self._hand_close = skill[1]
                self._set_notice(skill[0].value.replace("_", " "))

    def poll(self) -> TeleopFrame:
        now = self._clock()
        with self._lock:
            if (
                self.motion_timeout_s is not None
                and now - self._motion_updated_s > self.motion_timeout_s
            ):
                self._clear_motion()
            self._sequence += 1
            move_x = self._motion["move_x"]
            move_y = self._motion["move_y"]
            yaw = self._motion["yaw"]
            return TeleopFrame(
                player_id=self.player_id,
                sequence=self._sequence,
                timestamp_s=now,
                connected=True,
                deadman=bool(move_x or move_y or yaw),
                move_x=move_x,
                move_y=move_y,
                yaw=yaw,
                skill=self._skill,
                hand_close=self._hand_close,
            )

    def consume_reset_request(self) -> bool:
        with self._lock:
            requested = self._reset_requested
            self._reset_requested = False
            return requested

    def consume_notice(self) -> str | None:
        with self._lock:
            notice = self._notice
            self._notice = None
            return notice

    def _set_notice(self, notice: str) -> None:
        if notice != self._last_notice:
            self._notice = notice
            self._last_notice = notice

    def _clear_motion(self) -> None:
        self._motion["move_x"] = 0.0
        self._motion["move_y"] = 0.0
        self._motion["yaw"] = 0.0
        self._motion_updated_s = 0.0
