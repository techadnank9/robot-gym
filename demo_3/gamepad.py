from __future__ import annotations

import multiprocessing as mp
import queue
import time
from typing import Any, Literal

from demo_3.schemas import Skill, TeleopFrame


class MacGamepad:
    """Crash-isolated SDL gamepad reader.

    pygame initializes SDL's video subsystem even when it is used only for a
    joystick. On macOS that conflicts with MuJoCo's Cocoa viewer when both live
    in the same process. A spawned helper owns SDL and sends validated input
    frames back to the simulator.
    """

    def __init__(self, player_id: Literal["p1", "p2"], index: int = 0) -> None:
        context = mp.get_context("spawn")
        self.player_id = player_id
        self._frames = context.Queue(maxsize=8)
        self._status = context.Queue(maxsize=2)
        self._stop = context.Event()
        self._process = context.Process(
            target=_gamepad_worker,
            args=(player_id, index, self._frames, self._status, self._stop),
            daemon=True,
            name=f"demo3-{player_id}-gamepad",
        )
        self._process.start()
        try:
            status = self._status.get(timeout=8.0)
        except queue.Empty as exc:
            self.close()
            raise RuntimeError("Gamepad helper did not start within eight seconds") from exc
        if status.get("type") != "ready":
            detail = str(status.get("detail") or "unknown gamepad startup failure")
            self.close()
            raise RuntimeError(detail)
        self._name = str(status["name"])
        self._last_frame = TeleopFrame.neutral(player_id)

    @property
    def name(self) -> str:
        return self._name

    def close(self) -> None:
        self._stop.set()
        process = getattr(self, "_process", None)
        if process is not None:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for pipe in (getattr(self, "_frames", None), getattr(self, "_status", None)):
            if pipe is not None:
                pipe.close()
                pipe.join_thread()

    def poll(self) -> TeleopFrame:
        latest: dict[str, Any] | None = None
        while True:
            try:
                latest = self._frames.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self._last_frame = TeleopFrame.model_validate(latest)
        if not self._process.is_alive() and self._last_frame.connected:
            self._last_frame = self._last_frame.model_copy(
                update={
                    "sequence": self._last_frame.sequence + 1,
                    "timestamp_s": time.monotonic(),
                    "connected": False,
                    "deadman": False,
                    "move_x": 0.0,
                    "move_y": 0.0,
                    "yaw": 0.0,
                }
            )
        return self._last_frame


def _gamepad_worker(
    player_id: Literal["p1", "p2"],
    index: int,
    frames: Any,
    status: Any,
    stop: Any,
) -> None:
    import os

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    try:
        import pygame

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= index:
            _put_latest(
                status,
                {
                    "type": "error",
                    "detail": (
                        f"No gamepad found at index {index}; connect it before launching Demo 3"
                    ),
                },
            )
            return
        joystick = pygame.joystick.Joystick(index)
        joystick.init()
        _put_latest(status, {"type": "ready", "name": joystick.get_name()})
        sequence = 0
        current_skill = Skill.WAIT
        previous_buttons: dict[int, bool] = {}
        while not stop.is_set():
            pygame.event.pump()
            sequence += 1
            left_x = _axis(joystick, 0)
            left_y = -_axis(joystick, 1)
            right_x = _axis(joystick, 2)
            hat_x, hat_y = _hat(joystick)
            if left_x == 0.0:
                left_x = hat_x
            if left_y == 0.0:
                left_y = hat_y
            left_trigger = _trigger(joystick, 4)
            right_trigger = _trigger(joystick, 5)
            deadman = _motion_active(left_x, left_y, right_x)
            if _pressed(joystick, 0, previous_buttons):
                current_skill = Skill.GRASP
            elif _pressed(joystick, 1, previous_buttons):
                current_skill = Skill.RELEASE
            elif _pressed(joystick, 2, previous_buttons):
                current_skill = Skill.NAVIGATE_GOAL
            elif _pressed(joystick, 3, previous_buttons):
                current_skill = Skill.RECOVER
            hand_close = max(
                right_trigger,
                1.0 if current_skill == Skill.GRASP else 0.0,
            )
            if left_trigger > 0.6:
                hand_close = 0.0
            frame = TeleopFrame(
                player_id=player_id,
                sequence=sequence,
                timestamp_s=time.monotonic(),
                connected=True,
                deadman=deadman,
                move_x=left_x,
                move_y=left_y,
                yaw=right_x,
                skill=current_skill,
                hand_close=hand_close,
            )
            _put_latest(frames, frame.model_dump(mode="json"))
            time.sleep(0.02)
    except Exception as exc:  # noqa: BLE001
        _put_latest(status, {"type": "error", "detail": f"Gamepad helper failed: {exc}"})
    finally:
        try:
            pygame.quit()
        except (NameError, Exception):
            pass


def _put_latest(pipe: Any, value: dict[str, Any]) -> None:
    try:
        pipe.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        pipe.get_nowait()
    except queue.Empty:
        pass
    try:
        pipe.put_nowait(value)
    except queue.Full:
        pass


def _axis(joystick: Any, index: int) -> float:
    if joystick.get_numaxes() <= index:
        return 0.0
    value = float(joystick.get_axis(index))
    deadzone = 0.12
    if abs(value) <= deadzone:
        return 0.0
    return max(-1.0, min(1.0, (abs(value) - deadzone) / (1.0 - deadzone))) * (
        -1.0 if value < 0 else 1.0
    )


def _trigger(joystick: Any, index: int) -> float:
    return max(0.0, min(1.0, (_axis(joystick, index) + 1.0) / 2.0))


def _hat(joystick: Any) -> tuple[float, float]:
    if joystick.get_numhats() <= 0:
        return 0.0, 0.0
    x, y = joystick.get_hat(0)
    return float(x), float(y)


def _motion_active(move_x: float, move_y: float, yaw: float) -> bool:
    return bool(move_x or move_y or yaw)


def _button(joystick: Any, index: int) -> bool:
    if joystick.get_numbuttons() <= index:
        return False
    return bool(joystick.get_button(index))


def _pressed(joystick: Any, index: int, previous: dict[int, bool]) -> bool:
    current = _button(joystick, index)
    before = previous.get(index, False)
    previous[index] = current
    return current and not before
