from __future__ import annotations

import math
import time
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


PROTOCOL_VERSION = "3.0"


class PlayerMode(str, Enum):
    POLICY = "policy"
    HUMAN = "human"


class MatchMode(str, Enum):
    AI_VS_AI = "ai-vs-ai"
    HUMAN_VS_AI = "human-vs-ai"
    HUMAN_VS_HUMAN = "human-vs-human"


class MatchPhase(str, Enum):
    LOBBY = "lobby"
    COUNTDOWN = "countdown"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


class Skill(str, Enum):
    WAIT = "wait"
    NAVIGATE_OBJECT = "navigate_object"
    GRASP = "grasp"
    NAVIGATE_GOAL = "navigate_goal"
    RELEASE = "release"
    RECOVER = "recover"


class PlayerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_id: Literal["p1", "p2"]
    display_name: str = Field(min_length=1, max_length=48)
    mode: PlayerMode = PlayerMode.POLICY
    model_adapter: str = Field(default="gemini-er", min_length=1, max_length=64)
    model_name: str | None = Field(default=None, max_length=128)

    @property
    def public_model_name(self) -> str:
        if self.mode == PlayerMode.HUMAN:
            return "Mac gamepad"
        return self.model_name or "Gemini Robotics-ER"


class TeleopFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal["3.0"] = PROTOCOL_VERSION
    player_id: Literal["p1", "p2"]
    sequence: int = Field(ge=0)
    timestamp_s: float
    connected: bool = True
    deadman: bool = False
    move_x: float = Field(ge=-1.0, le=1.0)
    move_y: float = Field(ge=-1.0, le=1.0)
    yaw: float = Field(ge=-1.0, le=1.0)
    skill: Skill = Skill.WAIT
    hand_close: float = Field(default=0.0, ge=0.0, le=1.0)
    camera_reset: bool = False

    @field_validator("timestamp_s")
    @classmethod
    def finite_timestamp(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("timestamp_s must be finite")
        return value

    @classmethod
    def neutral(cls, player_id: Literal["p1", "p2"], sequence: int = 0) -> "TeleopFrame":
        return cls(
            player_id=player_id,
            sequence=sequence,
            timestamp_s=time.monotonic(),
            connected=True,
            deadman=False,
            move_x=0.0,
            move_y=0.0,
            yaw=0.0,
        )


class PlayerStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_id: Literal["p1", "p2"]
    display_name: str
    mode: PlayerMode
    model_name: str
    connected: bool = True
    disqualified: bool = False
    fallen: bool = False
    carrying: bool = False
    checkpoint_crossed: bool = False
    delivered: bool = False
    current_skill: Skill = Skill.WAIT
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    api_calls_remaining: int | None = Field(default=None, ge=0, le=5)
    rationale: str = ""


class MatchEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal["3.0"] = PROTOCOL_VERSION
    event_type: str = Field(min_length=1, max_length=80)
    match_id: str
    timestamp_s: float
    simulation_time_s: float
    player_id: Literal["p1", "p2"] | None = None
    payload: dict[str, object] = Field(default_factory=dict)


def match_mode(players: tuple[PlayerConfig, PlayerConfig]) -> MatchMode:
    human_count = sum(player.mode == PlayerMode.HUMAN for player in players)
    if human_count == 0:
        return MatchMode.AI_VS_AI
    if human_count == 2:
        return MatchMode.HUMAN_VS_HUMAN
    return MatchMode.HUMAN_VS_AI
