"""Shared Pydantic contracts between FastAPI backend and React frontend."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RobotId = Literal["franka_emika_panda", "wonik_allegro"]
DecisionType = Literal["ACCEPT_TRAJECTORY", "REFINE_GRASP", "GIVE_UP"]


class Waypoint(BaseModel):
    """Object-centric 6-DoF grasp/manipulation waypoint and gripper/finger target."""

    phase: Literal["pre_grasp", "grasp", "close", "lift"]
    position_offset: list[float] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="[dx, dy, dz] offset in meters relative to the object center.",
    )
    orientation_rpy: list[float] = Field(
        default_factory=lambda: [3.14159, 0.0, 0.0],
        min_length=3,
        max_length=3,
        description="[roll, pitch, yaw] target orientation in radians.",
    )
    gripper_target: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Normalized gripper/finger closure (0.0 = open, 1.0 = closed).",
    )
    finger_targets: dict[str, float] | None = Field(
        default=None,
        description="Optional per-finger closure overrides for multi-fingered hands.",
    )


class RunRequest(BaseModel):
    """Input configuration for a closed-loop trajectory generation run."""

    task_prompt: str = Field(
        default="Pick up the object and lift it 15 cm above the table.",
        min_length=1,
    )
    robot_id: RobotId = "franka_emika_panda"
    scene_seed: int = 42
    disabled_joints: list[str] = Field(
        default_factory=list,
        description="Joint or finger names to lock/disable via live fault injection.",
    )
    max_attempts: int = Field(default=4, ge=1, le=8)
    feedback_mode: Literal["one_shot", "text_only", "video"] = "video"


class IterationEvent(BaseModel):
    """Shared backend <-> frontend event emitted after each closed-loop attempt."""

    attempt_id: int
    model_used: str
    waypoints: list[Waypoint]
    video_url: str
    telemetry_summary: str
    sim_success: bool
    decision: DecisionType
    reason: str
    latency_s: float
