"""Shared Pydantic contracts between FastAPI backend and React frontend."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RobotId = Literal["franka_emika_panda", "wonik_allegro"]
TaskType = Literal["pick_and_lift", "in_hand_rotation"]
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
    task_type: TaskType = "pick_and_lift"
    robot_id: RobotId = "franka_emika_panda"
    scene_seed: int = 42
    object_shape: Literal["auto", "box", "cylinder", "sphere"] = "auto"
    disabled_joints: list[str] = Field(
        default_factory=list,
        description="Joint or finger names to lock/disable via live fault injection.",
    )
    max_attempts: int = Field(default=4, ge=1, le=12)
    target_rotation_deg: float = Field(default=60.0, ge=15.0, le=720.0)
    feedback_mode: Literal["one_shot", "text_only", "video"] = "video"
    planner_model: str = "models/gemini-4-carbon"
    critic_model: str = "models/gemini-4-neon"


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
    decision_confidence: float | None = None
    decision_probabilities: dict[str, float] | None = None
    failure_mode: str | None = None
    grasp_score: int | None = None
    trajectory_path: str | None = None
