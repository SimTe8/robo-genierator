"""Shared Pydantic contracts between FastAPI backend and React frontend."""

from __future__ import annotations

from typing import Any, Literal

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


# ---------------------------------------------------------------------------
# Web UI -> Gemini 4 planner hand-off (object + task library; robot is the Allegro hand)
# ---------------------------------------------------------------------------


class TaskParamInfo(BaseModel):
    key: str
    label: str
    unit: str
    min: float
    max: float
    step: float
    default: float


class TaskPresetInfo(BaseModel):
    id: str
    name: str
    skill: str
    summary: str
    goal: str
    fixture: str | None
    initial_pose: str
    params: list[TaskParamInfo]
    prompt_template: str
    criteria: list[str]
    robots: list[str]


class ObjectTaskInfo(BaseModel):
    """A task preset as offered for one object (names/params resolved for that object)."""

    task_id: str
    name: str
    params: dict[str, float]


class ObjectInfo(BaseModel):
    id: str
    name: str
    category: str
    description: str
    dims_cm: list[float]
    mass_kg: float
    grasp_note: str
    parts: list[str]
    articulated: bool
    thumbnail_url: str
    tasks: list[ObjectTaskInfo]
    physics: dict[str, Any] = Field(description="PG-VLM concepts: material, fragility, deformability, transparency, contents, can_contain_liquid, sealed")


class RobotInfo(BaseModel):
    id: str
    name: str
    kind: str
    description: str
    fingers: list[str]
    license: str


class Catalog(BaseModel):
    robots: list[RobotInfo]
    objects: list[ObjectInfo]
    tasks: list[TaskPresetInfo]


class PlannerRequest(BaseModel):
    """What the UI sends: object + task + the (editable) prompt. No conditions — the planner decides those."""

    robot_id: str = "wonik_allegro"
    object_id: str
    task_id: str
    task_params: dict[str, float] = Field(default_factory=dict)
    prompt: str = Field(min_length=1)


class PlannerBrief(BaseModel):
    """Everything the Gemini 4 planner receives for a job (saved as outputs/jobs/<job_id>/job.json)."""

    job_id: str
    created_at: str
    robot: RobotInfo
    prompt: str
    object: ObjectInfo
    task: TaskPresetInfo
    task_params: dict[str, float]
    success_criteria: list[str]
    default_scene: dict[str, Any] = Field(description="Object/fixture pose of the default scene (planner may randomise).")
    kinematic_summary: str
    how_to_build_scene: str


class CheckResult(BaseModel):
    """Simulator ground truth for one attempt (sim/check.py)."""

    success: bool
    checks: dict[str, bool]
    metrics: dict[str, float]


class JobEvent(BaseModel):
    """One line of a job's event stream (outputs/jobs/<job_id>/events.jsonl, streamed to the UI over SSE).

    The planner emits these. `iteration` carries the shared IterationEvent when an attempt finishes.
    """

    job_id: str
    seq: int
    kind: Literal["status", "attempt", "done", "error"]
    message: str
    iteration: IterationEvent | None = None
    check: CheckResult | None = None
    data: dict[str, Any] | None = None
