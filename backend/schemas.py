"""Shared Pydantic contracts between FastAPI backend and React frontend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RobotId = Literal["franka_emika_panda", "wonik_allegro"]
TaskType = Literal["pick_and_lift", "obstacle_grasp", "in_hand_rotation", "in_hand_spindle"]
DecisionType = Literal["ACCEPT_TRAJECTORY", "REFINE_GRASP", "GIVE_UP"]
FeedbackMode = Literal["one_shot", "text_only", "video"]
TaskTypeOverride = Literal["auto", "pick_and_lift", "obstacle_grasp", "in_hand_rotation", "in_hand_spindle"]
ObjectShapeOverride = Literal["auto", "box", "cylinder", "sphere"]


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
    duration_s: float = Field(
        default=0.6,
        ge=0.2,
        le=2.0,
        description="Duration of this waypoint/gait step in seconds (larger = slower, gentler motion).",
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
    chunk_step_deg: float | None = Field(
        default=None,
        ge=30.0,
        le=180.0,
        description="Optional receding-horizon checkpoint milestone in degrees (e.g. 90 for 360 deg).",
    )
    feedback_mode: FeedbackMode = "video"
    planner_model: str = "models/gemini-4-carbon"
    critic_model: str = "models/gemini-4-neon"


class IterationEvent(BaseModel):
    """Shared backend <-> frontend event emitted after each closed-loop attempt."""

    attempt_id: int
    chunk_id: int | None = None
    model_used: str
    waypoints: list[Waypoint] = Field(default_factory=list)
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


# ---------------------------------------------------------------------------
# Web UI -> Gemini 4 planner hand-off (object + task library)
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
    physics: dict[str, Any] = Field(
        description="PG-VLM concepts: material, fragility, deformability, transparency, contents, can_contain_liquid, sealed"
    )


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


class PlannerSettings(BaseModel):
    """User-adjustable controls for the Gemini 4 Planner-Critic loop."""

    max_attempts: int = Field(default=4, ge=1, le=12)
    feedback_mode: FeedbackMode = "video"
    planner_model: str = "models/gemini-4-neon"
    critic_model: str = "models/gemini-4-hydrogen"
    scene_seed: int = 42
    disabled_joints: list[str] = Field(default_factory=list)
    chunk_step_deg: float | None = Field(default=None, ge=30.0, le=180.0)
    dr_episodes: int = Field(default=0, ge=0, le=20)
    task_type_override: TaskTypeOverride = "auto"
    object_shape_override: ObjectShapeOverride = "auto"


class PlannerRequest(BaseModel):
    """What the UI sends: object + task + prompt + optional planner-critic loop controls."""

    robot_id: str = "wonik_allegro"
    object_id: str
    task_id: str
    task_params: dict[str, float] = Field(default_factory=dict)
    prompt: str = Field(min_length=1)
    settings: PlannerSettings = Field(default_factory=PlannerSettings)


class PlannerBrief(BaseModel):
    """Everything the Gemini 4 planner receives for a job (saved as outputs/jobs/<job_id>/job.json)."""

    job_id: str
    created_at: str
    robot: RobotInfo
    prompt: str
    object: ObjectInfo
    task: TaskPresetInfo
    task_params: dict[str, float]
    settings: PlannerSettings = Field(default_factory=PlannerSettings)
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
    """One line of a job's event stream (outputs/jobs/<job_id>/events.jsonl, streamed to the UI over SSE)."""

    job_id: str
    seq: int
    kind: Literal["status", "terminal", "attempt", "done", "error"]
    message: str
    iteration: IterationEvent | None = None
    check: CheckResult | None = None
    data: dict[str, Any] | None = None
