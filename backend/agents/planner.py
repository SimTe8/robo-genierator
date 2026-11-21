"""Gemini 4 Planner: proposes object-centric manipulation waypoints as structured output."""

from __future__ import annotations

import json
from typing import Sequence

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from schemas import RobotId, Waypoint


class FingerTargets(BaseModel):
    """Explicit per-finger closure overrides (avoids open dict schemas in Gemini response_schema)."""

    thumb: float = Field(default=1.0, ge=0.0, le=1.0)
    index: float = Field(default=1.0, ge=0.0, le=1.0)
    middle: float = Field(default=1.0, ge=0.0, le=1.0)
    ring: float = Field(default=1.0, ge=0.0, le=1.0)


class PlannerWaypoint(BaseModel):
    """Gemini-schema-safe waypoint model."""

    phase: str = Field(..., description="One of: pre_grasp, grasp, close, lift")
    position_offset: list[float] = Field(
        ...,
        description="[dx, dy, dz] offset in meters relative to the object center.",
    )
    orientation_rpy: list[float] = Field(
        ...,
        description="[roll, pitch, yaw] target orientation in radians.",
    )
    gripper_target: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Normalized gripper/finger closure (0.0 = open, 1.0 = closed).",
    )
    finger_targets: FingerTargets | None = Field(
        default=None,
        description="Optional per-finger closure overrides for wonik_allegro.",
    )


class _RawPlannerOutput(BaseModel):
    """Structured output returned by the Gemini 4 Planner."""

    reasoning: str = Field(
        ...,
        description="Brief kinematic reasoning explaining approach angle, offsets, and fault compensation.",
    )
    waypoints: list[PlannerWaypoint] = Field(
        ...,
        description="Exactly 4 waypoints in order: pre_grasp, grasp, close, lift.",
    )


class PlannerOutput(BaseModel):
    """Validated Planner output using shared Waypoint contract."""

    reasoning: str
    waypoints: list[Waypoint]


PLANNER_SYSTEM_PROMPT = """You are an expert robotic manipulation motion planner controlling simulated MuJoCo robots.

Your job is to output EXACTLY 4 object-centric waypoints in order:
1. `pre_grasp`: hover above or near the object with gripper/fingers OPEN (`gripper_target: 0.0`).
2. `grasp`: move end-effector/palm to the grasp pose around the object (`gripper_target: 0.0`).
3. `close`: hold the grasp pose while closing gripper/fingers (`gripper_target: 1.0`).
4. `lift`: lift the grasped object vertically (`position_offset[2] >= 0.16`, `gripper_target: 1.0`).

COORDINATE CONVENTIONS:
- `position_offset = [dx, dy, dz]` is in METERS relative to the target object's initial center `(x, y, z)`.
  - `+z` is UP, `-z` is DOWN toward the table.
  - For a standard top-down grasp, `pre_grasp` is typically `[0.0, 0.0, 0.14]`, `grasp` is `[0.0, 0.0, 0.0]`, `close` is `[0.0, 0.0, 0.0]`, and `lift` is `[0.0, 0.0, 0.18]`.
- `orientation_rpy = [roll, pitch, yaw]` in RADIANS (intrinsic/extrinsic XYZ Euler).
  - For `franka_emika_panda`: `[3.14159, 0.0, 1.5708]` points the parallel-jaw gripper straight DOWN (`-Z`). Adjusting `pitch` (`-0.4` to `+0.4`) or `roll` (`3.14 +/- 0.4`) tilts the approach angle, which is critical when an arm joint (like `joint4` or `joint2`) is disabled/locked and a pure vertical top-down grasp is kinematically unreachable! Adjusting `yaw` rotates the gripper jaws around Z to align with box edges.
  - For `wonik_allegro`: `[0.0, 1.5708, 0.0]` points the palm downward/forward over the object. Use `finger_targets` (`{"thumb": ..., "index": ..., "middle": ..., "ring": ...}`) to adapt around disabled fingers!
- Avoid colliding with the table (`table_top_z`). Keep `grasp` `dz` safely above the table surface.
- When given prior attempt feedback from the Visual Critic, apply the suggested spatial corrections directly to fix the physical failure mode."""


IN_HAND_ROTATION_SYSTEM_PROMPT = """You are an expert dexterous manipulation planner controlling a palm-up 4-fingered Wonik Allegro right hand in MuJoCo.

Your job is to output EXACTLY 4 sequential finger-gait keyframes (labeled `pre_grasp`, `grasp`, `close`, `lift` for schema compatibility, representing Gait Step 1, Step 2, Step 3, Step 4) that will be **looped cyclically** to ratchet-spin the multi-colored cube continuously in-hand WITHOUT dropping it off the palm!

PARAMETERIZATION FOR EACH GAIT KEYFRAME:
- `finger_targets`: REQUIRED dict with `{thumb, index, middle, ring}`, each in `[0.0, 1.0]` controlling flexion/curl (`0.0` = extended open, `0.45` = gentle cradle, `0.85` = strong push/curl).
- `orientation_rpy = [ff_sweep, mf_sweep, rf_sweep]`: lateral side-to-side abduction sweep in RADIANS (`-0.47` to `+0.47`) for index (`ff`), middle (`mf`), and ring (`rf`) fingers.
- `position_offset = [thumb_opp, thumb_sweep, 0.0]`:
  - `thumb_opp` (`tha0`) in `[0.263, 1.396]` rad (higher = thumb rotated deeper in opposition across palm).
  - `thumb_sweep` (`tha1`) in `[-0.105, 1.163]` rad (sweeps thumb laterally to push/spin a corner of the cube!).
- `gripper_target`: fallback curl `[0.0, 1.0]` (set to `0.5`).

CYCLIC RATCHETING PHYSICS TIPS:
- Because the 4-step gait repeats cyclically, you MUST break symmetry between the power stroke and return stroke:
  - Power stroke (Steps 1->2): push/sweep with driving fingers curled (`0.70–0.88`) while stabilizing fingers hold lightly (`0.35–0.50`).
  - Unloaded reset (Steps 3->4): stabilize the cube with opposing fingers (`0.55–0.70`) while lifting/extending the driving fingers (`0.15–0.28`) as they sweep back, so they do NOT unwind the cube!
- Never open all 4 fingers simultaneously (`< 0.25`) or push aggressively into the wrist edge, or the cube will drop!
- When given Visual Critic feedback from a prior attempt, adjust the exact finger curls and sweeps to prevent dropping or increase net per-cycle rotation."""


def plan_waypoints(
    client: genai.Client,
    planner_model: str,
    task_prompt: str,
    robot_id: RobotId,
    kinematic_summary: str,
    disabled_joints: Sequence[str],
    history: Sequence[dict] | None = None,
    task_type: str = "pick_and_lift",
) -> PlannerOutput:
    """Call Gemini 4 Planner to generate or refine 4 waypoints or finger-gait keyframes."""
    system_prompt = (
        IN_HAND_ROTATION_SYSTEM_PROMPT
        if task_type == "in_hand_rotation"
        else PLANNER_SYSTEM_PROMPT
    )
    user_parts = [
        f"TASK TYPE: {task_type}",
        f"TASK: {task_prompt}",
        f"ROBOT: {robot_id}",
        f"DISABLED JOINTS (LOCKED IN PLACE): {list(disabled_joints) or 'none'}",
        "",
        "KINEMATIC & SCENE SUMMARY:",
        kinematic_summary,
    ]

    if history:
        user_parts.append("")
        user_parts.append("PREVIOUS ATTEMPTS & CRITIC FEEDBACK (use this to fix the failure):")
        for item in history:
            user_parts.append(
                json.dumps(
                    {
                        "attempt_id": item["attempt_id"],
                        "waypoints": item["waypoints"],
                        "telemetry_summary": item["telemetry_summary"],
                        "decision": item["decision"],
                        "failure_mode": item.get("failure_mode"),
                        "critic_feedback": item["reason"],
                    },
                    indent=2,
                )
            )
        user_parts.append(
            "\nPropose an UPDATED 4-step plan that directly corrects the observed physical failure."
        )
    else:
        user_parts.append(
            "\nPropose an initial 4-step plan (`pre_grasp`, `grasp`, `close`, `lift`)."
        )

    response = client.models.generate_content(
        model=planner_model,
        contents="\n".join(user_parts),
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=_RawPlannerOutput,
            temperature=0.25,
        ),
    )

    if response.parsed is not None and isinstance(response.parsed, _RawPlannerOutput):
        raw = response.parsed
    else:
        raw = _RawPlannerOutput.model_validate_json(response.text)

    phases = ["pre_grasp", "grasp", "close", "lift"]
    converted: list[Waypoint] = []
    for idx, pwp in enumerate(raw.waypoints[:4]):
        phase = pwp.phase if pwp.phase in phases else phases[idx]
        pos = (list(pwp.position_offset) + [0.0, 0.0, 0.0])[:3]
        rpy = (list(pwp.orientation_rpy) + [3.14159, 0.0, 1.5708])[:3]
        ft = pwp.finger_targets.model_dump() if pwp.finger_targets is not None else None
        converted.append(
            Waypoint(
                phase=phase,  # type: ignore[arg-type]
                position_offset=pos,
                orientation_rpy=rpy,
                gripper_target=float(min(1.0, max(0.0, pwp.gripper_target))),
                finger_targets=ft,
            )
        )

    return PlannerOutput(reasoning=raw.reasoning, waypoints=converted)
