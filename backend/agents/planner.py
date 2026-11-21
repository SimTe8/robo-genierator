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
    duration_s: float = Field(
        default=0.6,
        ge=0.2,
        le=2.0,
        description="Step duration in seconds (0.35=fast, 0.6=normal, 1.0+=slow & gentle).",
    )
    finger_targets: FingerTargets | None = Field(
        default=None,
        description="Optional per-finger closure overrides for wonik_allegro.",
    )


class _RawPlannerOutput(BaseModel):
    """Structured output returned by the Gemini 4 Planner."""

    reasoning: str = Field(
        ...,
        description="Brief kinematic reasoning explaining approach angle, offsets, and task strategy.",
    )
    waypoints: list[PlannerWaypoint] = Field(
        ...,
        description="4 to 7 sequential waypoints accomplishing the full task (label each phase as one of: pre_grasp, grasp, close, lift).",
    )


class PlannerOutput(BaseModel):
    """Validated Planner output using shared Waypoint contract."""

    reasoning: str
    waypoints: list[Waypoint]


PLANNER_SYSTEM_PROMPT = """You are an expert robotic manipulation motion planner controlling simulated MuJoCo robots.

Your job is to output a sequence of **4 to 7** object-centric waypoints (labeling each `phase` as one of `pre_grasp`, `grasp`, `close`, or `lift`) that cleanly accomplish the requested TASK!

TASK-SPECIFIC MOTION PATTERNS:
- **Pick & Lift / Obstacle Grasp** (4 waypoints):
  1. `pre_grasp`: hover above or in front of the object (`gripper_target: 0.0`).
  2. `grasp`: move to object center (`gripper_target: 0.0`).
  3. `close`: hold pose while closing gripper (`gripper_target: 1.0`).
  4. `lift`: lift vertically (`position_offset[2] >= 0.16`, `gripper_target: 1.0`).
- **Press Button (`press_button`)** (4 waypoints):
  - Close the gripper (`gripper_target: 1.0`) to form a rigid finger probe, hover above the specific target button (`button1`/`button2`/`button3` coordinates from the scene geometry!), press STRAIGHT DOWN onto the button (`dz` down into the button travel), and **HOLD IT PRESSED on the final waypoint** (do NOT lift back up — the simulator checks if the button is depressed at the end!).
- **Open Drawer (`open_drawer`)** (4–5 waypoints):
  - Approach the drawer handle (`gripper_target: 0.0`), move onto handle, close firmly (`gripper_target: 1.0`), then **pull straight back along the drawer's slide axis (`-X`)** at handle height (do NOT lift vertically!).
- **Turn Valve / Dial / Lid (`turn_joint`)** (4–5 waypoints):
  - Hover above knob/valve, descend onto it (`gripper_target: 0.0`), grip (`gripper_target: 1.0`), then **rotate `orientation_rpy[2]` (yaw)** by the target angle while holding position!
- **Pick & Place / Insert Key (`place_on_pad`, `place_in_bin`, `insert_key`)** (6–7 waypoints):
  - `pre_grasp` -> `grasp` -> `close` -> `lift` (carry above target pad/keyhole) -> move down to target pad/keyhole offset (relative to initial object center!) -> open gripper (`gripper_target: 0.0`) or push in!

COORDINATE CONVENTIONS:
- `position_offset = [dx, dy, dz]` is ALWAYS in METERS relative to the target object's INITIAL center `(x, y, z)`.
  - If a fixture (green pad, lock keyhole, button) is at world pos `(fx, fy, fz)` and initial object is at `(ox, oy, oz)`, the offset to reach the fixture is `[fx - ox, fy - oy, fz - oz]`!
  - `+z` is UP, `-z` is DOWN toward the table.
- `orientation_rpy = [roll, pitch, yaw]` in RADIANS (XYZ Euler).
  - For `franka_emika_panda`:
    - `[3.14159, 0.0, 1.5708]` points the parallel-jaw gripper straight DOWN (`-Z`).
    - `[3.14159, -0.40, -1.20]` or `[3.14159, -0.40, 0.0]` tilts the gripper ~23° forward along `+X` to slide underneath an overhang shelf from the open `-X` front side.
    - IMPORTANT: Keep Panda `pitch` in `[-0.50, +0.40]` — pitching steeper than `-0.55` drives the wrist/hand into the tabletop!
    - Adjusting `yaw` rotates the parallel gripper jaws around Z to align with box edges, handles, or keyholes.
  - For `wonik_allegro`: `[0.0, 1.5708, 0.0]` points the palm downward/forward over the object.
- When given prior attempt feedback from the Visual Critic, apply the suggested spatial/speed corrections directly."""


IN_HAND_ROTATION_SYSTEM_PROMPT = """You are an expert dexterous manipulation planner controlling a palm-up 4-fingered Wonik Allegro right hand in MuJoCo.

Your job is to output EXACTLY 4 sequential finger-gait keyframes (labeled `pre_grasp`, `grasp`, `close`, `lift` for schema compatibility, representing Gait Step 1, Step 2, Step 3, Step 4) that will be **looped cyclically** to ratchet-spin the multi-colored cube continuously in-hand WITHOUT dropping it off the palm!

PARAMETERIZATION FOR EACH GAIT KEYFRAME:
- `finger_targets`: REQUIRED `{thumb, index, middle, ring}`, each in `[0.0, 1.0]` controlling flexion/curl (`0.0` = extended open, `0.45` = gentle cradle, `0.85` = strong push/curl).
- `orientation_rpy = [index_sweep, middle_sweep, ring_sweep]`: lateral side-to-side abduction sweep in RADIANS (`-0.47` to `+0.47`) for index (`ff`), middle (`mf`), and ring (`rf`) fingers. (This does NOT tilt the palm — the palm is fixed palm-up!)
- `position_offset = [thumb_opp, thumb_sweep, 0.0]`:
  - `thumb_opp` (`tha0`) in `[0.263, 1.396]` rad (higher = thumb rotated deeper in opposition across palm).
  - `thumb_sweep` (`tha1`) in `[-0.105, 1.163]` rad (sweeps thumb laterally to push/spin a corner of the cube!).
- `duration_s`: duration of this gait step in SECONDS (`0.25` to `1.5`). Increase `duration_s` (e.g. `0.8–1.1s`) when the Critic warns the motion is too fast/abrupt or flinging the cube; decrease (`0.35–0.5s`) for brisk unloaded finger resets.
- `gripper_target`: fallback curl `[0.0, 1.0]` (set to `0.5`).

CYCLIC RATCHETING & DRIFT RECOVERY TIPS:
- Break symmetry between power stroke and return stroke so rotation accumulates monotonically instead of rocking back and forth:
  - Power stroke (Steps 1->2): sweep driving fingers while curled (`0.68–0.85`) and thumb pushes opposite corner while stabilizing fingers hold lightly (`0.35–0.50`).
  - Unloaded reset (Steps 3->4): stabilize the cube with opposing fingers (`0.55–0.70`) while lifting/extending driving fingers (`0.15–0.28`) as they sweep back, so they do NOT unwind the cube!
- If the cube has drifted toward the wrist (`-X`) or thumb (`-Y`) in a checkpointed closed-loop chunk, modulate finger curls gently (slightly less index/middle inward pull or stronger thumb backstop) to keep the cube centered near `[0.055, 0.005]`!
- Never open all 4 fingers simultaneously (`< 0.25`) or slam curls above `0.90`, or the cube will slip off the palm.
- Directly apply qualitative & numeric guidance from the Visual Critic (speed, which finger pushed too hard, which finger acted as a brake)."""


def plan_waypoints(
    client: genai.Client,
    planner_model: str,
    task_prompt: str,
    robot_id: RobotId,
    kinematic_summary: str,
    disabled_joints: Sequence[str],
    history: Sequence[dict] | None = None,
    task_type: str = "pick_and_lift",
    chunk_context: str | None = None,
) -> PlannerOutput:
    """Call Gemini 4 Planner to generate or refine 4 waypoints or finger-gait keyframes."""
    system_prompt = (
        IN_HAND_ROTATION_SYSTEM_PROMPT
        if task_type in ("in_hand_rotation", "in_hand_spindle")
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
    if chunk_context:
        user_parts.append("")
        user_parts.append("CURRENT CLOSED-LOOP CHECKPOINT STATE:")
        user_parts.append(chunk_context)

    if history:
        user_parts.append("")
        user_parts.append("RECENT ATTEMPTS & VISUAL CRITIC FEEDBACK (apply directly to fix the failure):")
        # Keep only the most recent 3 attempts so planner context stays fast and focused
        for item in history[-3:]:
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
            "\nPropose an UPDATED 4-step plan that directly fixes the observed physical failure."
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
    max_wps = 4 if task_type in ("in_hand_rotation", "in_hand_spindle") else 7
    converted: list[Waypoint] = []
    for idx, pwp in enumerate(raw.waypoints[:max_wps]):
        phase = pwp.phase if pwp.phase in phases else phases[min(idx, 3)]
        pos = (list(pwp.position_offset) + [0.0, 0.0, 0.0])[:3]
        rpy = (list(pwp.orientation_rpy) + [3.14159, 0.0, 1.5708])[:3]
        ft = pwp.finger_targets.model_dump() if pwp.finger_targets is not None else None
        converted.append(
            Waypoint(
                phase=phase,  # type: ignore[arg-type]
                position_offset=pos,
                orientation_rpy=rpy,
                gripper_target=float(min(1.0, max(0.0, pwp.gripper_target))),
                duration_s=float(min(2.0, max(0.2, pwp.duration_s))),
                finger_targets=ft,
            )
        )

    return PlannerOutput(reasoning=raw.reasoning, waypoints=converted)
