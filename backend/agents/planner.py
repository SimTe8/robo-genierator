"""Gemini 4 Planner: proposes object-centric & multi-frame manipulation waypoints as structured output."""

from __future__ import annotations

import json
import time
from typing import Any, Sequence

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from agents import gemini_client
from schemas import RobotId, Waypoint


class FingerTargets(BaseModel):
    """Explicit per-finger closure overrides (avoids open dict schemas in Gemini response_schema)."""

    thumb: float = Field(default=1.0, ge=0.0, le=1.0)
    index: float = Field(default=1.0, ge=0.0, le=1.0)
    middle: float = Field(default=1.0, ge=0.0, le=1.0)
    ring: float = Field(default=1.0, ge=0.0, le=1.0)
    little: float = Field(default=1.0, ge=0.0, le=1.0)


class PlannerWaypoint(BaseModel):
    """Gemini-schema-safe waypoint model."""

    phase: str = Field(..., description="One of: pre_grasp, grasp, close, lift")
    position_offset: list[float] = Field(
        ...,
        description="[dx, dy, dz] offset in meters relative to the waypoint's reference_frame origin.",
    )
    reference_frame: str = Field(
        default="target_object",
        description="Coordinate frame anchor for position_offset: 'target_object' for picking/lifting the target object; 'target_pad', 'bin', 'bowl', 'lock', 'hole', 'base_block', 'clutter_1', or 'world' for placing/inserting onto a second object or fixture.",
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
        description="Optional per-finger closure overrides for wonik_allegro (thumb..ring) or wuji_hand (thumb..little).",
    )


class _RawPlannerOutput(BaseModel):
    """Structured output returned by the Gemini 4 Planner."""

    reasoning: str = Field(
        ...,
        description="Brief kinematic reasoning explaining reference_frame anchors, jaw yaw alignment, approach angle, and task strategy.",
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

MIMICGEN-STYLE `reference_frame` ANCHORING (CRITICAL FOR MULTI-OBJECT TASKS):
- Every waypoint has a `reference_frame` field that anchors `position_offset = [dx, dy, dz]` (in meters, `+z` UP, `-z` DOWN):
  1. **Pick / Lift Subtask (`waypoints 1..4`)**: Use `reference_frame: "target_object"` when approaching, grasping, closing, and lifting the target object (`[0.0, 0.0, dz]` relative to `target_object`'s initial center).
  2. **Place / Insert / Stack Subtask (`waypoints 5..7`)**: Whenever placing onto a pad/bin/bowl, inserting into a lock/hole, or stacking onto another block (`base_block`, `clutter_1`, `stack_1`), set `reference_frame` on the transport, lower, and release waypoints (`waypoints 5..7`) directly to THAT destination frame (e.g. `"target_pad"`, `"bin"`, `"bowl"`, `"base_block"`, `"clutter_1"`, `"stack_1"`, `"lock"`, `"hole"`) with local offset `[0.0, 0.0, z_clearance]`!
  - Why this matters: Anchoring pick waypoints to `"target_object"` and place/insert waypoints to `"target_pad"` / `"base_block"` / `"bin"` / `"lock"` / `"hole"` makes the skill generalize **zero-shot** under domain randomization even when BOTH objects/fixtures move independently!
  - Check `available_reference_frames` in the scene geometry summary to verify exact frame names in the scene (or use `"world"` only when targeting raw world `[x, y, z]` coordinates).

TASK-SPECIFIC MOTION PATTERNS:
- **Pick & Lift** (4 waypoints, `reference_frame: "target_object"`):
  1. `pre_grasp`: hover above the object (`[0.0, 0.0, 0.14]`, `gripper_target: 0.0`).
  2. `grasp`: move to object grasp center (`[0.0, 0.0, 0.0]`, `gripper_target: 0.0`).
  3. `close`: hold pose while closing gripper/fingers (`gripper_target: 1.0`).
  4. `lift`: lift vertically (`position_offset[2] >= 0.16`, `gripper_target: 1.0`).
- **Obstacle Grasp (`obstacle_grasp` with overhead shelf)** (5 waypoints, `reference_frame: "target_object"`):
  1. `pre_grasp`: stage in front on the open `-X` side (`[-0.12, 0.0, 0.05]`, `orientation_rpy: [3.14159, -0.42, 0.0]`, `gripper_target: 0.0`).
  2. `grasp`: reach horizontally under the shelf to object center (`[0.0, 0.0, 0.0]`, `orientation_rpy: [3.14159, -0.42, 0.0]`, `gripper_target: 0.0`).
  3. `close`: close jaws firmly (`[0.0, 0.0, 0.0]`, `orientation_rpy: [3.14159, -0.42, 0.0]`, `gripper_target: 1.0`).
  4. `lift`: RETRACT toward `-X` first (`[-0.12, 0.0, 0.05]`, `gripper_target: 1.0`) to clear the overhang roof!
  5. `lift`: lift high in the open `-X` space (`[-0.12, 0.0, 0.18]`, `gripper_target: 1.0`).
- **Pick & Place / Stack / Insert (`place_on_pad`, `place_in_bin`, `stack_on_block`, `stack_tower`, `insert_key`, `insert_matching_hole`)** (6–7 waypoints):
  - Waypoints 1–4 (`reference_frame: "target_object"`): `pre_grasp` -> `grasp` -> `close` -> `lift` (`dz >= 0.14`, `gripper_target: 1.0`).
  - Waypoint 5 (`reference_frame` = destination frame e.g. `"target_pad"`, `"bin"`, `"base_block"`, `"lock"`, `"hole"`): transport horizontally above destination `[0.0, 0.0, 0.14]` (`gripper_target: 1.0`).
  - Waypoint 6 (same destination `reference_frame`): descend gently onto pad/block/hole `[0.0, 0.0, z_place]` (`gripper_target: 1.0`, `duration_s: 0.8`).
  - Waypoint 7 (same destination `reference_frame`): open gripper (`gripper_target: 0.0`) at `[0.0, 0.0, z_place]` to release cleanly (or push key/peg down into `"lock"`/`"hole"` while holding!).
- **Press Button (`press_button`)** (4 waypoints):
  - Close the gripper (`gripper_target: 1.0`) to form a rigid finger probe, hover above the specific target button (`button1`/`button2`/`button3` coordinates from scene geometry!), press STRAIGHT DOWN onto the button, and **HOLD IT PRESSED on the final waypoint** (do NOT lift back up — the simulator checks if the button is depressed at the end!).
- **Open Drawer (`open_drawer`)** (4–5 waypoints):
  - Approach the drawer handle (`gripper_target: 0.0`), move onto handle, close firmly (`gripper_target: 1.0`), then **pull straight back along the drawer's slide axis (`-X`)** at handle height (do NOT lift vertically!).
- **Turn Valve / Dial / Lid (`turn_joint`)** (4–5 waypoints):
  - Hover above knob/valve, descend onto it (`gripper_target: 0.0`), grip (`gripper_target: 1.0`), then **rotate `orientation_rpy[2]` (yaw)** by the target angle while holding position!

ROBOT ORIENTATION & JAW-YAW ALIGNMENT:
- `orientation_rpy = [roll, pitch, yaw]` in RADIANS (XYZ Euler).
- **`franka_emika_panda`** (parallel-jaw gripper, **8.0 cm max jaw opening**):
  - `[3.14159, 0.0, 1.5708]` points the gripper straight DOWN (`-Z`).
  - **JAW-YAW ALIGNMENT**: Always inspect `recommended_topdown_grasp_yaw_rad` and `yaw_rad` in `KINEMATIC & SCENE SUMMARY` when grasping boxes, handles, T-blocks, L-blocks, or keys! Set `orientation_rpy[2]` (`yaw`) to `recommended_topdown_grasp_yaw_rad` (when provided) so the 8.0 cm parallel jaws straddle the object's **narrow axis** cleanly instead of clipping wide corners or knocking the object skewed!
  - `[3.14159, -0.40, -1.20]` or `[3.14159, -0.40, 0.0]` tilts the gripper ~23° forward along `+X` to slide underneath an overhang shelf from the open `-X` front side.
  - IMPORTANT: Keep Panda `pitch` in `[-0.50, +0.40]` — pitching steeper than `-0.55` drives the wrist/hand into the tabletop!
- **`wonik_allegro` (4-finger hand: `thumb`, `index`, `middle`, `ring`) & `wuji_hand` (5-finger hand: `thumb`, `index`, `middle`, `ring`, `little`)**:
  - `[0.0, 1.5708, 0.0]` (mocap tabletop orientation) positions the palm downward/forward over tabletop objects ready for enveloping power grasps or precision pinches.
  - Control all fingers together via `gripper_target` (`0.0` = open, `1.0` = closed) or individually via `finger_targets` (`thumb..ring` for `wonik_allegro`; all 5 fingers `thumb`, `index`, `middle`, `ring`, `little` for `wuji_hand`).
- When given prior attempt feedback from the Visual Critic, apply the suggested spatial/frame/yaw/speed corrections surgically while preserving working stages."""


IN_HAND_ROTATION_SYSTEM_PROMPT = """You are an expert dexterous manipulation planner controlling a palm-up dexterous hand (`wonik_allegro` 4-finger or `wuji_hand` 5-finger) in MuJoCo.

Your job is to output EXACTLY 4 sequential finger-gait keyframes (labeled `pre_grasp`, `grasp`, `close`, `lift` for schema compatibility, representing Gait Step 1, Step 2, Step 3, Step 4) that will be **looped cyclically** to ratchet-spin the object continuously in-hand WITHOUT dropping it off the palm!

PARAMETERIZATION FOR EACH GAIT KEYFRAME:
- `reference_frame`: keep `"target_object"` for all in-hand gait steps.
- `finger_targets`: REQUIRED `{thumb, index, middle, ring, little}`, each in `[0.0, 1.0]` controlling flexion/curl (`0.0` = extended open, `0.45` = gentle cradle, `0.85` = strong push/curl; `wonik_allegro` uses `thumb..ring`, `wuji_hand` uses all 5 `thumb..little`).
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
- Never open all fingers simultaneously (`< 0.25`) or slam curls above `0.90`, or the object will slip off the palm.
- Directly apply qualitative & numeric guidance from the Visual Critic (speed, which finger pushed too hard, which finger acted as a brake)."""


def _parse_planner_response(response: Any) -> _RawPlannerOutput:
    if response.parsed is not None and isinstance(response.parsed, _RawPlannerOutput):
        return response.parsed
    return _RawPlannerOutput.model_validate_json(response.text or "")


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
    thinking_level: str = "medium",
    timeout_s: float = 120.0,
    retries: int = 2,
) -> PlannerOutput:
    """Call Gemini 4 Planner to generate or refine 4-7 manipulation waypoints or 4 finger-gait keyframes."""
    is_in_hand = task_type in ("in_hand_rotation", "in_hand_spindle")
    system_prompt = IN_HAND_ROTATION_SYSTEM_PROMPT if is_in_hand else PLANNER_SYSTEM_PROMPT
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
        if is_in_hand:
            user_parts.append("\nPropose an UPDATED 4-step finger-gait plan that directly fixes the observed failure.")
        else:
            user_parts.append(
                "\nPropose an UPDATED 4-to-7 waypoint plan that surgically fixes the observed physical failure while preserving working stages."
            )
    else:
        if is_in_hand:
            user_parts.append("\nPropose an initial 4-step finger-gait plan (`pre_grasp`, `grasp`, `close`, `lift`).")
        else:
            user_parts.append(
                "\nPropose an initial 4-to-7 waypoint plan (`pre_grasp`, `grasp`, `close`, `lift`, plus destination-frame place/insert steps if needed)."
            )

    contents_str = "\n".join(user_parts)
    thinking_cfg = gemini_client.build_thinking_config(thinking_level, model=planner_model)
    active_thinking: str | None = thinking_level if thinking_cfg is not None else None
    call_http_options = gemini_client.ensure_call_http_options(client, timeout_s=timeout_s)

    def _make_config(tc: Any) -> types.GenerateContentConfig:
        kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "response_schema": _RawPlannerOutput,
            "temperature": 0.25,
        }
        if tc is not None:
            kwargs["thinking_config"] = tc
        if call_http_options is not None:
            kwargs["http_options"] = call_http_options
        return types.GenerateContentConfig(**kwargs)

    raw: _RawPlannerOutput | None = None
    last_err: Exception | None = None

    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            response = client.models.generate_content(
                model=planner_model,
                contents=contents_str,
                config=_make_config(thinking_cfg),
            )
            raw = _parse_planner_response(response)
            gemini_client.log_call(
                purpose="planner",
                model=planner_model,
                latency_s=time.time() - t0,
                ok=True,
                usage=getattr(response, "usage_metadata", None),
                extra={
                    "thinking_level": active_thinking,
                    "task_type": task_type,
                    "robot_id": robot_id,
                    "attempt": attempt + 1,
                    "waypoints": len(raw.waypoints),
                },
            )
            break
        except Exception as e:
            # Graceful 400 fallback if a model rejects thinking_config
            if thinking_cfg is not None and gemini_client.is_thinking_config_rejected(e):
                gemini_client.log_call(
                    purpose="planner",
                    model=planner_model,
                    latency_s=time.time() - t0,
                    ok=False,
                    extra={
                        "thinking_level": active_thinking,
                        "fallback_without_thinking": True,
                        "task_type": task_type,
                        "robot_id": robot_id,
                    },
                    error=f"{type(e).__name__}: {str(e)[:300]}",
                )
                thinking_cfg = None
                active_thinking = None
                t0_fb = time.time()
                try:
                    response = client.models.generate_content(
                        model=planner_model,
                        contents=contents_str,
                        config=_make_config(None),
                    )
                    raw = _parse_planner_response(response)
                    gemini_client.log_call(
                        purpose="planner",
                        model=planner_model,
                        latency_s=time.time() - t0_fb,
                        ok=True,
                        usage=getattr(response, "usage_metadata", None),
                        extra={
                            "thinking_level": None,
                            "fallback_without_thinking": True,
                            "task_type": task_type,
                            "robot_id": robot_id,
                            "attempt": attempt + 1,
                            "waypoints": len(raw.waypoints),
                        },
                    )
                    break
                except Exception as fb_err:
                    e = fb_err
                    t0 = t0_fb

            last_err = e
            gemini_client.log_call(
                purpose="planner",
                model=planner_model,
                latency_s=time.time() - t0,
                ok=False,
                extra={
                    "thinking_level": active_thinking,
                    "task_type": task_type,
                    "robot_id": robot_id,
                    "attempt": attempt + 1,
                },
                error=f"{type(e).__name__}: {str(e)[:300]}",
            )
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

    if raw is None:
        raise RuntimeError(
            f"Planner Gemini call ({planner_model}) failed after {retries + 1} attempts: {last_err}"
        )

    phases = ["pre_grasp", "grasp", "close", "lift"]
    max_wps = 4 if is_in_hand else 7
    default_rpy = [0.0, 1.5708, 0.0] if robot_id in ("wonik_allegro", "wuji_hand") else [3.14159, 0.0, 1.5708]
    converted: list[Waypoint] = []
    for idx, pwp in enumerate(raw.waypoints[:max_wps]):
        phase = pwp.phase if pwp.phase in phases else phases[min(idx, 3)]
        pos = (list(pwp.position_offset) + [0.0, 0.0, 0.0])[:3]
        rpy = (list(pwp.orientation_rpy) + default_rpy)[:3]
        ft = pwp.finger_targets.model_dump() if pwp.finger_targets is not None else None
        ref_frame = (pwp.reference_frame or "target_object").strip() or "target_object"
        converted.append(
            Waypoint(
                phase=phase,  # type: ignore[arg-type]
                position_offset=pos,
                reference_frame=ref_frame,
                orientation_rpy=rpy,
                gripper_target=float(min(1.0, max(0.0, pwp.gripper_target))),
                duration_s=float(min(2.0, max(0.2, pwp.duration_s))),
                finger_targets=ft,
            )
        )

    return PlannerOutput(reasoning=raw.reasoning, waypoints=converted)
