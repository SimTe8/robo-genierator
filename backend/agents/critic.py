"""Gemini 4 Visual Critic + Decisions API Gate."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from schemas import DecisionType, Waypoint


class VisualCritique(BaseModel):
    """Constructive physical diagnosis and actionable parameter/speed advice from the Critic."""

    diagnosis: str = Field(
        ...,
        description="Rich physical description of what went wrong in the video (e.g., movement too abrupt/fast, specific finger pushing too strongly or acting as a brake, direction of cube drift, slip, or collision).",
    )
    motion_qualitative_feedback: str = Field(
        ...,
        description="Qualitative coaching on timing, speed, force balance, and finger/arm coordination (e.g., 'Step 2 is too fast and flings the cube toward the wrist; slow down duration_s and soften index finger curl').",
    )
    suggested_adjustments: str = Field(
        ...,
        description="Concrete parameter adjustments matching the active task schema.",
    )


@dataclass
class CriticEvaluation:
    decision: DecisionType
    confidence: float
    probabilities: dict[str, float]
    failure_mode: str
    grasp_score: int
    reason: str


def _encode_video_b64(video_path: Path) -> str:
    return base64.b64encode(video_path.read_bytes()).decode("ascii")


def evaluate_rollout(
    client: genai.Client,
    critic_model: str,
    task_prompt: str,
    robot_id: str,
    disabled_joints: Sequence[str],
    waypoints: Sequence[Waypoint],
    telemetry_summary: str,
    attempt_id: int,
    max_attempts: int,
    feedback_mode: Literal["text_only", "video"] = "video",
    video_path: Path | None = None,
    decision_model: str = "gemini-decision-1",
    task_type: str = "pick_and_lift",
    target_rotation_deg: float = 60.0,
    on_status: Callable[[str], None] | None = None,
) -> CriticEvaluation:
    """Evaluate a rollout using the Decisions API + constructive diagnosis when refining."""
    use_video = feedback_mode == "video" and video_path is not None and video_path.exists()

    context_text = (
        f"Task: {task_prompt}\n"
        f"Robot: {robot_id} | Disabled joints: {list(disabled_joints) or 'none'}\n"
        f"Attempt: {attempt_id} of {max_attempts}\n"
        f"Contact/Height Summary: {telemetry_summary}\n"
        f"Executed Waypoints: {[wp.model_dump() for wp in waypoints]}"
    )

    # --- STEP 4a: Calibrated Decisions API Gate (1 call, 3 simultaneous questions) ---
    decision_input: list[dict] = [{"type": "text", "text": context_text}]
    if use_video and video_path is not None:
        decision_input.insert(
            0,
            {
                "type": "video",
                "data": _encode_video_b64(video_path),
                "mime_type": "video/mp4",
            },
        )

    give_up_note = (
        "Choose GIVE_UP if this is the final attempt or the fault makes the task physically impossible."
        if attempt_id >= max_attempts
        else "Prefer REFINE_GRASP over GIVE_UP if an offset, orientation tilt, or finger adjustment could still succeed."
    )

    if task_type in ("in_hand_rotation", "in_hand_spindle"):
        verdict_instructions = (
            f"Judge whether the Allegro hand rotated the multi-colored cube in-hand by at least {target_rotation_deg:.0f} degrees "
            "(visible via the red/yellow face markers spinning and the YAW overlay) WITHOUT dropping the cube off the palm. "
            f"{give_up_note}"
        )
        verdict_criteria = {
            "ACCEPT_TRAJECTORY": f"The cube rotated by at least {target_rotation_deg:.0f} degrees in-hand and stayed safely cradled on the palm without dropping.",
            "REFINE_GRASP": f"The cube either dropped off the palm, stayed stuck with too little rotation (< {target_rotation_deg:.0f} degrees), or rocked back and forth without net spin; adjusting finger curls/sweeps could fix it.",
            "GIVE_UP": "The task appears physically impossible under the current finger faults, or attempts are exhausted.",
        }
        failure_criteria = {
            "NONE_SUCCESS": f"Cube rotated >= {target_rotation_deg:.0f} degrees in-hand without dropping.",
            "DROPPED_OFF_PALM": "The cube slipped or was pushed off the palm/fingers and fell.",
            "INSUFFICIENT_ROTATION": f"Cube stayed in hand but rotated less than {target_rotation_deg:.0f} deg due to symmetric pinching, unwinding, or weak torque.",
            "JAMMED_OR_BLOCKED": "One finger blocked the cube from turning or a disabled finger jammed the motion.",
        }
    else:
        verdict_instructions = (
            "Judge whether the robot successfully grasped and stably lifted the target object "
            "above the table without dropping it, crashing into the table, or hitting any red obstacle/overhang shelf. "
            f"{give_up_note}"
        )
        verdict_criteria = {
            "ACCEPT_TRAJECTORY": "The robot cleanly grasped the object, lifted it noticeably (>= 8 cm) above the table, and held it stably without dropping or colliding with the table or any obstacle shelf.",
            "REFINE_GRASP": "The attempt failed (hit an obstacle/shelf, missed, grazed, slipped, dropped, stopped short, or hit the table), but adjusting waypoints/approach direction/tilt/gripper could fix it.",
            "GIVE_UP": "The task appears physically unreachable/impossible under the current faults/obstacles, or attempts are exhausted.",
        }
        failure_criteria = {
            "NONE_SUCCESS": "Clean grasp and stable lift succeeded with no obstacle collision.",
            "OBSTACLE_COLLISION": "Robot arm or gripper collided with a red shelf/overhang/barrier obstacle.",
            "STOPPED_SHORT_UNREACHABLE": "End-effector could not reach the object (e.g., due to a locked joint or too-high z offset).",
            "MISSED_LATERAL": "Gripper/fingers closed beside the object (x/y offset error).",
            "TABLE_COLLISION": "Robot collided with or pressed heavily into the tabletop.",
            "SLIPPED_OR_DROPPED": "Robot touched or briefly lifted the object, but it slipped out or dropped.",
            "KNOCKED_OVER": "Gripper knocked the object away before closing.",
        }

    if on_status is not None:
        on_status(f"Evaluating with Decisions API ({decision_model})")

    dec_resp = client.decisions.create(
        model=decision_model,
        input=decision_input,
        questions={
            "verdict": {
                "type": "choice",
                "instructions": verdict_instructions,
                "criteria": verdict_criteria,
            },
            "failure_mode": {
                "type": "choice",
                "instructions": "Classify the primary physical outcome observed in the rollout.",
                "criteria": failure_criteria,
            },
            "grasp_quality": {
                "type": "score",
                "instructions": "Score the physical execution quality from 0 to 3.",
                "criteria": [
                    "0 - Complete failure (dropped or no contact)",
                    "1 - Poor contact or very little progress toward goal",
                    "2 - Partial progress (some lift/rotation but unstable or below target)",
                    "3 - Clean, stable task completion meeting the goal",
                ],
            },
        },
    )

    verdict_ans = dec_resp.answers["verdict"]
    failure_ans = dec_resp.answers["failure_mode"]
    score_ans = dec_resp.answers["grasp_quality"]

    decision: DecisionType = verdict_ans.choice  # type: ignore[assignment]
    confidence = float(verdict_ans.confidence)
    probabilities = {k: round(float(v), 4) for k, v in verdict_ans.probabilities.items()}
    failure_mode = str(failure_ans.choice)
    grasp_score = int(score_ans.score)

    # If last attempt and model still said REFINE_GRASP, keep the diagnosis but note limit
    if decision == "ACCEPT_TRAJECTORY":
        reason = (
            f"Decisions API accepted trajectory (conf={confidence:.2f}, "
            f"quality={grasp_score}/3, outcome={failure_mode})."
        )
        return CriticEvaluation(
            decision=decision,
            confidence=confidence,
            probabilities=probabilities,
            failure_mode=failure_mode,
            grasp_score=grasp_score,
            reason=reason,
        )

    # --- STEP 4b: Constructive Visual/Text Diagnosis for Planner Refinement ---
    if on_status is not None:
        on_status(f"Generating visual critique ({critic_model})")

    if task_type in ("in_hand_rotation", "in_hand_spindle"):
        schema_guide = (
            "IMPORTANT — ALLEGRO IN-HAND ROTATION CONTROL SCHEMA (palm is FIXED palm-up; do NOT suggest palm dx/dy/dz or palm tilt!):\n"
            "- `finger_targets` (`thumb`, `index`, `middle`, `ring` in `[0.0, 1.0]`): finger curl/force (`0.0`=open, `0.45`=cradle, `0.85`=strong push).\n"
            "- `orientation_rpy = [index_sweep, middle_sweep, ring_sweep]` in `[-0.47, +0.47]` rad: lateral side-to-side abduction sweep of each finger.\n"
            "- `position_offset = [thumb_opp (0.26..1.40), thumb_sweep (-0.10..1.16), 0.0]`: thumb opposition & lateral sweep angles.\n"
            "- `duration_s` in `[0.25, 1.5]` sec: motion speed per gait step (increase to slow down abrupt/flinging motions; decrease for brisk resets).\n\n"
            "Describe qualitatively WHY the rotation failed or drifted (e.g., which specific finger moved too strongly, acted as a brake during reset, "
            "or pushed the cube off the wrist/thumb edge; whether the motion was too fast/abrupt; whether the cube rocked back and forth instead of ratcheting). "
            "Then give concrete adjustments to `finger_targets`, finger sweeps (`orientation_rpy`), thumb angles (`position_offset[0..1]`), and `duration_s`."
        )
    else:
        schema_guide = (
            "Explain concisely WHY the grasp failed physically (citing exact visual evidence from side_cam/top_cam: "
            "if a red overhang shelf/obstacle blocked the arm, explicitly state WHICH direction is blocked—e.g. directly above and behind/+X—and "
            "WHICH side is open—e.g. approach from -X front/side low underneath the shelf, pull out along -X before lifting! "
            "Also note alignment, finger gap, joint reachability, speed/force, or slip direction). "
            "CRITICAL: Never recommend Panda pitch steeper than -0.50 rad (steeper tilts crash the wrist into the table — keep pitch in [-0.48, -0.30] and adjust pre_grasp dx/dz and jaw yaw instead!). "
            "Give concrete numeric adjustments (dx, dy, dz in meters; roll/pitch/yaw in radians; gripper_target; duration_s) "
            "for each of the 4 waypoints (`pre_grasp`, `grasp`, `close`, `lift`) so the Planner avoids the obstacle/fault and succeeds."
        )

    critique_prompt = (
        f"{context_text}\n\n"
        f"The Decisions API classified this rollout as:\n"
        f"- verdict: {decision} (confidence={confidence:.2f}, probs={probabilities})\n"
        f"- failure_mode: {failure_mode}\n"
        f"- grasp_quality: {grasp_score}/3\n\n"
        f"{schema_guide}"
    )

    gen_contents: list = []
    if use_video and video_path is not None:
        gen_contents.append(
            types.Part.from_bytes(
                data=video_path.read_bytes(),
                mime_type="video/mp4",
            )
        )
    gen_contents.append(critique_prompt)

    crit_resp = client.models.generate_content(
        model=critic_model,
        contents=gen_contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VisualCritique,
            temperature=0.2,
        ),
    )

    if crit_resp.parsed is not None and isinstance(crit_resp.parsed, VisualCritique):
        crit = crit_resp.parsed
    else:
        crit = VisualCritique.model_validate_json(crit_resp.text)

    reason = (
        f"[{failure_mode} | score={grasp_score}/3 | conf={confidence:.2f}] "
        f"{crit.diagnosis} | Coaching: {crit.motion_qualitative_feedback} "
        f"-> Fix: {crit.suggested_adjustments}"
    )

    return CriticEvaluation(
        decision=decision,
        confidence=confidence,
        probabilities=probabilities,
        failure_mode=failure_mode,
        grasp_score=grasp_score,
        reason=reason,
    )
