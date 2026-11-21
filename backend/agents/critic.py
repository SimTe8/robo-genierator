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
    """Constructive physical diagnosis and parameter adjustment advice from the Critic."""

    diagnosis: str = Field(
        ...,
        description="Concrete description of what physically happened in the rollout (e.g., gripper stopped 5cm short, fingers grazed left edge, object slipped on lift).",
    )
    suggested_adjustments: str = Field(
        ...,
        description="Specific actionable changes for the Planner (e.g., shift dx +0.02m, lower dz -0.015m, tilt pitch +0.3 rad to compensate for locked joint4, or increase gripper_target).",
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

    if task_type == "in_hand_rotation":
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
            "above the table without dropping it or crashing into the table. "
            f"{give_up_note}"
        )
        verdict_criteria = {
            "ACCEPT_TRAJECTORY": "The robot cleanly grasped the object, lifted it noticeably (>= 8 cm) above the table, and held it stably without dropping or colliding harshly with the table.",
            "REFINE_GRASP": "The attempt failed (missed, grazed, slipped, dropped, stopped short, or hit the table), but adjusting waypoints/orientation/gripper could fix it.",
            "GIVE_UP": "The task appears physically unreachable/impossible under the current joint faults, or attempts are exhausted.",
        }
        failure_criteria = {
            "NONE_SUCCESS": "Clean grasp and stable lift succeeded.",
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
    critique_prompt = (
        f"{context_text}\n\n"
        f"The Decisions API classified this rollout as:\n"
        f"- verdict: {decision} (confidence={confidence:.2f}, probs={probabilities})\n"
        f"- failure_mode: {failure_mode}\n"
        f"- grasp_quality: {grasp_score}/3\n\n"
        "Explain concisely WHY the grasp failed physically (citing exact visual evidence if video is provided, "
        "such as side_cam/top_cam alignment, finger gap, joint reachability, or slip direction) and give "
        "concrete numeric adjustments (dx, dy, dz in meters; roll/pitch/yaw in radians; gripper_target) "
        "so the Planner can succeed on the next attempt."
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

    reason = f"[{failure_mode} | score={grasp_score}/3 | conf={confidence:.2f}] {crit.diagnosis} -> Fix: {crit.suggested_adjustments}"

    return CriticEvaluation(
        decision=decision,
        confidence=confidence,
        probabilities=probabilities,
        failure_mode=failure_mode,
        grasp_score=grasp_score,
        reason=reason,
    )
