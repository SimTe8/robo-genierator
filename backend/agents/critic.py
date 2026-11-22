"""Gemini 4 Visual Critic + Decisions API Gate."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from agents import gemini_client
from schemas import DecisionType, Waypoint


class VisualCritique(BaseModel):
    """Concise, high-signal physical diagnosis and surgical parameter fix from the Critic."""

    diagnosis: str = Field(
        ...,
        description="MAX 2 SHORT SENTENCES: the SINGLE primary physical root cause seen in the video (first check gripper timing: did it stay OPEN until lift? Then check collision, miss direction, jaw yaw, or slip).",
    )
    motion_qualitative_feedback: str = Field(
        ...,
        description="MAX 1 SHORT SENTENCE: what worked well (keep it!) vs. the one timing/speed/alignment/frame issue to fix.",
    )
    suggested_adjustments: str = Field(
        ...,
        description="MAX 2 SHORT SENTENCES: minimal surgical fix changing ONLY the 1-2 parameters that caused the failure (do NOT rewrite working waypoints or drop multi-step waypoints).",
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
    is_custom_prompt: bool = False,
    on_status: Callable[[str], None] | None = None,
    timeout_s: float = 120.0,
    retries: int = 2,
) -> CriticEvaluation:
    """Evaluate a rollout using the Decisions API + constructive diagnosis when refining."""
    use_video = feedback_mode == "video" and video_path is not None and video_path.exists()

    # For custom user prompts, omit the mismatched preset sim_success=False flag so
    # gemini-decision-1 judges purely from the video against the user's custom goal!
    telemetry_line = (
        "Custom user prompt active — judge strictly from what you visually observe in the video."
        if is_custom_prompt
        else f"Contact/Height Summary: {telemetry_summary}"
    )

    context_text = (
        f"Task: {task_prompt}\n"
        f"Robot: {robot_id} | Disabled joints: {list(disabled_joints) or 'none'}\n"
        f"Attempt: {attempt_id} of {max_attempts}\n"
        f"{telemetry_line}\n"
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
        else "Prefer REFINE_GRASP over GIVE_UP if an offset, reference_frame, orientation tilt, or finger adjustment could still succeed."
    )

    if task_type in ("in_hand_rotation", "in_hand_spindle"):
        verdict_instructions = (
            f"Judge whether the hand rotated the multi-colored cube in-hand by at least {target_rotation_deg:.0f} degrees "
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
            f"Judge whether the robot VISUALLY completed the stated user task ('{task_prompt}') in the video "
            "cleanly without dropping, crashing into the table, or hitting any red obstacle/overhang shelf. "
            "Note: if the user wrote a custom prompt (e.g., placing one object onto another), judge by what you SEE "
            "in the video against the user's stated Task prompt even if a generic preset telemetry flag says False. "
            f"{give_up_note}"
        )
        verdict_criteria = {
            "ACCEPT_TRAJECTORY": "The video shows the robot cleanly accomplishing the stated user task goal without harsh collisions.",
            "REFINE_GRASP": "The attempt failed to accomplish the stated user task (missed, dropped, missed destination pad/block, hit obstacle, or collided with table), and adjusting waypoints/reference_frame/orientation/gripper could fix it.",
            "GIVE_UP": "The task appears physically unreachable/impossible under the current constraints, or attempts are exhausted.",
        }
        failure_criteria = {
            "NONE_SUCCESS": "Clean grasp and stable task completion succeeded with no obstacle collision.",
            "OBSTACLE_COLLISION": "Robot arm or gripper collided with a red shelf/overhang/barrier obstacle.",
            "STOPPED_SHORT_UNREACHABLE": "End-effector could not reach the object (e.g., due to a locked joint or too-high z offset).",
            "MISSED_LATERAL": "Gripper/fingers closed beside the object or missed the destination pad/block (x/y or reference_frame error).",
            "TABLE_COLLISION": "Robot collided with or pressed heavily into the tabletop.",
            "SLIPPED_OR_DROPPED": "Robot touched or briefly lifted the object, but it slipped out or dropped.",
            "KNOCKED_OVER": "Gripper knocked the object away before closing.",
        }

    if on_status is not None:
        on_status(f"Evaluating with Decisions API ({decision_model})")

    questions_payload = {
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
                "2 - Partial progress (some lift/rotation/transport but unstable or below target)",
                "3 - Clean, stable task completion meeting the goal",
            ],
        },
    }

    dec_resp: Any = None
    last_dec_err: Exception | None = None
    for retry_idx in range(retries + 1):
        t0_dec = time.time()
        try:
            dec_resp = client.decisions.create(
                model=decision_model,
                input=decision_input,
                questions=questions_payload,
            )
            dec_usage = getattr(dec_resp, "usage_metadata", None) or getattr(dec_resp, "usage", None)
            gemini_client.log_call(
                purpose="decisions_gate",
                model=decision_model,
                latency_s=time.time() - t0_dec,
                ok=True,
                usage=dec_usage,
                extra={
                    "attempt_id": attempt_id,
                    "task_type": task_type,
                    "robot_id": robot_id,
                    "verdict": str(dec_resp.answers["verdict"].choice),
                    "confidence": round(float(dec_resp.answers["verdict"].confidence), 4),
                    "retry": retry_idx,
                },
            )
            break
        except Exception as e:
            last_dec_err = e
            gemini_client.log_call(
                purpose="decisions_gate",
                model=decision_model,
                latency_s=time.time() - t0_dec,
                ok=False,
                extra={
                    "attempt_id": attempt_id,
                    "task_type": task_type,
                    "robot_id": robot_id,
                    "retry": retry_idx,
                },
                error=f"{type(e).__name__}: {str(e)[:300]}",
            )
            if retry_idx < retries:
                time.sleep(1.5 * (retry_idx + 1))

    if dec_resp is None:
        raise RuntimeError(
            f"Decisions API ({decision_model}) failed after {retries + 1} attempts: {last_dec_err}"
        )

    verdict_ans = dec_resp.answers["verdict"]
    failure_ans = dec_resp.answers["failure_mode"]
    score_ans = dec_resp.answers["grasp_quality"]

    decision: DecisionType = verdict_ans.choice  # type: ignore[assignment]
    confidence = float(verdict_ans.confidence)
    probabilities = {k: round(float(v), 4) for k, v in verdict_ans.probabilities.items()}
    failure_mode = str(failure_ans.choice)
    grasp_score = int(score_ans.score)

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
            "IMPORTANT — DEXTEROUS IN-HAND ROTATION CONTROL SCHEMA (palm is FIXED palm-up; do NOT suggest palm dx/dy/dz or palm tilt!):\n"
            "- `finger_targets` (`thumb`, `index`, `middle`, `ring`, `little` in `[0.0, 1.0]`): finger curl/force (`0.0`=open, `0.45`=cradle, `0.85`=strong push).\n"
            "- `orientation_rpy = [index_sweep, middle_sweep, ring_sweep]` in `[-0.47, +0.47]` rad: lateral side-to-side abduction sweep of each finger.\n"
            "- `position_offset = [thumb_opp (0.26..1.40), thumb_sweep (-0.10..1.16), 0.0]`: thumb opposition & lateral sweep angles.\n"
            "- `duration_s` in `[0.25, 1.5]` sec: motion speed per gait step (increase to slow down abrupt/flinging motions; decrease for brisk resets).\n\n"
            "Describe qualitatively WHY the rotation failed or drifted (e.g., which specific finger moved too strongly, acted as a brake during reset, "
            "or pushed the cube off the wrist/thumb edge; whether the motion was too fast/abrupt; whether the cube rocked back and forth instead of ratcheting). "
            "Then give concrete adjustments to `finger_targets`, finger sweeps (`orientation_rpy`), thumb angles (`position_offset[0..1]`), and `duration_s`."
        )
    else:
        schema_guide = (
            "BE ULTRA-CONCISE (<= 70 words total across all fields). Identify the SINGLE primary root cause from the video:\n"
            "1. FIRST check GRIPPER TIMING (read the 'GRIP:OPEN/CLOSED' overlay): did the gripper wait OPEN at the object and only close AFTER lift started? If so, say THAT and fix only gripper_target/close timing!\n"
            "2. Otherwise check the single main spatial error: red overhang collision (state which side is open), lateral miss (dx/dy), depth (dz), jaw yaw alignment across narrow object axis, or multi-object placement miss.\n"
            "3. If placement/insertion missed a pad, bin, lock, hole, or block on a multi-object task, explicitly suggest setting `reference_frame=\"target_pad\"` (or `\"base_block\"` / `\"clutter_1\"` / `\"bin\"` / `\"lock\"` / `\"hole\"`) with local `[0.0, 0.0, z_clearance]` on the transport/place/release waypoints!\n"
            "CRITICAL: Keep everything that already worked! Change ONLY the 1-2 broken parameters, keep Panda pitch in [-0.48, 0.0], and preserve all waypoints needed for multi-step tasks."
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

    call_http_options = gemini_client.ensure_call_http_options(client, timeout_s=timeout_s)
    cfg_kwargs: dict[str, Any] = {
        "response_mime_type": "application/json",
        "response_schema": VisualCritique,
        "temperature": 0.2,
    }
    if call_http_options is not None:
        cfg_kwargs["http_options"] = call_http_options
    crit_config = types.GenerateContentConfig(**cfg_kwargs)

    crit: VisualCritique | None = None
    last_crit_err: Exception | None = None
    for retry_idx in range(retries + 1):
        t0_crit = time.time()
        try:
            crit_resp = client.models.generate_content(
                model=critic_model,
                contents=gen_contents,
                config=crit_config,
            )
            if crit_resp.parsed is not None and isinstance(crit_resp.parsed, VisualCritique):
                crit = crit_resp.parsed
            else:
                crit = VisualCritique.model_validate_json(crit_resp.text or "")
            gemini_client.log_call(
                purpose="visual_critic",
                model=critic_model,
                latency_s=time.time() - t0_crit,
                ok=True,
                usage=getattr(crit_resp, "usage_metadata", None),
                extra={
                    "attempt_id": attempt_id,
                    "task_type": task_type,
                    "robot_id": robot_id,
                    "failure_mode": failure_mode,
                    "retry": retry_idx,
                },
            )
            break
        except Exception as e:
            last_crit_err = e
            gemini_client.log_call(
                purpose="visual_critic",
                model=critic_model,
                latency_s=time.time() - t0_crit,
                ok=False,
                extra={
                    "attempt_id": attempt_id,
                    "task_type": task_type,
                    "robot_id": robot_id,
                    "failure_mode": failure_mode,
                    "retry": retry_idx,
                },
                error=f"{type(e).__name__}: {str(e)[:300]}",
            )
            if retry_idx < retries:
                time.sleep(1.5 * (retry_idx + 1))

    if crit is None:
        raise RuntimeError(
            f"Visual Critic ({critic_model}) failed after {retries + 1} attempts: {last_crit_err}"
        )

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
