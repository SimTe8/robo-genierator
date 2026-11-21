"""Closed-loop Planner -> MuJoCo Executor -> Visual Critic + Decisions API orchestrator."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from agents.critic import evaluate_rollout
from agents.planner import plan_waypoints
from schemas import DecisionType, IterationEvent, RunRequest
from sim.executor import execute_rollout
from sim.scene_builder import REPO_ROOT, build_scene, summarize_kinematics


@dataclass
class LoopSummary:
    run_id: str
    request: RunRequest
    events: list[IterationEvent] = field(default_factory=list)
    final_sim_success: bool = False
    final_decision: DecisionType = "GIVE_UP"
    total_latency_s: float = 0.0
    exported_trajectory_path: str | None = None


def run_closed_loop(
    request: RunRequest,
    on_event: Callable[[IterationEvent], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    output_dir: Path | None = None,
) -> LoopSummary:
    """Run the full closed-loop trajectory generation pipeline for a RunRequest."""
    load_dotenv(REPO_ROOT / ".env")
    client = genai.Client(http_options=types.HttpOptions(api_version="v1beta"))

    def _status(msg: str) -> None:
        if on_status is not None:
            on_status(msg)

    run_id = uuid.uuid4().hex[:8]
    out_root = output_dir or (REPO_ROOT / "outputs")
    videos_dir = out_root / "videos" / run_id
    traj_dir = out_root / "trajectories"
    videos_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Probe scene once to get kinematic summary for the Planner
    init_bundle = build_scene(
        robot_id=request.robot_id,
        scene_seed=request.scene_seed,
        disabled_joints=request.disabled_joints,
        task_type=request.task_type,
    )
    kinematic_summary = summarize_kinematics(init_bundle)

    max_attempts = 1 if request.feedback_mode == "one_shot" else request.max_attempts
    history: list[dict] = []
    events: list[IterationEvent] = []
    t_run_start = time.time()
    exported_traj_path: str | None = None

    for attempt_id in range(1, max_attempts + 1):
        t_iter_start = time.time()
        prefix = f"Attempt {attempt_id}/{max_attempts}"

        # 1. Plan (Gemini 4 Planner)
        _status(f"{prefix} | Planning trajectory ({request.planner_model})")
        plan = plan_waypoints(
            client=client,
            planner_model=request.planner_model,
            task_prompt=request.task_prompt,
            robot_id=request.robot_id,
            kinematic_summary=kinematic_summary,
            disabled_joints=request.disabled_joints,
            history=history if history else None,
            task_type=request.task_type,
        )

        # 2. Execute & Render (MuJoCo EGL) on a fresh scene instance
        _status(f"{prefix} | Simulating & rendering MuJoCo rollout")
        bundle = build_scene(
            robot_id=request.robot_id,
            scene_seed=request.scene_seed,
            disabled_joints=request.disabled_joints,
            task_type=request.task_type,
        )
        video_path = videos_dir / f"attempt_{attempt_id}.mp4"
        render_video = request.feedback_mode in ("video", "one_shot") or True
        rollout = execute_rollout(
            bundle=bundle,
            waypoints=plan.waypoints,
            video_path=video_path,
            render_video=render_video,
            target_rotation_deg=request.target_rotation_deg,
        )

        # 3. Critique & Decide (Decisions API + Visual Critic)
        if request.feedback_mode == "one_shot":
            decision: DecisionType = "ACCEPT_TRAJECTORY" if rollout.sim_success else "GIVE_UP"
            reason = f"[one_shot] Planner reasoning: {plan.reasoning}"
            confidence = None
            probabilities = None
            failure_mode = "NONE_SUCCESS" if rollout.sim_success else "ONE_SHOT_FAIL"
            grasp_score = 3 if rollout.sim_success else 0
            model_used = request.planner_model
        else:
            eval_res = evaluate_rollout(
                client=client,
                critic_model=request.critic_model,
                task_prompt=request.task_prompt,
                robot_id=request.robot_id,
                disabled_joints=request.disabled_joints,
                waypoints=plan.waypoints,
                telemetry_summary=rollout.telemetry_summary,
                attempt_id=attempt_id,
                max_attempts=max_attempts,
                feedback_mode=request.feedback_mode,
                video_path=rollout.video_path,
                task_type=request.task_type,
                target_rotation_deg=request.target_rotation_deg,
                on_status=lambda s: _status(f"{prefix} | {s}"),
            )
            decision = eval_res.decision
            reason = eval_res.reason
            confidence = eval_res.confidence
            probabilities = eval_res.probabilities
            failure_mode = eval_res.failure_mode
            grasp_score = eval_res.grasp_score
            model_used = f"{request.planner_model} -> {request.critic_model}"

        # Export verified expert trajectory ONLY when BOTH critic accepts (or one_shot) AND simulator confirms success
        traj_path_str: str | None = None
        if rollout.sim_success and decision == "ACCEPT_TRAJECTORY":
            traj_file = traj_dir / f"verified_{request.robot_id}_seed{request.scene_seed}_{run_id}_att{attempt_id}.npz"
            np.savez_compressed(traj_file, **rollout.trajectory)
            traj_path_str = str(traj_file)
            exported_traj_path = traj_path_str

        latency_s = round(time.time() - t_iter_start, 2)
        rel_video_url = f"/outputs/videos/{run_id}/attempt_{attempt_id}.mp4"

        event = IterationEvent(
            attempt_id=attempt_id,
            model_used=model_used,
            waypoints=plan.waypoints,
            video_url=rel_video_url,
            telemetry_summary=rollout.telemetry_summary,
            sim_success=rollout.sim_success,
            decision=decision,
            reason=reason,
            latency_s=latency_s,
            decision_confidence=confidence,
            decision_probabilities=probabilities,
            failure_mode=failure_mode,
            grasp_score=grasp_score,
            trajectory_path=traj_path_str,
        )
        events.append(event)
        if on_event is not None:
            on_event(event)

        history.append(
            {
                "attempt_id": attempt_id,
                "waypoints": [wp.model_dump() for wp in plan.waypoints],
                "telemetry_summary": rollout.telemetry_summary,
                "decision": decision,
                "failure_mode": failure_mode,
                "reason": reason,
            }
        )

        # Stop if accepted or gave up
        if decision in ("ACCEPT_TRAJECTORY", "GIVE_UP"):
            break

    return LoopSummary(
        run_id=run_id,
        request=request,
        events=events,
        final_sim_success=events[-1].sim_success if events else False,
        final_decision=events[-1].decision if events else "GIVE_UP",
        total_latency_s=round(time.time() - t_run_start, 2),
        exported_trajectory_path=exported_traj_path,
    )
