"""Closed-loop Planner -> MuJoCo Executor -> Visual Critic + Decisions API orchestrator.

Supports both:
1. Single-horizon iterative refinement (pick_and_lift, multi-frame manipulation, short in_hand_rotation)
2. Checkpointed Closed-Loop / Receding-Horizon Generation (e.g. 360-deg in-hand rotation
   broken into verified 90-deg chunks with MuJoCo qpos/qvel/ctrl state checkpointing and
   continuous MP4 + .npz stitching).
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import imageio.v3 as iio
import numpy as np
from google import genai

from agents import gemini_client
from agents.critic import evaluate_rollout
from agents.planner import plan_waypoints
from schemas import DecisionType, IterationEvent, RunRequest
from sim.executor import SimState, execute_rollout
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
    stitched_video_path: str | None = None


def _merge_chunk_trajectories(chunk_trajectories: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate matching time-series arrays across chunks while copying scalar/metadata keys from the last chunk."""
    if not chunk_trajectories:
        return {}
    chunk_t_lens = [int(np.asarray(ct["observation_qpos"]).shape[0]) for ct in chunk_trajectories]
    merged: dict[str, np.ndarray] = {}
    for key, last_val in chunk_trajectories[-1].items():
        last_arr = np.asarray(last_val)
        if (
            last_arr.ndim >= 1
            and all(key in ct for ct in chunk_trajectories)
            and all(
                np.asarray(ct[key]).ndim == last_arr.ndim
                and np.asarray(ct[key]).shape[0] == t_len
                and np.asarray(ct[key]).shape[1:] == last_arr.shape[1:]
                for ct, t_len in zip(chunk_trajectories, chunk_t_lens)
            )
        ):
            merged[key] = np.concatenate([np.asarray(ct[key]) for ct in chunk_trajectories], axis=0)
        else:
            merged[key] = last_arr
    return merged


def _run_checkpointed_rotation(
    client: genai.Client,
    request: RunRequest,
    kinematic_summary: str,
    run_id: str,
    videos_dir: Path,
    traj_dir: Path,
    on_event: Callable[[IterationEvent], None] | None,
    status_fn: Callable[[str], None],
) -> LoopSummary:
    """Receding-horizon closed-loop generator with MuJoCo state checkpointing."""
    t_run_start = time.time()
    chunk_deg = float(request.chunk_step_deg or 90.0)
    total_goal = float(request.target_rotation_deg)
    max_chunks = max(1, int(math.ceil(total_goal / chunk_deg)) + 2)
    attempts_per_chunk = 1 if request.feedback_mode == "one_shot" else request.max_attempts

    checkpoint: SimState | None = None
    events: list[IterationEvent] = []
    stitched_frames: list[np.ndarray] = []
    chunk_trajectories: list[dict[str, np.ndarray]] = []
    global_attempt = 0

    for chunk_id in range(1, max_chunks + 1):
        cum_so_far = checkpoint.cum_yaw_deg if checkpoint is not None else 0.0
        remaining = total_goal - abs(cum_so_far)
        if remaining <= 0.0:
            break

        this_chunk_goal = min(chunk_deg, remaining)
        chunk_history: list[dict] = []
        chunk_passed = False

        if checkpoint is not None:
            drift_dx = (float(checkpoint.obj_pos[0]) - 0.015) * 100.0
            drift_dy = (float(checkpoint.obj_pos[1]) - 0.0) * 100.0
            spin_dir = "positive (+Z / CCW)" if cum_so_far >= 0 else "negative (-Z / CW)"
            chunk_ctx = (
                f"Chunk {chunk_id}: Continuing from SAVED MuJoCo checkpoint!\n"
                f"- Cumulative rotation achieved so far: {cum_so_far:+.1f} deg / {total_goal:.0f} deg.\n"
                f"- This chunk's milestone: add at least {this_chunk_goal:.0f} deg MORE in the SAME {spin_dir} direction without dropping.\n"
                f"- Current cube center: [{checkpoint.obj_pos[0]:.3f}, {checkpoint.obj_pos[1]:.3f}, {checkpoint.obj_pos[2]:.3f}] "
                f"(palm drift: {drift_dx:+.1f} cm X, {drift_dy:+.1f} cm Y from ideal palm center [0.015, 0.000]).\n"
                "- If drift > 1.0 cm, gently recenter the cube with balanced finger curl while advancing the next rotation stroke!"
            )
        else:
            chunk_ctx = (
                f"Chunk 1: Starting from initial settled cube on palm (0.0 deg / {total_goal:.0f} deg).\n"
                f"- This chunk's milestone: rotate by at least {this_chunk_goal:.0f} deg in-hand while keeping the cube centered on the palm."
            )

        for local_att in range(1, attempts_per_chunk + 1):
            global_attempt += 1
            t_iter_start = time.time()
            prefix = f"Chunk {chunk_id} ({abs(cum_so_far):.0f}/{total_goal:.0f}°) | Att {local_att}/{attempts_per_chunk}"

            status_fn(f"{prefix} | Planning gait ({request.planner_model}, thinking={request.thinking_level})")
            plan = plan_waypoints(
                client=client,
                planner_model=request.planner_model,
                task_prompt=(
                    f"Rotate the cube by an additional {this_chunk_goal:.0f} degrees "
                    f"(cumulative goal: {total_goal:.0f} deg) without dropping it."
                ),
                robot_id=request.robot_id,
                kinematic_summary=kinematic_summary,
                disabled_joints=request.disabled_joints,
                history=chunk_history if chunk_history else None,
                task_type=request.task_type,
                chunk_context=chunk_ctx,
                thinking_level=request.thinking_level,
            )

            status_fn(f"{prefix} | Simulating & rendering MuJoCo chunk")
            bundle = build_scene(
                robot_id=request.robot_id,
                scene_seed=request.scene_seed,
                disabled_joints=request.disabled_joints,
                task_type=request.task_type,
            )
            video_path = videos_dir / f"chunk{chunk_id}_att{local_att}.mp4"
            cycles = max(2, min(4, int(round(this_chunk_goal / 35.0))))
            rollout = execute_rollout(
                bundle=bundle,
                waypoints=plan.waypoints,
                video_path=video_path,
                render_video=True,
                target_rotation_deg=this_chunk_goal,
                initial_state=checkpoint,
                gait_cycles=cycles,
                total_goal_deg=total_goal,
            )

            if request.feedback_mode == "one_shot":
                decision: DecisionType = "ACCEPT_TRAJECTORY" if rollout.sim_success else "GIVE_UP"
                reason = f"[one_shot | chunk {chunk_id}] {plan.reasoning}"
                confidence = None
                probabilities = None
                failure_mode = "NONE_SUCCESS" if rollout.sim_success else "ONE_SHOT_FAIL"
                grasp_score = 3 if rollout.sim_success else 0
                model_used = request.planner_model
            else:
                eval_res = evaluate_rollout(
                    client=client,
                    critic_model=request.critic_model,
                    task_prompt=(
                        f"Chunk {chunk_id}: add >= {this_chunk_goal:.0f} deg rotation without dropping "
                        f"(starting from {cum_so_far:+.1f} deg)."
                    ),
                    robot_id=request.robot_id,
                    disabled_joints=request.disabled_joints,
                    waypoints=plan.waypoints,
                    telemetry_summary=rollout.telemetry_summary,
                    attempt_id=local_att,
                    max_attempts=attempts_per_chunk,
                    feedback_mode=request.feedback_mode,
                    video_path=rollout.video_path,
                    task_type=request.task_type,
                    target_rotation_deg=this_chunk_goal,
                    on_status=lambda s: status_fn(f"{prefix} | {s}"),
                )
                decision = eval_res.decision
                reason = eval_res.reason
                confidence = eval_res.confidence
                probabilities = eval_res.probabilities
                failure_mode = eval_res.failure_mode
                grasp_score = eval_res.grasp_score
                model_used = f"{request.planner_model} -> {request.critic_model}"

            latency_s = round(time.time() - t_iter_start, 2)
            rel_video_url = f"/outputs/videos/{run_id}/chunk{chunk_id}_att{local_att}.mp4"

            event = IterationEvent(
                attempt_id=global_attempt,
                chunk_id=chunk_id,
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
            )
            events.append(event)
            if on_event is not None:
                on_event(event)

            chunk_history.append(
                {
                    "attempt_id": local_att,
                    "waypoints": [wp.model_dump() for wp in plan.waypoints],
                    "telemetry_summary": rollout.telemetry_summary,
                    "decision": decision,
                    "failure_mode": failure_mode,
                    "reason": reason,
                }
            )

            # Commit checkpoint ONLY when simulator confirms chunk milestone succeeded without dropping!
            if rollout.sim_success and rollout.end_state is not None:
                checkpoint = rollout.end_state
                stitched_frames.extend(rollout.frames)
                chunk_trajectories.append(rollout.trajectory)
                chunk_passed = True
                break

            if decision == "GIVE_UP" and local_att >= attempts_per_chunk:
                break

        if not chunk_passed:
            break

    final_cum = abs(checkpoint.cum_yaw_deg) if checkpoint is not None else 0.0
    final_success = bool(final_cum >= total_goal)

    stitched_video_str: str | None = None
    if stitched_frames:
        stitched_mp4 = videos_dir / "stitched_closed_loop.mp4"
        iio.imwrite(
            stitched_mp4,
            np.stack(stitched_frames, axis=0),
            fps=15,
            codec="libx264",
            pixelformat="yuv420p",
            output_params=["-profile:v", "baseline", "-level", "3.0", "-movflags", "+faststart"],
        )
        stitched_video_str = str(stitched_mp4)

    exported_traj_path: str | None = None
    if final_success and chunk_trajectories:
        traj_file = traj_dir / f"verified_closedloop_{request.robot_id}_seed{request.scene_seed}_{run_id}.npz"
        merged_traj = _merge_chunk_trajectories(chunk_trajectories)
        np.savez_compressed(traj_file, **merged_traj)
        exported_traj_path = str(traj_file)

    return LoopSummary(
        run_id=run_id,
        request=request,
        events=events,
        final_sim_success=final_success,
        final_decision="ACCEPT_TRAJECTORY" if final_success else "GIVE_UP",
        total_latency_s=round(time.time() - t_run_start, 2),
        exported_trajectory_path=exported_traj_path,
        stitched_video_path=stitched_video_str,
    )


def run_closed_loop(
    request: RunRequest,
    on_event: Callable[[IterationEvent], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    output_dir: Path | None = None,
) -> LoopSummary:
    """Run the full closed-loop trajectory generation pipeline for a RunRequest."""
    client = gemini_client.get_client(api_version="v1beta", timeout_s=120.0)

    def _status(msg: str) -> None:
        if on_status is not None:
            on_status(msg)

    run_id = uuid.uuid4().hex[:8]
    out_root = output_dir or (REPO_ROOT / "outputs")
    videos_dir = out_root / "videos" / run_id
    traj_dir = out_root / "trajectories"
    videos_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    def _make_bundle():
        if request.library_object_id and request.library_task_id:
            from library.scene import build_task_scene

            return build_task_scene(
                request.library_object_id,
                request.library_task_id,
                request.library_task_params,
                robot_id=request.robot_id,
                disabled_joints=request.disabled_joints,
                clutter=request.library_clutter or None,
                layout_seed=request.library_layout_seed,
            )
        return build_scene(
            robot_id=request.robot_id,
            scene_seed=request.scene_seed,
            disabled_joints=request.disabled_joints,
            task_type=request.task_type,
            object_shape=request.object_shape,
        )

    # Probe scene once to get kinematic summary (+ library scene/fixture geometry if applicable)
    init_bundle = _make_bundle()
    kinematic_summary = summarize_kinematics(init_bundle)
    if request.library_object_id and request.library_task_id:
        import json as _json
        import mujoco as _mj
        from library.scene import describe as describe_lib_scene

        desc = describe_lib_scene(init_bundle)
        # Include exact world positions of articulated child parts AND added clutter objects
        part_positions = {}
        clutter_objects = []
        target_bid = _mj.mj_name2id(init_bundle.model, _mj.mjtObj.mjOBJ_BODY, "target_object")
        target_pos = init_bundle.data.xpos[target_bid] if target_bid >= 0 else np.zeros(3)
        clutter_ids = list(request.library_clutter or [])

        for bid in range(init_bundle.model.nbody):
            bname = _mj.mj_id2name(init_bundle.model, _mj.mjtObj.mjOBJ_BODY, bid) or ""
            if bname.startswith("clutter_"):
                pos = init_bundle.data.xpos[bid]
                rel = pos - target_pos
                # Look up corresponding clutter object metadata + primary geom RGBA color
                idx = -1
                try:
                    idx = int(bname.split("_")[1]) - 1
                except (IndexError, ValueError):
                    pass
                oid = clutter_ids[idx] if 0 <= idx < len(clutter_ids) else bname
                rgba = None
                for gid in range(init_bundle.model.ngeom):
                    if init_bundle.model.geom_bodyid[gid] == bid:
                        rgba = [round(float(v), 2) for v in init_bundle.model.geom_rgba[gid]]
                        break
                clutter_objects.append(
                    {
                        "body": bname,
                        "object_id": oid,
                        "rgba": rgba,
                        "world_pos": [round(float(v), 4) for v in pos],
                        "offset_from_target_object": [round(float(v), 4) for v in rel],
                    }
                )
            elif any(k in bname for k in ("button", "handle", "drawer", "lock", "lid", "valve", "dial", "pad", "bin", "bowl", "hole")):
                part_positions[bname] = [round(float(v), 4) for v in init_bundle.data.xpos[bid]]

        available_frames: list[str] = ["target_object", "world"]
        fixture_name = desc.get("fixture")
        if fixture_name and str(fixture_name) not in available_frames:
            available_frames.append(str(fixture_name))
        if "hole_center" in desc and "hole" not in available_frames:
            available_frames.append("hole")
        for extra_obj in desc.get("other_objects") or []:
            body_name = extra_obj.get("body") if isinstance(extra_obj, dict) else None
            if body_name and body_name not in available_frames:
                available_frames.append(str(body_name))
        for c_obj in clutter_objects:
            if c_obj["body"] not in available_frames:
                available_frames.append(c_obj["body"])

        desc["available_reference_frames"] = available_frames
        if part_positions:
            desc["part_world_positions"] = part_positions
        if clutter_objects:
            desc["added_clutter_objects"] = clutter_objects
        if len(available_frames) > 2:
            desc["multi_object_note"] = (
                f"MIMICGEN MULTI-FRAME ANCHORING: Available `reference_frame` values in this scene are {available_frames}. "
                "Use `reference_frame='target_object'` when picking/lifting `target_object`, and set `reference_frame` "
                "directly to the destination fixture or extra/clutter body (e.g. 'target_pad', 'bin', 'bowl', 'lock', "
                "'hole', 'base_block', 'stack_1', 'clutter_1') with local `position_offset=[0.0, 0.0, z_clearance]` "
                "when transporting, placing, stacking, or inserting so the skill transfers zero-shot under domain randomization!"
            )

        kinematic_summary += (
            "\n\nLIBRARY TASK SCENE GEOMETRY (object, fixtures, target pads, articulated parts):\n"
            + _json.dumps(desc, indent=2)
        )

    if request.task_type in ("in_hand_rotation", "in_hand_spindle") and request.chunk_step_deg is not None:
        return _run_checkpointed_rotation(
            client=client,
            request=request,
            kinematic_summary=kinematic_summary,
            run_id=run_id,
            videos_dir=videos_dir,
            traj_dir=traj_dir,
            on_event=on_event,
            status_fn=_status,
        )

    max_attempts = 1 if request.feedback_mode == "one_shot" else request.max_attempts
    history: list[dict] = []
    events: list[IterationEvent] = []
    t_run_start = time.time()
    exported_traj_path: str | None = None

    for attempt_id in range(1, max_attempts + 1):
        t_iter_start = time.time()
        prefix = f"Attempt {attempt_id}/{max_attempts}"

        # 1. Plan (Gemini 4 Planner)
        _status(f"{prefix} | Planning trajectory ({request.planner_model}, thinking={request.thinking_level})")
        plan = plan_waypoints(
            client=client,
            planner_model=request.planner_model,
            task_prompt=request.task_prompt,
            robot_id=request.robot_id,
            kinematic_summary=kinematic_summary,
            disabled_joints=request.disabled_joints,
            history=history if history else None,
            task_type=request.task_type,
            thinking_level=request.thinking_level,
        )

        # 2. Execute & Render (MuJoCo EGL) on a fresh scene instance
        _status(f"{prefix} | Simulating & rendering MuJoCo rollout")
        bundle = _make_bundle()
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
                is_custom_prompt=request.is_custom_prompt,
                on_status=lambda s: _status(f"{prefix} | {s}"),
            )
            decision = eval_res.decision
            reason = eval_res.reason
            confidence = eval_res.confidence
            probabilities = eval_res.probabilities
            failure_mode = eval_res.failure_mode
            grasp_score = eval_res.grasp_score
            model_used = f"{request.planner_model} -> {request.critic_model}"

        # For custom user prompts (where preset sim/check rules don't apply), trust the visual Decisions API acceptance
        effective_success = bool(
            rollout.sim_success
            or (request.is_custom_prompt and decision == "ACCEPT_TRAJECTORY")
        )

        # Export verified expert trajectory when accepted and verified
        traj_path_str: str | None = None
        if effective_success and decision == "ACCEPT_TRAJECTORY":
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
            sim_success=effective_success,
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
