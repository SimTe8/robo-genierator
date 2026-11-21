"""Hand-off between the web UI and the Gemini 4 planner-critic pipeline.

The UI sends a PlannerRequest (robot + object + task + prompt + PlannerSettings).
This module turns it into a PlannerBrief, saves outputs/jobs/<job_id>/job.json, and runs the
planner-critic pipeline in a worker process while streaming BOTH live terminal lines and
structured attempt/video events over SSE.

Discovery order:
1. Local plugin hook: `backend/agents/planner.py` exposing `PLANNER` (name + run(brief, job_dir, emit))
2. Local CLI pipeline: `<repo>/backend/agents/run_loop.py`
3. Sibling worktree / configured pipeline: `$PLANNER_REPO_DIR/backend/agents/run_loop.py`
   or `<repo_parent>/gemini4nyc/backend/agents/run_loop.py`
"""

from __future__ import annotations

import ast
import importlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from config import OUTPUTS_DIR, REPO_ROOT
from library.catalog import ROBOTS_BY_ID, object_info, task_info
from library.objects import OBJECTS_BY_ID
from library.scene import build_task_scene, check_combo, describe, resolve_params
from library.tasks import TASKS_BY_ID
from schemas import (
    CheckResult,
    DecisionType,
    IterationEvent,
    JobEvent,
    PlannerBrief,
    PlannerRequest,
)
from sim.scene_builder import summarize_kinematics

JOBS_DIR = OUTPUTS_DIR / "jobs"

HOW_TO_BUILD = (
    "from library.scene import build_task_scene; "
    "bundle = build_task_scene(object_id, task_id, task_params, robot_id=robot_id, object_xy=(x, y), object_yaw_deg=yaw, "
    "size_scale=s, mass_scale=m, friction=f, disabled_joints=[...]). All conditions are optional; "
    "defaults reproduce default_scene. Score with sim.check.snapshot/evaluate."
)

CYLINDER_OBJECTS = {"bottle", "can", "mug", "jar_lid", "cup_beads", "cans_three", "round_peg", "pen", "valve", "dial"}
SPHERE_OBJECTS = {"ball", "egg"}
BOX_OBJECTS = {"cube", "die", "t_block", "l_block", "square_peg", "card"}
IN_HAND_TASKS = {"rotate_in_hand", "reorient_to_target", "flip_over", "stand_upright", "turn_part_to_hand"}


class Emit(Protocol):
    def __call__(
        self,
        kind: Literal["status", "terminal", "attempt", "done", "error"],
        message: str,
        *,
        iteration: IterationEvent | None = None,
        check: CheckResult | None = None,
        data: dict | None = None,
    ) -> None: ...


class Planner(Protocol):
    name: str

    def run(self, brief: PlannerBrief, job_dir: Path, emit: Emit) -> None: ...


class SelectionError(ValueError):
    pass


def load_planner() -> Planner | None:
    try:
        return importlib.import_module("agents.planner").PLANNER
    except (ModuleNotFoundError, AttributeError):
        return None


def find_run_loop_script() -> tuple[Path, Path] | None:
    """Locate backend/agents/run_loop.py in this repo or a configured/sibling worktree.

    Returns (script_path, repo_root) or None.
    """
    candidates: list[Path] = [REPO_ROOT]
    env_dir = os.environ.get("PLANNER_REPO_DIR")
    if env_dir:
        candidates.insert(0, Path(env_dir).expanduser().resolve())
    sibling = REPO_ROOT.parent / "gemini4nyc"
    if sibling.resolve() != REPO_ROOT.resolve():
        candidates.append(sibling)

    for root in candidates:
        script = root / "backend" / "agents" / "run_loop.py"
        if script.is_file():
            return script.resolve(), root.resolve()
    return None


def planner_status() -> dict[str, str | bool | None]:
    """Return status metadata for /api/health."""
    planner = load_planner()
    if planner is not None:
        return {
            "planner": getattr(planner, "name", "connected"),
            "planner_mode": "plugin",
            "planner_path": str(REPO_ROOT / "backend" / "agents" / "planner.py"),
        }
    found = find_run_loop_script()
    if found is not None:
        script, root = found
        label = "Gemini 4 Planner-Critic" if root == REPO_ROOT else "Gemini 4 Planner-Critic (live bridge)"
        return {
            "planner": label,
            "planner_mode": "run_loop_cli",
            "planner_path": str(script),
        }
    return {"planner": None, "planner_mode": None, "planner_path": None}


def make_brief(req: PlannerRequest) -> PlannerBrief:
    problem = check_combo(req.robot_id, req.object_id, req.task_id)
    if problem:
        raise SelectionError(problem)
    obj, task = OBJECTS_BY_ID[req.object_id], TASKS_BY_ID[req.task_id]
    params = resolve_params(task, obj, req.task_params)
    for p in task.params:
        if not p.min <= params[p.key] <= p.max:
            raise SelectionError(f"{p.label} must be between {p.min:g} and {p.max:g} {p.unit}.")

    bundle = build_task_scene(obj.id, task.id, params, robot_id=req.robot_id)
    info = task_info(task, obj, params)
    job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{req.robot_id}_{obj.id}_{task.id}_{uuid.uuid4().hex[:4]}"
    return PlannerBrief(
        job_id=job_id,
        created_at=datetime.now().isoformat(timespec="seconds"),
        robot=ROBOTS_BY_ID[req.robot_id],
        prompt=req.prompt.strip(),
        object=object_info(obj),
        task=info,
        task_params=params,
        settings=req.settings,
        success_criteria=info.criteria,
        default_scene=describe(bundle),
        kinematic_summary=summarize_kinematics(bundle),
        how_to_build_scene=HOW_TO_BUILD,
    )


class EventLog:
    """Appends JobEvents to <job_dir>/events.jsonl; the API streams that file to the UI."""

    def __init__(self, job_dir: Path, job_id: str):
        self.path = job_dir / "events.jsonl"
        self.job_id = job_id
        self.seq = sum(1 for _ in open(self.path, encoding="utf-8")) if self.path.exists() else 0

    def __call__(self, kind, message, *, iteration=None, check=None, data=None) -> None:
        self.seq += 1
        ev = JobEvent(
            job_id=self.job_id,
            seq=self.seq,
            kind=kind,
            message=message,
            iteration=iteration,
            check=check,
            data=data,
        )
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(ev.model_dump_json() + "\n")


def _build_run_loop_command(brief: PlannerBrief, script_path: Path) -> tuple[list[str], list[str]]:
    """Map UI PlannerBrief + PlannerSettings to backend/agents/run_loop.py CLI args."""
    s = brief.settings
    notes: list[str] = []

    # 1. Task type
    if s.task_type_override != "auto":
        task_type = s.task_type_override
    elif brief.task.id in IN_HAND_TASKS or brief.task.skill == "in-hand":
        task_type = "in_hand_rotation"
    else:
        task_type = "pick_and_lift"

    # 2. Robot mapping (run_loop.py currently supports franka_emika_panda and wonik_allegro)
    robot_arg = brief.robot.id
    if task_type in ("in_hand_rotation", "in_hand_spindle"):
        if robot_arg != "wonik_allegro":
            notes.append(
                f"[bridge] Task type '{task_type}' uses the dexterous Allegro hand in run_loop.py (selected: {brief.robot.name})."
            )
            robot_arg = "wonik_allegro"
    elif task_type == "obstacle_grasp":
        if robot_arg != "franka_emika_panda":
            notes.append(
                f"[bridge] Task type 'obstacle_grasp' uses Franka Panda in run_loop.py (selected: {brief.robot.name})."
            )
            robot_arg = "franka_emika_panda"
    elif robot_arg == "wuji_hand":
        notes.append(
            "[bridge] Wuji hand MJCF is active for scene preview; routing planner-critic execution to wonik_allegro in run_loop.py."
        )
        robot_arg = "wonik_allegro"

    # 3. Object shape mapping
    if s.object_shape_override != "auto":
        shape_arg = s.object_shape_override
    elif brief.object.id in CYLINDER_OBJECTS:
        shape_arg = "cylinder"
    elif brief.object.id in SPHERE_OBJECTS:
        shape_arg = "sphere"
    elif brief.object.id in BOX_OBJECTS:
        shape_arg = "box"
    else:
        shape_arg = "auto"

    # 4. Target rotation
    target_rot = float(brief.task_params.get("angle_deg", 60.0))
    target_rot = max(15.0, min(720.0, target_rot))

    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--robot",
        robot_arg,
        "--task-type",
        task_type,
        "--object-shape",
        shape_arg,
        "--seed",
        str(s.scene_seed),
        "--max-attempts",
        str(s.max_attempts),
        "--mode",
        s.feedback_mode,
        "--planner",
        s.planner_model,
        "--critic",
        s.critic_model,
        "--target-rotation",
        f"{target_rot:g}",
        "--task",
        brief.prompt,
    ]
    if s.chunk_step_deg is not None and task_type in ("in_hand_rotation", "in_hand_spindle"):
        cmd.extend(["--chunk-deg", f"{s.chunk_step_deg:g}"])
    if s.dr_episodes > 0:
        cmd.extend(["--dr-episodes", str(s.dr_episodes)])
    if s.disabled_joints:
        cmd.append("--disable")
        cmd.extend(s.disabled_joints)

    return cmd, notes


_ATTEMPT_HDR = re.compile(r"^---\s+(?:Chunk\s+(\d+)\s+\|\s+)?Attempt\s+(\d+)\s+\(([0-9.]+)s\)\s+---$")
_DECISION_LINE = re.compile(
    r"^Decision API:\s+(ACCEPT_TRAJECTORY|REFINE_GRASP|GIVE_UP)\s+\(conf=([^,]+),\s*score=([^/]+)/3,\s*mode=([^)]+)\)$"
)


def _mirror_media(rel_or_abs_path: str, planner_root: Path, job_dir: Path, job_id: str) -> str:
    """Copy a video generated by run_loop.py into job_dir so the UI always serves it at /outputs/jobs/<id>/..."""
    clean = rel_or_abs_path.strip()
    src = Path(clean)
    if not src.is_absolute():
        src = planner_root / clean.lstrip("/")
    if not src.is_file():
        return clean
    dst = job_dir / src.name
    if dst.resolve() != src.resolve():
        try:
            shutil.copy2(src, dst)
        except OSError:
            return clean
    return f"/outputs/jobs/{job_id}/{dst.name}"


def _run_via_cli_bridge(brief: PlannerBrief, job_dir: Path, emit: Emit, script_path: Path, planner_root: Path) -> None:
    """Invoke backend/agents/run_loop.py as a live subprocess, streaming terminal output & parsed attempts."""
    cmd, notes = _build_run_loop_command(brief, script_path)
    display_cmd = " ".join(shlex.quote(c) for c in cmd)

    emit(
        "status",
        f"Launching Gemini 4 Planner-Critic ({brief.settings.planner_model} → {brief.settings.critic_model})",
        data={"command": display_cmd, "planner_root": str(planner_root)},
    )
    emit("terminal", f"$ {display_cmd}")
    for note in notes:
        emit("terminal", note)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Ensure headless EGL on Linux (and keep macOS cgl if set)
    if sys.platform.startswith("linux"):
        env.setdefault("MUJOCO_GL", "egl")

    proc = subprocess.Popen(
        cmd,
        cwd=str(planner_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    (job_dir / "worker.pid").write_text(str(proc.pid), encoding="utf-8")

    cur: dict | None = None
    stitched_video_url: str | None = None
    saved_trajectory: str | None = None
    final_sim_success: bool | None = None
    final_decision: str | None = None

    def flush_attempt() -> None:
        nonlocal cur
        if not cur:
            return
        video_url = _mirror_media(cur.get("video_url", ""), planner_root, job_dir, brief.job_id) if cur.get("video_url") else ""
        sim_ok = bool(cur.get("sim_success", False))
        decision: DecisionType = cur.get("decision", "REFINE_GRASP")
        ev_iter = IterationEvent(
            attempt_id=int(cur["attempt_id"]),
            chunk_id=cur.get("chunk_id"),
            model_used=cur.get("model_used", f"{brief.settings.planner_model} -> {brief.settings.critic_model}"),
            waypoints=[],
            video_url=video_url,
            telemetry_summary=cur.get("telemetry_summary", ""),
            sim_success=sim_ok,
            decision=decision,
            reason=cur.get("reason", ""),
            latency_s=float(cur.get("latency_s", 0.0)),
            decision_confidence=cur.get("confidence"),
            decision_probabilities=cur.get("probabilities"),
            failure_mode=cur.get("failure_mode"),
            grasp_score=cur.get("grasp_score"),
            trajectory_path=cur.get("trajectory_path"),
        )
        chunk_prefix = f"Chunk {ev_iter.chunk_id} · " if ev_iter.chunk_id is not None else ""
        emit(
            "attempt",
            f"{chunk_prefix}Attempt {ev_iter.attempt_id}",
            iteration=ev_iter,
            check=CheckResult(success=sim_ok, checks={"simulator_ground_truth": sim_ok}, metrics={}),
        )
        cur = None

    assert proc.stdout is not None
    term_log = job_dir / "terminal.log"
    with open(term_log, "a", encoding="utf-8") as logf:
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\r\n")
            logf.write(line + "\n")
            logf.flush()
            emit("terminal", line)

            stripped = line.strip()
            if not stripped:
                if cur and cur.get("video_url"):
                    flush_attempt()
                continue

            # Update live status banner when StageSpinner prints stage progress
            if stripped.startswith("… ") or stripped.startswith("✓ "):
                emit("status", stripped)

            m_hdr = _ATTEMPT_HDR.match(stripped)
            if m_hdr:
                flush_attempt()
                cur = {
                    "chunk_id": int(m_hdr.group(1)) if m_hdr.group(1) else None,
                    "attempt_id": int(m_hdr.group(2)),
                    "latency_s": float(m_hdr.group(3)),
                }
                continue

            if cur is not None:
                if stripped.startswith("Models:"):
                    cur["model_used"] = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("Sim Truth:"):
                    rest = stripped.split(":", 1)[1].strip()
                    cur["sim_success"] = "sim_success=True" in rest
                    cur["telemetry_summary"] = rest.split("|", 1)[1].strip() if "|" in rest else rest
                elif stripped.startswith("Decision API:"):
                    md = _DECISION_LINE.match(stripped)
                    if md:
                        cur["decision"] = md.group(1)
                        try:
                            cur["confidence"] = None if md.group(2).strip() == "None" else float(md.group(2))
                        except ValueError:
                            cur["confidence"] = None
                        try:
                            cur["grasp_score"] = None if md.group(3).strip() == "None" else int(md.group(3))
                        except ValueError:
                            cur["grasp_score"] = None
                        fm = md.group(4).strip()
                        cur["failure_mode"] = None if fm == "None" else fm
                elif stripped.startswith("Probs:"):
                    raw_probs = stripped.split(":", 1)[1].strip()
                    try:
                        parsed = ast.literal_eval(raw_probs)
                        if isinstance(parsed, dict):
                            cur["probabilities"] = {str(k): float(v) for k, v in parsed.items()}
                    except (ValueError, SyntaxError):
                        pass
                elif stripped.startswith("Reason:"):
                    cur["reason"] = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("Video:"):
                    cur["video_url"] = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("Exported:"):
                    cur["trajectory_path"] = stripped.split(":", 1)[1].strip()

            # Summary lines
            if stripped.startswith("Final Sim Success:"):
                final_sim_success = "True" in stripped
            elif stripped.startswith("Final Decision:"):
                final_decision = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("Stitched Video:"):
                stitched_video_url = _mirror_media(stripped.split(":", 1)[1].strip(), planner_root, job_dir, brief.job_id)
            elif stripped.startswith("Saved Trajectory:"):
                saved_trajectory = stripped.split(":", 1)[1].strip()

    flush_attempt()
    rc = proc.wait()
    (job_dir / "worker.pid").unlink(missing_ok=True)

    summary_data = {
        "job_json": f"/outputs/jobs/{brief.job_id}/job.json",
        "exit_code": rc,
        "final_sim_success": final_sim_success,
        "final_decision": final_decision,
        "stitched_video_url": stitched_video_url,
        "saved_trajectory": saved_trajectory,
    }
    if rc == 0:
        verdict = (
            "Verified trajectory accepted by Gemini 4 Critic & MuJoCo simulator."
            if final_sim_success
            else f"Closed loop finished ({final_decision or 'no accepted trajectory'})."
        )
        emit("done", verdict, data=summary_data)
    elif rc < 0:
        emit("error", f"Planner process stopped (signal {-rc}).", data=summary_data)
    else:
        emit("error", f"Planner process exited with code {rc}. Check terminal output above.", data=summary_data)


def run_job(job_dir: Path) -> None:
    """Worker-process entry: hand the saved brief to the planner plugin or run_loop.py CLI bridge."""
    brief = PlannerBrief.model_validate_json((job_dir / "job.json").read_text())
    emit = EventLog(job_dir, brief.job_id)

    planner = load_planner()
    if planner is not None:
        emit("status", f"Sent to {planner.name}.")
        planner.run(brief, job_dir, emit)
        return

    found = find_run_loop_script()
    if found is not None:
        script_path, planner_root = found
        _run_via_cli_bridge(brief, job_dir, emit, script_path, planner_root)
        return

    emit(
        "done",
        "Brief saved. No planner connected yet (backend/agents/planner.py → PLANNER or backend/agents/run_loop.py).",
        data={"job_json": f"/outputs/jobs/{brief.job_id}/job.json"},
    )
