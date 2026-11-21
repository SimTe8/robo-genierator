"""Hand-off between the web UI and the Gemini 4 planner.

The UI sends a PlannerRequest (object + task + prompt). This module turns it into a PlannerBrief,
saves it to outputs/jobs/<job_id>/job.json, and starts the planner in a worker process.

──────────────────────────────────────────────────────────────────────────────────────────────
PLUGGING IN THE PLANNER (teammate)
  Create backend/agents/planner.py exposing:

      class GeminiPlanner:
          name = "gemini-4-neon"
          def run(self, brief: PlannerBrief, job_dir: Path, emit: Emit) -> None: ...
      PLANNER = GeminiPlanner()

  Inside run() you own everything: conditions / domain randomisation, calling Gemini, executing.
  Useful building blocks:
      from library.scene import build_task_scene      # scene for (object, task) + your conditions
      from sim.executor import execute_rollout         # waypoints -> MuJoCo rollout + MP4
      from sim.check import snapshot, evaluate         # simulator ground truth for the task
  Report progress with emit(kind, message, iteration=IterationEvent(...), check=CheckResult(...)).
  Videos saved under job_dir are served at /outputs/jobs/<job_id>/<file>.
──────────────────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import importlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from config import OUTPUTS_DIR
from library.catalog import ROBOTS_BY_ID, object_info, task_info
from library.objects import OBJECTS_BY_ID
from library.scene import build_task_scene, check_combo, describe, resolve_params
from library.tasks import TASKS_BY_ID
from schemas import CheckResult, IterationEvent, JobEvent, PlannerBrief, PlannerRequest
from sim.scene_builder import summarize_kinematics

JOBS_DIR = OUTPUTS_DIR / "jobs"

HOW_TO_BUILD = (
    "from library.scene import build_task_scene; "
    "bundle = build_task_scene(object_id, task_id, task_params, robot_id=robot_id, object_xy=(x, y), object_yaw_deg=yaw, "
    "size_scale=s, mass_scale=m, friction=f, disabled_joints=[...]). All conditions are optional; "
    "defaults reproduce default_scene. Score with sim.check.snapshot/evaluate."
)


class Emit(Protocol):
    def __call__(
        self,
        kind: Literal["status", "attempt", "done", "error"],
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
        ev = JobEvent(job_id=self.job_id, seq=self.seq, kind=kind, message=message, iteration=iteration, check=check, data=data)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(ev.model_dump_json() + "\n")


def run_job(job_dir: Path) -> None:
    """Worker-process entry: hand the saved brief to the planner (or report that none is connected)."""
    brief = PlannerBrief.model_validate_json((job_dir / "job.json").read_text())
    emit = EventLog(job_dir, brief.job_id)
    planner = load_planner()
    if planner is None:
        emit("done", "Brief saved. No planner connected yet (backend/agents/planner.py → PLANNER).",
             data={"job_json": f"/outputs/jobs/{brief.job_id}/job.json"})
        return
    emit("status", f"Sent to {planner.name}.")
    planner.run(brief, job_dir, emit)
