"""FastAPI + SSE backend serving the Gemini 4 Closed-Loop Trajectory Generator."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import threading
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agents.loop import run_closed_loop
from schemas import IterationEvent, RobotId, RunRequest, TaskType, Waypoint
from sim.domain_rand import run_domain_randomization

app = FastAPI(
    title="Zero-Shot Cross-Embodiment Trajectory Generator",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


class DomainRandRequest(BaseModel):
    waypoints: list[Waypoint]
    robot_id: RobotId = "franka_emika_panda"
    task_type: TaskType = "pick_and_lift"
    base_seed: int = 42
    object_shape: str = "auto"
    disabled_joints: list[str] = Field(default_factory=list)
    target_rotation_deg: float = 45.0
    num_episodes: int = Field(default=20, ge=1, le=50)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "mujoco_gl": os.environ.get("MUJOCO_GL", "default")}


@app.get("/api/assets")
def list_assets() -> dict:
    """Return available robots, tasks, objects, fault presets, and Gemini 4 models for the Web UI."""
    return {
        "robots": [
            {
                "id": "franka_emika_panda",
                "name": "Franka Emika Panda (7-DoF Arm + Gripper)",
                "fault_presets": ["joint2", "joint4", "joint6"],
                "license": "Apache-2.0",
            },
            {
                "id": "wonik_allegro",
                "name": "Wonik Allegro Right Hand (16-DoF, 4 Fingers)",
                "fault_presets": ["ffj1", "mfj1", "rfj1", "thj1"],
                "license": "BSD-2-Clause",
            },
        ],
        "tasks": [
            {
                "id": "obstacle_grasp",
                "name": "Obstacle-Guarded Grasp (Overhang Shelf)",
                "default_robot": "franka_emika_panda",
                "default_prompt": "Pick up the object and lift it at least 12 cm while avoiding the red overhang obstacle.",
            },
            {
                "id": "pick_and_lift",
                "name": "Tabletop Pick & Lift (Joint-Fault Recovery)",
                "default_robot": "franka_emika_panda",
                "default_prompt": "Pick up the object and lift it 15 cm above the table.",
            },
            {
                "id": "in_hand_spindle",
                "name": "Dexterous In-Hand Cube Spinner (180°/360° Spindle)",
                "default_robot": "wonik_allegro",
                "default_prompt": "Rotate the multi-colored cube in-hand by at least 180 degrees using coordinated finger gaits.",
            },
            {
                "id": "in_hand_rotation",
                "name": "Free-Cube In-Hand Reorientation (45°)",
                "default_robot": "wonik_allegro",
                "default_prompt": "Rotate the free multi-colored cube in-hand by at least 45 degrees without dropping it.",
            },
        ],
        "objects": ["auto", "box", "cylinder", "sphere"],
        "models": [
            "models/gemini-4-neon",
            "models/gemini-4-hydrogen",
            "models/gemini-4-carbon",
            "models/gemini-4-argon",
        ],
        "decisions_model": "gemini-decision-1",
    }


@app.post("/api/run/stream")
async def stream_closed_loop(req: RunRequest) -> EventSourceResponse:
    """Stream live stage statuses, IterationEvents, and final LoopSummary via SSE."""
    q: queue.Queue[tuple[str, dict] | None] = queue.Queue()

    def _worker() -> None:
        try:
            def _on_status(msg: str) -> None:
                q.put(("status", {"message": msg}))

            def _on_event(ev: IterationEvent) -> None:
                q.put(("iteration", ev.model_dump()))

            summary = run_closed_loop(req, on_event=_on_event, on_status=_on_status)
            q.put(
                (
                    "summary",
                    {
                        "run_id": summary.run_id,
                        "attempts": len(summary.events),
                        "final_sim_success": summary.final_sim_success,
                        "final_decision": summary.final_decision,
                        "total_latency_s": summary.total_latency_s,
                        "exported_trajectory_path": summary.exported_trajectory_path,
                        "stitched_video_path": summary.stitched_video_path,
                    },
                )
            )
        except Exception as exc:
            q.put(("error", {"error": str(exc)}))
        finally:
            q.put(None)

    threading.Thread(target=_worker, daemon=True).start()

    async def _event_gen():
        while True:
            item = await asyncio.to_thread(q.get)
            if item is None:
                break
            event_type, payload = item
            yield {"event": event_type, "data": json.dumps(payload)}

    return EventSourceResponse(_event_gen())


@app.post("/api/domain-randomize")
def domain_randomize(req: DomainRandRequest) -> dict:
    """Multiply a verified skill across N domain-randomized episodes headless (~2-3s)."""
    report = run_domain_randomization(
        waypoints=req.waypoints,
        robot_id=req.robot_id,
        task_type=req.task_type,
        base_seed=req.base_seed,
        object_shape=req.object_shape,
        disabled_joints=req.disabled_joints,
        target_rotation_deg=req.target_rotation_deg,
        num_episodes=req.num_episodes,
    )
    return {
        "batch_id": report.batch_id,
        "total_episodes": report.total_episodes,
        "passed_episodes": report.passed_episodes,
        "transfer_rate": report.transfer_rate,
        "elapsed_s": report.elapsed_s,
        "exported_count": len(report.exported_paths),
        "sample_paths": report.exported_paths[:5],
        "failures": report.failure_summaries[:5],
    }

