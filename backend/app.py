"""FastAPI backend: object/task library, scene previews, hand-off to the Gemini 4 planner (SSE progress + live terminal).

Run from the repo root:
    uvicorn app:app --app-dir backend --port 8000
"""

from __future__ import annotations

import gl_setup  # noqa: F401  (must precede mujoco)

import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import mujoco
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agents import gemini_client
from config import OUTPUTS_DIR
from library.catalog import get_catalog, object_info, render_prompt
from library.generated import GeneratedAsset, refresh_generated, to_object_def
from library.objects import OBJECTS_BY_ID
from library.render import render_preview, render_thumbnail
from library.scene import check_combo
from planner_hub import JOBS_DIR, EventLog, SelectionError, make_brief, planner_status
from schemas import Catalog, PlannerBrief, PlannerRequest
from sidequest.asset_forge import forge, save_draft, sim_check

BACKEND_DIR = Path(__file__).resolve().parent
THUMBS_DIR = OUTPUTS_DIR / "thumbs"
PREVIEW_DIR = OUTPUTS_DIR / "previews"
pool: ProcessPoolExecutor | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global pool
    for d in (JOBS_DIR, THUMBS_DIR, PREVIEW_DIR):
        d.mkdir(parents=True, exist_ok=True)
    pool = ProcessPoolExecutor(max_workers=2)
    yield
    pool.shutdown(cancel_futures=True)


app = FastAPI(title="Robo Genierator", lifespan=lifespan)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


async def _in_pool(fn, *args):
    assert pool is not None
    return await asyncio.get_running_loop().run_in_executor(pool, fn, *args)


@app.get("/api/health")
def health() -> dict:
    return {"mujoco": mujoco.__version__, **planner_status()}


@app.get("/api/catalog", response_model=Catalog)
def catalog() -> Catalog:
    return get_catalog()


@app.get("/api/objects/{object_id}/thumb.png")
async def thumbnail(object_id: str):
    refresh_generated()
    if object_id not in OBJECTS_BY_ID:
        raise HTTPException(404, "unknown object")
    out = THUMBS_DIR / f"{object_id}.png"
    await _in_pool(render_thumbnail, object_id, str(out))
    return FileResponse(out, media_type="image/png")


class SceneQuery(BaseModel):
    robot_id: str = "wonik_allegro"
    object_id: str
    task_id: str
    task_params: dict[str, float] = Field(default_factory=dict)


@app.post("/api/scene")
async def scene(q: SceneQuery) -> dict:
    """Default scene for (object, task): preview still + numbers + the template prompt."""
    problem = check_combo(q.robot_id, q.object_id, q.task_id)
    if problem:
        raise HTTPException(422, problem)
    key = hashlib.sha1(json.dumps(q.model_dump(), sort_keys=True).encode()).hexdigest()[:16]
    out = PREVIEW_DIR / f"{key}.png"
    info = await _in_pool(render_preview, q.robot_id, q.object_id, q.task_id, q.task_params, str(out))
    return {"scene": info, "preview_url": f"/outputs/previews/{out.name}", "prompt": render_prompt(q.object_id, q.task_id, q.task_params)}


@app.post("/api/planner/jobs", response_model=PlannerBrief)
async def create_job(req: PlannerRequest) -> PlannerBrief:
    try:
        brief: PlannerBrief = await _in_pool(make_brief, req)
    except SelectionError as e:
        raise HTTPException(422, str(e)) from e
    job_dir = JOBS_DIR / brief.job_id
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(brief.model_dump_json(indent=2))
    EventLog(job_dir, brief.job_id)("status", "Brief compiled and saved; starting Gemini 4 planner-critic worker…")
    subprocess.Popen(
        [sys.executable, str(BACKEND_DIR / "run_worker.py"), str(job_dir)],
        stdout=open(job_dir / "worker.log", "w"),
        stderr=subprocess.STDOUT,
    )
    return brief


@app.post("/api/planner/jobs/{job_id}/stop")
async def stop_job(job_id: str) -> dict:
    """Terminate an active planner-critic subprocess cleanly."""
    job_dir = (JOBS_DIR / job_id).resolve()
    if job_dir.parent != JOBS_DIR.resolve() or not job_dir.is_dir():
        raise HTTPException(404, "unknown job")
    pid_file = job_dir / "worker.pid"
    if not pid_file.exists():
        return {"stopped": False, "reason": "no running process"}
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        os.kill(pid, signal.SIGTERM)
        return {"stopped": True, "pid": pid}
    except (ValueError, ProcessLookupError):
        pid_file.unlink(missing_ok=True)
        return {"stopped": False, "reason": "process already finished"}


@app.get("/api/planner/jobs/{job_id}/events")
async def job_events(job_id: str):
    job_dir = (JOBS_DIR / job_id).resolve()
    if job_dir.parent != JOBS_DIR.resolve() or not job_dir.is_dir():
        raise HTTPException(404, "unknown job")
    log = job_dir / "events.jsonl"

    async def stream():
        pos, buf, idle = 0, "", 0.0
        while idle < 900:  # stop after 15 min of silence
            if log.exists():
                with open(log, encoding="utf-8") as f:
                    f.seek(pos)
                    buf += f.read()
                    pos = f.tell()
                *lines, buf = buf.split("\n")  # keep a partial last line for the next read
                for line in filter(None, lines):
                    yield {"event": "job", "data": line}
                    if json.loads(line)["kind"] in ("done", "error"):
                        return
                idle = 0.0 if lines else idle
            await asyncio.sleep(0.15)
            idle += 0.15

    return EventSourceResponse(stream())


# ---------------------------------------------------------------------------------------------
# Side quest: generate new objects from a text prompt with Gemini (see sidequest/asset_forge.py)
# ---------------------------------------------------------------------------------------------


@app.get("/api/assets/status")
def assets_status() -> dict:
    from config import load_config

    return {"configured": gemini_client.available(), "model": load_config()["models"]["asset_generator"]}


class ForgeRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=400)


@app.post("/api/assets/generate")
async def generate_asset(req: ForgeRequest) -> dict:
    if not gemini_client.available():
        raise HTTPException(503, "GEMINI_API_KEY is not set. Put it in the repo-root .env file (not .env.example), then restart the backend.")
    assert pool is not None

    def run_sim(*args):  # MuJoCo build + render in the worker process (macOS GL needs a main thread)
        return pool.submit(sim_check, *args).result(timeout=120)

    try:
        draft = await asyncio.get_running_loop().run_in_executor(None, lambda: forge(req.prompt, sim_runner=run_sim))
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:
        raise HTTPException(502, f"Gemini request failed: {type(e).__name__}: {str(e)[:300]}") from e
    preview = object_info(to_object_def(GeneratedAsset.model_validate(draft["asset"]), draft["object_id"]))
    return {**draft, "object": preview.model_dump()}


@app.post("/api/assets/drafts/{draft_id}/save")
def save_asset(draft_id: str) -> dict:
    try:
        object_id = save_draft(draft_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, f"unknown draft: {e}") from e
    refresh_generated()
    return {"object_id": object_id}
