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
import numpy as np
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
    clutter: list[str] = Field(default_factory=list, max_length=5)
    layout_seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


@app.post("/api/scene")
async def scene(q: SceneQuery) -> dict:
    """Default scene for (object, task): preview still + numbers + the template prompt."""
    problem = check_combo(q.robot_id, q.object_id, q.task_id, q.clutter)
    if problem:
        raise HTTPException(422, problem)
    key = hashlib.sha1(json.dumps(q.model_dump(), sort_keys=True).encode()).hexdigest()[:16]
    out = PREVIEW_DIR / f"{key}.png"
    try:
        info = await _in_pool(render_preview, q.robot_id, q.object_id, q.task_id, q.task_params, str(out), q.clutter, q.layout_seed)
    except ValueError as e:  # e.g. not enough free table space for the extra objects
        raise HTTPException(422, str(e)) from e
    return {"scene": info, "preview_url": f"/outputs/previews/{out.name}",
            "prompt": render_prompt(q.object_id, q.task_id, q.task_params, q.clutter)}


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


# ---------------------------------------------------------------------------------------------
# Research Studio: Evaluation Benchmark & Verified Trajectory Dataset Explorer
# ---------------------------------------------------------------------------------------------


def _to_outputs_url(path: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(OUTPUTS_DIR.resolve())
        return f"/outputs/{rel.as_posix()}"
    except ValueError:
        return None


def _normalize_output_url(raw: object) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    norm = raw.replace("\\", "/").strip()
    if norm.startswith("/outputs/"):
        return norm
    if norm.startswith("outputs/"):
        return f"/{norm}"
    idx = norm.find("/outputs/")
    if idx != -1:
        return norm[idx:]
    candidate = OUTPUTS_DIR / norm
    if candidate.exists():
        return _to_outputs_url(candidate)
    return None


def _infer_robot_id(path: Path, qpos_dim: int, ctrl_dim: int) -> str:
    lower = path.as_posix().lower()
    if "panda" in lower or "franka" in lower:
        return "franka_emika_panda"
    if "allegro" in lower:
        return "wonik_allegro"
    if "wuji" in lower:
        return "wuji_hand"
    if ctrl_dim in (7, 8):
        return "franka_emika_panda"
    if ctrl_dim == 16:
        return "wonik_allegro"
    if ctrl_dim in (12, 20) or ctrl_dim >= 18:
        return "wuji_hand"
    if qpos_dim in (7, 9):
        return "franka_emika_panda"
    if qpos_dim == 16:
        return "wonik_allegro"
    if qpos_dim == 20:
        return "wuji_hand"
    return "unknown"


def _read_npz_meta(path: Path, traj_root: Path) -> dict:
    stat = path.stat()
    rel_path = path.relative_to(traj_root).as_posix()
    download_url = _to_outputs_url(path) or f"/outputs/trajectories/{rel_path}"
    steps = 0
    qpos_dim = 0
    ctrl_dim = 0
    keys: list[str] = []
    robot_id: str | None = None

    for allow_pickle in (False, True):
        try:
            with np.load(path, allow_pickle=allow_pickle) as data:
                keys = list(data.files)
                qpos_key = "observation_qpos" if "observation_qpos" in data.files else ("qpos" if "qpos" in data.files else None)
                if qpos_key:
                    qpos_shape = data[qpos_key].shape
                    steps = int(qpos_shape[0]) if len(qpos_shape) >= 1 else 0
                    qpos_dim = int(qpos_shape[1]) if len(qpos_shape) >= 2 else 0

                ctrl_key = "action_ctrl" if "action_ctrl" in data.files else ("ctrl" if "ctrl" in data.files else None)
                if ctrl_key:
                    ctrl_shape = data[ctrl_key].shape
                    ctrl_dim = int(ctrl_shape[1]) if len(ctrl_shape) >= 2 else (int(ctrl_shape[0]) if len(ctrl_shape) == 1 and steps == 0 else 0)

                if "robot_id" in data.files:
                    raw_rid = data["robot_id"]
                    val = raw_rid.item() if raw_rid.ndim == 0 else (raw_rid.flat[0] if raw_rid.size > 0 else "")
                    if isinstance(val, bytes):
                        val = val.decode("utf-8", errors="ignore")
                    val_str = str(val).strip()
                    if val_str:
                        robot_id = val_str
                break
        except Exception:
            continue

    if not robot_id:
        robot_id = _infer_robot_id(path, qpos_dim, ctrl_dim)

    return {
        "filename": path.name,
        "rel_path": rel_path,
        "download_url": download_url,
        "size_kb": round(stat.st_size / 1024.0, 1),
        "mtime": stat.st_mtime,
        "steps": steps,
        "qpos_dim": qpos_dim,
        "ctrl_dim": ctrl_dim,
        "keys": keys,
        "robot_id": robot_id,
    }


@app.get("/api/benchmark")
def get_benchmark() -> dict:
    """Return latest evaluation benchmark report if available without raising 404."""
    bench_path = OUTPUTS_DIR / "benchmark_results.json"
    v1_path = OUTPUTS_DIR / "benchmark_results_v1_initial.json"
    has_v1_archive = v1_path.is_file()
    if not bench_path.is_file():
        return {"available": False, "report": None, "has_v1_archive": has_v1_archive}
    try:
        report = json.loads(bench_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            return {"available": False, "report": None, "has_v1_archive": has_v1_archive}
        return {"available": True, "report": report, "has_v1_archive": has_v1_archive}
    except (OSError, json.JSONDecodeError):
        return {"available": False, "report": None, "has_v1_archive": has_v1_archive}


@app.get("/api/datasets")
def get_datasets() -> dict:
    """Scan outputs/trajectories for verified .npz demonstrations and DR batches."""
    traj_root = OUTPUTS_DIR / "trajectories"
    if not traj_root.is_dir():
        return {"total_npz_files": 0, "dr_batches": [], "recent_trajectories": []}

    npz_files: list[tuple[float, Path]] = []
    for p in traj_root.rglob("*.npz"):
        if p.is_file():
            try:
                npz_files.append((p.stat().st_mtime, p))
            except OSError:
                continue

    npz_files.sort(key=lambda item: item[0], reverse=True)
    total_npz_files = len(npz_files)
    recent_trajectories = [_read_npz_meta(p, traj_root) for _, p in npz_files[:12]]

    dr_dirs_map: dict[Path, Path] = {}
    for d in traj_root.rglob("dr_*"):
        if d.is_dir():
            dr_dirs_map[d.resolve()] = d
    for m in traj_root.rglob("manifest.json"):
        if m.is_file():
            dr_dirs_map[m.parent.resolve()] = m.parent

    # Also check top-level dr_sample_*.mp4 videos in outputs/videos or outputs/trajectories
    global_dr_samples: list[str] = []
    for search_dir in (traj_root, OUTPUTS_DIR / "videos", OUTPUTS_DIR):
        if search_dir.is_dir():
            for mp4 in sorted(search_dir.glob("dr_sample_*.mp4")):
                url = _to_outputs_url(mp4)
                if url and url not in global_dr_samples:
                    global_dr_samples.append(url)

    dr_batches: list[dict] = []
    for dr_dir in dr_dirs_map.values():
        try:
            mtime = dr_dir.stat().st_mtime
        except OSError:
            continue

        manifest_path = dr_dir / "manifest.json"
        manifest: dict = {}
        if manifest_path.is_file():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    manifest = loaded
            except (OSError, json.JSONDecodeError):
                manifest = {}

        ep_files = sorted(p for p in dr_dir.glob("*.npz") if p.is_file())
        npz_count = len(ep_files)

        sample_urls: list[str] = []
        for key in ("sample_video_urls", "dr_video_urls", "video_urls", "sample_videos"):
            raw_list = manifest.get(key)
            if isinstance(raw_list, list):
                for item in raw_list:
                    u = _normalize_output_url(item)
                    if u and u not in sample_urls:
                        sample_urls.append(u)

        for mp4 in sorted(dr_dir.glob("*.mp4")):
            u = _to_outputs_url(mp4)
            if u and u not in sample_urls:
                sample_urls.append(u)

        if not sample_urls and len(dr_dirs_map) == 1 and global_dr_samples:
            sample_urls = list(global_dr_samples[:3])

        if not manifest and npz_count == 0 and not sample_urls:
            continue

        job_meta: dict = {}
        suffix = dr_dir.name.removeprefix("dr_")
        job_json_path = JOBS_DIR / suffix / "job.json"
        if suffix and job_json_path.is_file():
            try:
                loaded_job = json.loads(job_json_path.read_text(encoding="utf-8"))
                if isinstance(loaded_job, dict):
                    job_meta = loaded_job
            except (OSError, json.JSONDecodeError):
                job_meta = {}

        robot_id = manifest.get("robot_id") or (job_meta.get("robot") or {}).get("id")
        if not isinstance(robot_id, str) or not robot_id:
            if ep_files:
                robot_id = _read_npz_meta(ep_files[0], traj_root)["robot_id"]
            else:
                robot_id = _infer_robot_id(dr_dir, 0, 0)

        verified = manifest.get("verified_episodes") or manifest.get("successes") or manifest.get("saved_episodes") or npz_count
        attempted = (
            manifest.get("attempted_episodes")
            or manifest.get("episodes")
            or manifest.get("total_episodes")
            or manifest.get("n_episodes")
            or verified
        )
        try:
            verified_int = int(verified)
        except (TypeError, ValueError):
            verified_int = npz_count
        try:
            attempted_int = max(int(attempted), verified_int)
        except (TypeError, ValueError):
            attempted_int = verified_int

        raw_sr = manifest.get("success_rate")
        if isinstance(raw_sr, (int, float)):
            success_rate = float(raw_sr) if raw_sr <= 1.0 else round(float(raw_sr) / 100.0, 4)
        else:
            success_rate = round(verified_int / attempted_int, 4) if attempted_int > 0 else 0.0

        task_name = (
            manifest.get("task_id")
            or manifest.get("task_type")
            or manifest.get("task")
            or (job_meta.get("task") or {}).get("id")
        )
        if not isinstance(task_name, str) or not task_name:
            task_name = suffix

        obj_id = (
            manifest.get("object_id")
            or manifest.get("object_name")
            or (job_meta.get("object") or {}).get("id")
        )

        dr_batches.append({
            "batch_id": str(manifest.get("batch_id") or dr_dir.name),
            "rel_dir": dr_dir.relative_to(traj_root).as_posix(),
            "robot_id": robot_id,
            "task": task_name,
            "object_id": obj_id,
            "npz_count": npz_count,
            "verified_episodes": verified_int,
            "attempted_episodes": attempted_int,
            "success_rate": success_rate,
            "sample_video_urls": sample_urls[:4],
            "manifest_url": _to_outputs_url(manifest_path) if manifest_path.is_file() else None,
            "mtime": mtime,
        })

    dr_batches.sort(key=lambda b: b["mtime"], reverse=True)

    return {
        "total_npz_files": total_npz_files,
        "dr_batches": dr_batches,
        "recent_trajectories": recent_trajectories,
    }
