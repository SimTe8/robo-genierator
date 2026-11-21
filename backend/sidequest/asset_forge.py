"""Side quest: text prompt -> new manipulable object for the library, designed by Gemini.

    draft = forge("a TV remote with three buttons")   # Gemini -> JSON spec -> validated -> sim-tested -> draft
    save_draft(draft["draft_id"])                      # -> backend/library/generated/<id>.json (joins the library)

Gemini only returns data (parts, buttons, physical properties, task ids). Our code validates it,
re-centres it, builds it in MuJoCo with the hand, checks it rests stably, and renders a thumbnail.
CLI: python backend/sidequest/forge_cli.py "a padlock with a shackle" [--save]
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from config import OUTPUTS_DIR, load_config
from library.generated import GENERATED_DIR, GENERIC_TASKS, GeneratedAsset, inline_schema, normalise, slug, to_object_def
from library.objects import OBJECTS_BY_ID

DRAFTS_DIR = OUTPUTS_DIR / "drafts"

PROMPT = """You design a single rigid object for a robot-manipulation simulator (MuJoCo), built ONLY from primitives.
The robots are dexterous hands (Allegro 4-finger, Wuji 5-finger) and a Franka Panda gripper (8 cm max opening).

Object request: "{request}"

Rules
- Units are metres. Object frame: z is up, the object rests on the table in this pose. Typical size 2-20 cm.
- Primitive sizes follow MuJoCo: box=[half_x, half_y, half_z]; cylinder and capsule=[radius, half_length] along `axis`;
  sphere=[radius]; ellipsoid=[rx, ry, rz]. `pos` is the primitive centre. Use 1-12 primitives that touch or overlap.
- Give important graspable features a `name` (e.g. "handle", "lid", "blade"); the first named part is the one
  tasks refer to. Add small spring `buttons` only if the real object has push buttons (name the main one "button").
- Choose realistic mass and colours, and fill `physics` honestly (material, fragility, deformability, transparency,
  contents, can_contain_liquid, sealed) — these are physically grounded concepts the planner reasons about.
- yaw_symmetry_deg: 0 if it looks the same at every rotation about z (round), 90/180 if it repeats, else 360.
- tasks: pick 2-5 ids from this list that make physical sense for the object:
{tasks}
Return only JSON that matches the schema.{error}"""


def _prompt(request: str, error: str | None) -> str:
    tasks = "\n".join(f"  - {k}: {v}" for k, v in GENERIC_TASKS.items())
    err = f"\n\nYour previous answer was rejected: {error}\nFix that and answer again." if error else ""
    return PROMPT.format(request=request.strip(), tasks=tasks, error=err)


def _unique_id(name: str) -> str:
    base = f"gen_{slug(name)}"
    taken = set(OBJECTS_BY_ID) | {p.stem for p in GENERATED_DIR.glob("*.json")}
    oid, n = base, 2
    while oid in taken:
        oid, n = f"{base}_{n}", n + 1
    return oid


def sim_check(asset_json: str, obj_id: str, thumb_path: str) -> dict[str, Any]:
    """Runs in a worker process: register the draft, build it with the Allegro hand, simulate 1 s, render a thumbnail."""
    import gl_setup  # noqa: F401

    import mujoco
    import numpy as np

    from library.objects import OBJECTS, OBJECTS_BY_ID as BY_ID
    from library.render import render_thumbnail
    from library.scene import build_task_scene

    obj = to_object_def(GeneratedAsset.model_validate_json(asset_json), obj_id)
    BY_ID[obj_id] = obj
    if obj not in OBJECTS:
        OBJECTS.append(obj)
    task = obj.tasks[0].task_id
    bundle = build_task_scene(obj_id, task, robot_id="wonik_allegro")
    m, d = bundle.model, bundle.data
    bid = m.body("target_object").id
    p0 = d.xpos[bid].copy()
    for _ in range(500):
        mujoco.mj_step(m, d)
    p1 = d.xpos[bid].copy()
    top = bundle.extras["task"]["table_top_z"]
    problems = []
    if not np.all(np.isfinite(d.qpos)):
        problems.append("simulation became unstable (NaN)")
    elif p1[2] < top - 0.005:
        problems.append("object fell through the table")
    elif np.linalg.norm(p1[:2] - p0[:2]) > 0.05:
        problems.append(f"object does not rest stably (slid/rolled {np.linalg.norm(p1[:2] - p0[:2]) * 100:.1f} cm)")
    Path(thumb_path).unlink(missing_ok=True)
    render_thumbnail(obj_id, thumb_path)
    return {"problems": problems, "settled_drift_cm": round(float(np.linalg.norm(p1 - p0)) * 100, 2), "task_used": task}


def forge(request: str, sim_runner=None) -> dict[str, Any]:
    """Ask Gemini for an object; validate (one retry with the error appended); sim-test. Returns a draft dict."""
    from agents.gemini_client import generate_json  # the only SDK user

    models = load_config()["models"]
    model = models["asset_generator"]
    schema = inline_schema()
    error: str | None = None
    last: dict[str, Any] | None = None
    for attempt in range(2):
        raw = generate_json(model, _prompt(request, error), schema, purpose="asset_forge")
        last = raw
        try:
            asset, warnings = normalise(GeneratedAsset.model_validate(raw))
        except Exception as e:  # schema or geometry problem -> retry once with the error
            error = str(e)[:500]
            continue
        draft_id = uuid.uuid4().hex[:10]
        obj_id = _unique_id(asset.name)
        ddir = DRAFTS_DIR / draft_id
        ddir.mkdir(parents=True, exist_ok=True)
        runner = sim_runner or (lambda *a: sim_check(*a))
        sim = runner(asset.model_dump_json(), obj_id, str(ddir / "thumb.png"))
        if sim["problems"] and attempt == 0:
            error = "; ".join(sim["problems"]) + ". Make the base flatter/wider so it rests stably."
            shutil.rmtree(ddir, ignore_errors=True)
            continue
        draft = {
            "draft_id": draft_id, "object_id": obj_id, "prompt": request, "model": model,
            "created_at": datetime.now().isoformat(timespec="seconds"), "asset": asset.model_dump(),
            "warnings": warnings + sim["problems"], "sim": sim, "attempts": attempt + 1,
            "thumbnail_url": f"/outputs/drafts/{draft_id}/thumb.png",
        }
        (ddir / "draft.json").write_text(json.dumps(draft, indent=2))
        return draft
    raise ValueError(f"Gemini's object was rejected twice: {error}. Last answer: {json.dumps(last)[:400]}")


def save_draft(draft_id: str) -> str:
    """Promote a draft into the library. Returns the new object id."""
    if not draft_id.isalnum():
        raise ValueError("bad draft id")
    draft = json.loads((DRAFTS_DIR / draft_id / "draft.json").read_text())
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    keep = {k: draft[k] for k in ("object_id", "prompt", "model", "created_at", "asset", "warnings")}
    (GENERATED_DIR / f"{draft['object_id']}.json").write_text(json.dumps(keep, indent=2))
    return draft["object_id"]
