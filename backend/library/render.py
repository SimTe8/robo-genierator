"""Offscreen renders for the UI (object thumbnails, scene previews). Runs in a worker process:
macOS OpenGL wants the process's main thread."""

from __future__ import annotations

import gl_setup  # noqa: F401  (must precede mujoco)

from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from config import load_config
from library.objects import OBJECTS_BY_ID
from library.scene import build_task_scene, describe, object_thumbnail_scene


def _write(out: Path, img: np.ndarray) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.png")
    iio.imwrite(tmp, img)
    tmp.replace(out)


def render_thumbnail(object_id: str, out_path: str) -> None:
    out = Path(out_path)
    if out.exists():
        return
    px = int(load_config()["render"]["thumb_size"])
    model, data, look = object_thumbnail_scene(object_id)  # also refreshes generated objects
    obj = OBJECTS_BY_ID[object_id]
    size = max(obj.half_extents) + (obj.articulation.pivot_z / 2 if obj.articulation else 0.0)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = look
    cam.distance, cam.azimuth, cam.elevation = max(0.17, 3.4 * size), 135, -28
    r = mujoco.Renderer(model, px, px)
    try:
        r.update_scene(data, camera=cam)
        _write(out, r.render())
    finally:
        r.close()


def render_preview(robot_id: str, object_id: str, task_id: str, params: dict[str, float], out_path: str) -> dict:
    bundle = build_task_scene(object_id, task_id, params, robot_id=robot_id)
    m, d = bundle.model, bundle.data
    for _ in range(60):  # settle onto the table
        mujoco.mj_step(m, d)
    out = Path(out_path)
    if not out.exists():
        cfg = load_config()["render"]
        ov = cfg["overview"][robot_id]
        r = mujoco.Renderer(m, cfg["preview_height"], cfg["preview_width"])
        try:
            r.update_scene(d, camera="side_cam")
            side = r.render()
            cam = mujoco.MjvCamera()
            cam.lookat[:] = ov["lookat"]
            cam.distance, cam.azimuth, cam.elevation = ov["distance"], ov["azimuth"], ov["elevation"]
            r.update_scene(d, camera=cam)
            _write(out, np.concatenate([side, r.render()], axis=1))
        finally:
            r.close()
    return describe(bundle)
