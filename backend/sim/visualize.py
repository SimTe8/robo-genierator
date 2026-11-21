"""Simple MuJoCo visualization script to inspect robots and procedural objects.

Usage:
  # Render side + top composite PNG and MP4 previews for both robots (headless EGL):
  python backend/sim/visualize.py --robot all --seed 42

  # Render a specific robot and seed:
  python backend/sim/visualize.py --robot franka_emika_panda --seed 7

  # Open interactive MuJoCo viewer (requires display / WSLg):
  python backend/sim/visualize.py --robot franka_emika_panda --seed 42 --mode viewer
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Set MUJOCO_GL before importing mujoco when running headless render mode
if "--mode" not in sys.argv or "viewer" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio.v3 as iio
import mujoco
import numpy as np

# Ensure backend/ is importable when invoked from repo root
BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sim.scene_builder import RobotId, build_scene, summarize_kinematics


def _label_frame(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (8, 8), (8 + len(text) * 10 + 16, 36), (15, 20, 28), -1)
    cv2.putText(
        out,
        text,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 245, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def render_preview(
    robot_id: RobotId,
    seed: int,
    width: int = 640,
    height: int = 480,
    seconds: float = 2.0,
    fps: int = 30,
) -> tuple[Path, Path]:
    """Simulate settling and render a side+top composite PNG and MP4 into outputs/."""
    bundle = build_scene(robot_id=robot_id, scene_seed=seed)
    m, d = bundle.model, bundle.data

    print("=" * 60)
    print(summarize_kinematics(bundle))
    print("=" * 60)

    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"preview_{robot_id}_seed{seed}.png"
    mp4_path = out_dir / f"preview_{robot_id}_seed{seed}.mp4"

    renderer = mujoco.Renderer(m, height=height, width=width)
    frames: list[np.ndarray] = []

    steps_per_frame = max(1, int(round(1.0 / (fps * m.opt.timestep))))
    total_frames = int(seconds * fps)

    try:
        for i in range(total_frames):
            for _ in range(steps_per_frame):
                mujoco.mj_step(m, d)

            renderer.update_scene(d, camera="side_cam")
            side_img = _label_frame(renderer.render(), f"{robot_id} | side_cam | seed={seed}")

            renderer.update_scene(d, camera="top_cam")
            top_img = _label_frame(renderer.render(), f"top_cam | {bundle.obj_spec.shape}")

            composite = np.concatenate([side_img, top_img], axis=1)
            frames.append(composite)

        # Save final settled frame as PNG
        iio.imwrite(png_path, frames[-1])
        # Save browser-compatible H.264 MP4
        iio.imwrite(
            mp4_path,
            np.stack(frames, axis=0),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
        )
    finally:
        renderer.close()

    print(f"Saved preview image: {png_path}")
    print(f"Saved preview video: {mp4_path}")
    return png_path, mp4_path


def launch_viewer(robot_id: RobotId, seed: int) -> None:
    """Launch MuJoCo's interactive passive viewer."""
    import mujoco.viewer

    bundle = build_scene(robot_id=robot_id, scene_seed=seed)
    m, d = bundle.model, bundle.data
    print(summarize_kinematics(bundle))
    print("Launching interactive MuJoCo viewer... Close the window to exit.")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(m, d)
            viewer.sync()
            dt = m.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize MuJoCo robots and seeded objects.")
    parser.add_argument(
        "--robot",
        choices=["franka_emika_panda", "wonik_allegro", "all"],
        default="all",
        help="Robot model to visualize.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Procedural scene seed.")
    parser.add_argument(
        "--mode",
        choices=["render", "viewer"],
        default="render",
        help="'render' saves headless side+top PNG/MP4; 'viewer' opens interactive GUI.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    robots: list[RobotId] = (
        ["franka_emika_panda", "wonik_allegro"]
        if args.robot == "all"
        else [args.robot]
    )

    if args.mode == "viewer":
        launch_viewer(robots[0], args.seed)
    else:
        for r in robots:
            render_preview(r, args.seed, width=args.width, height=args.height)


if __name__ == "__main__":
    main()

