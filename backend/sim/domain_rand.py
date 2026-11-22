"""Fast Domain Randomization Multiplier: turn 1 verified skill into N verified trajectories in ~2s."""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from schemas import RobotId, TaskType, Waypoint
from sim.executor import execute_rollout
from sim.scene_builder import REPO_ROOT, SceneBundle, build_scene


def _extract_body_yaws(bundle: SceneBundle) -> dict[str, float]:
    """Return world yaw (radians) for every named body in the scene."""
    m, d = bundle.model, bundle.data
    yaws: dict[str, float] = {}
    for bid in range(m.nbody):
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname:
            mat = d.xmat[bid].reshape(3, 3)
            yaws[bname] = math.atan2(float(mat[1, 0]), float(mat[0, 0]))
    return yaws


def _transform_waypoints_se2(
    waypoints: Sequence[Waypoint],
    base_yaws: dict[str, float],
    ep_yaws: dict[str, float],
) -> list[Waypoint]:
    """Apply SE(2) MimicGen yaw-equivariant subtask transforms per waypoint reference_frame."""
    out: list[Waypoint] = []
    for wp in waypoints:
        ref = getattr(wp, "reference_frame", "target_object") or "target_object"
        if ref in base_yaws and ref in ep_yaws and ref != "world":
            d_yaw = ep_yaws[ref] - base_yaws[ref]
        else:
            d_yaw = 0.0

        if abs(d_yaw) < 1e-5:
            out.append(wp)
            continue

        c, s = math.cos(d_yaw), math.sin(d_yaw)
        dx, dy, dz = wp.position_offset
        new_pos = [dx * c - dy * s, dx * s + dy * c, dz]
        rpy = list(wp.orientation_rpy)
        # Extrinsic 'xyz' Euler: Rz(yaw + d_yaw) @ Ry(pitch) @ Rx(roll) = Rz(d_yaw) @ R_orig
        new_rpy = [rpy[0], rpy[1], rpy[2] + d_yaw]
        out.append(wp.model_copy(update={"position_offset": new_pos, "orientation_rpy": new_rpy}))
    return out


@dataclass
class DomainRandReport:
    batch_id: str
    total_episodes: int
    passed_episodes: int
    transfer_rate: float
    elapsed_s: float
    output_folder: str = ""
    exported_paths: list[str] = field(default_factory=list)
    sample_video_paths: list[str] = field(default_factory=list)
    failure_summaries: list[dict] = field(default_factory=list)


def run_domain_randomization(
    waypoints: Sequence[Waypoint],
    robot_id: RobotId,
    task_type: TaskType,
    base_seed: int = 42,
    object_shape: str = "auto",
    disabled_joints: Sequence[str] = (),
    target_rotation_deg: float = 45.0,
    num_episodes: int = 20,
    output_dir: Path | None = None,
    render_samples: int = 3,
    library_object_id: str | None = None,
    library_task_id: str | None = None,
    library_task_params: dict[str, float] | None = None,
    library_clutter: Sequence[str] = (),
    base_layout_seed: int | None = None,
    is_custom_prompt: bool = False,
) -> DomainRandReport:
    """Replay a verified skill across N domain-randomized poses/sizes/masses/frictions and render sample videos."""
    t0 = time.time()
    batch_id = uuid.uuid4().hex[:8]
    out_root = output_dir or (REPO_ROOT / "outputs")
    traj_dir = out_root / "trajectories" / f"dr_{batch_id}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Record nominal body yaws on the demonstration scene for SE(2) yaw-equivariant subtask transfer
    use_se2 = task_type not in ("obstacle_grasp", "in_hand_rotation", "in_hand_spindle")
    base_yaws: dict[str, float] = {}
    if use_se2:
        if library_object_id and library_task_id:
            from library.scene import build_task_scene

            nom_bundle = build_task_scene(
                library_object_id,
                library_task_id,
                library_task_params or {},
                robot_id=robot_id,
                disabled_joints=list(disabled_joints) or None,
                clutter=list(library_clutter) or None,
                layout_seed=base_layout_seed,
            )
        else:
            nom_bundle = build_scene(
                robot_id=robot_id,
                scene_seed=base_seed,
                disabled_joints=list(disabled_joints),
                task_type=task_type,
                object_shape=object_shape,
            )
        base_yaws = _extract_body_yaws(nom_bundle)

    passed = 0
    exported_paths: list[str] = []
    sample_video_paths: list[str] = []
    failure_summaries: list[dict] = []

    for i in range(num_episodes):
        dr_seed = base_seed * 1000 + i + 1
        if library_object_id and library_task_id:
            from library.scene import build_task_scene

            rng = np.random.default_rng(dr_seed)
            size_s = float(rng.uniform(0.90, 1.10))
            mass_s = float(rng.uniform(0.80, 1.25))
            fx_jitter = (float(rng.uniform(-0.035, 0.035)), float(rng.uniform(-0.035, 0.035)))
            bundle = build_task_scene(
                library_object_id,
                library_task_id,
                library_task_params or {},
                robot_id=robot_id,
                size_scale=size_s,
                mass_scale=mass_s,
                disabled_joints=list(disabled_joints) or None,
                clutter=list(library_clutter) or None,
                layout_seed=dr_seed,
                fixture_xy_jitter=fx_jitter,
            )
        else:
            bundle = build_scene(
                robot_id=robot_id,
                scene_seed=base_seed,
                disabled_joints=list(disabled_joints),
                task_type=task_type,
                object_shape=object_shape,
                dr_seed=dr_seed,
            )

        ep_waypoints = (
            _transform_waypoints_se2(waypoints, base_yaws, _extract_body_yaws(bundle))
            if use_se2
            else waypoints
        )
        want_video = len(sample_video_paths) < render_samples
        vid_path = (traj_dir / f"dr_sample_{len(sample_video_paths) + 1}.mp4") if want_video else None
        res = execute_rollout(
            bundle=bundle,
            waypoints=ep_waypoints,
            video_path=vid_path,
            width=320,
            height=240,
            fps=12,
            render_video=want_video,
            target_rotation_deg=target_rotation_deg,
        )
        ep_passed = res.sim_success or (is_custom_prompt and res.metrics.get("obj_contact_steps", 0) > 50)
        if ep_passed:
            passed += 1
            out_npz = traj_dir / f"ep_{i + 1:02d}_dr{dr_seed}.npz"
            np.savez_compressed(
                out_npz,
                **res.trajectory,
                episode_seed=np.int32(dr_seed),
                task_id=np.asarray(library_task_id or task_type),
            )
            exported_paths.append(str(out_npz))
            if want_video and vid_path is not None and vid_path.is_file():
                sample_video_paths.append(f"/outputs/trajectories/dr_{batch_id}/{vid_path.name}")
        else:
            if vid_path is not None and vid_path.is_file():
                vid_path.unlink(missing_ok=True)
            failure_summaries.append(
                {
                    "episode": i + 1,
                    "dr_seed": dr_seed,
                    "obj_pos": [round(float(v), 3) for v in bundle.obj_spec.pos],
                    "obj_mass": round(float(bundle.obj_spec.mass), 3),
                    "telemetry": res.telemetry_summary,
                }
            )

    elapsed = round(time.time() - t0, 2)
    transfer_rate = round(passed / max(1, num_episodes), 4)
    serialized_waypoints = [
        wp.model_dump() if hasattr(wp, "model_dump") else dict(wp.__dict__)
        for wp in waypoints
    ]
    manifest = {
        "batch_id": batch_id,
        "robot_id": robot_id,
        "task_type": task_type,
        "library_object_id": library_object_id,
        "library_task_id": library_task_id,
        "total_episodes": num_episodes,
        "passed_episodes": passed,
        "transfer_rate": transfer_rate,
        "elapsed_s": elapsed,
        "waypoints": serialized_waypoints,
        "exported_files": [Path(p).name for p in exported_paths],
    }
    (traj_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return DomainRandReport(
        batch_id=batch_id,
        total_episodes=num_episodes,
        passed_episodes=passed,
        transfer_rate=transfer_rate,
        elapsed_s=elapsed,
        output_folder=f"outputs/trajectories/dr_{batch_id}",
        exported_paths=exported_paths,
        sample_video_paths=sample_video_paths,
        failure_summaries=failure_summaries,
    )

