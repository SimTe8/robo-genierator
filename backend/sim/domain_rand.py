"""Fast Domain Randomization Multiplier: turn 1 verified skill into N verified trajectories in ~2s."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from schemas import RobotId, TaskType, Waypoint
from sim.executor import execute_rollout
from sim.scene_builder import REPO_ROOT, build_scene


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
    is_custom_prompt: bool = False,
) -> DomainRandReport:
    """Replay a verified skill across N domain-randomized poses/sizes/masses/frictions and render sample videos."""
    t0 = time.time()
    batch_id = uuid.uuid4().hex[:8]
    out_root = output_dir or (REPO_ROOT / "outputs")
    traj_dir = out_root / "trajectories" / f"dr_{batch_id}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    passed = 0
    exported_paths: list[str] = []
    sample_video_paths: list[str] = []
    failure_summaries: list[dict] = []

    for i in range(num_episodes):
        dr_seed = base_seed * 1000 + i + 1
        if library_object_id and library_task_id:
            from library.scene import build_task_scene

            rng = np.random.default_rng(dr_seed)
            bundle = build_task_scene(
                library_object_id,
                library_task_id,
                library_task_params or {},
                robot_id=robot_id,
                size_scale=float(rng.uniform(0.90, 1.10)),
                mass_scale=float(rng.uniform(0.80, 1.25)),
                disabled_joints=list(disabled_joints) or None,
                clutter=list(library_clutter) or None,
                layout_seed=dr_seed,
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

        want_video = len(sample_video_paths) < render_samples
        vid_path = (traj_dir / f"dr_sample_{len(sample_video_paths) + 1}.mp4") if want_video else None
        res = execute_rollout(
            bundle=bundle,
            waypoints=waypoints,
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
            np.savez_compressed(out_npz, **res.trajectory)
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
                    "obj_pos": [round(v, 3) for v in bundle.obj_spec.pos],
                    "obj_mass": round(bundle.obj_spec.mass, 3),
                    "telemetry": res.telemetry_summary,
                }
            )

    elapsed = round(time.time() - t0, 2)
    return DomainRandReport(
        batch_id=batch_id,
        total_episodes=num_episodes,
        passed_episodes=passed,
        transfer_rate=round(passed / max(1, num_episodes), 4),
        elapsed_s=elapsed,
        output_folder=f"outputs/trajectories/dr_{batch_id}",
        exported_paths=exported_paths,
        sample_video_paths=sample_video_paths,
        failure_summaries=failure_summaries,
    )

