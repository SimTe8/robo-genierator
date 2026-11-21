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
    exported_paths: list[str] = field(default_factory=list)
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
) -> DomainRandReport:
    """Replay a verified skill across N domain-randomized poses/sizes/masses/frictions headless."""
    t0 = time.time()
    batch_id = uuid.uuid4().hex[:8]
    traj_dir = (output_dir or (REPO_ROOT / "outputs")) / "trajectories" / f"dr_{batch_id}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    passed = 0
    exported_paths: list[str] = []
    failure_summaries: list[dict] = []

    for i in range(num_episodes):
        dr_seed = base_seed * 1000 + i + 1
        bundle = build_scene(
            robot_id=robot_id,
            scene_seed=base_seed,
            disabled_joints=list(disabled_joints),
            task_type=task_type,
            object_shape=object_shape,
            dr_seed=dr_seed,
        )
        res = execute_rollout(
            bundle=bundle,
            waypoints=waypoints,
            render_video=False,
            target_rotation_deg=target_rotation_deg,
        )
        if res.sim_success:
            passed += 1
            out_npz = traj_dir / f"ep_{i + 1:02d}_dr{dr_seed}.npz"
            np.savez_compressed(out_npz, **res.trajectory)
            exported_paths.append(str(out_npz))
        else:
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
        exported_paths=exported_paths,
        failure_summaries=failure_summaries,
    )

