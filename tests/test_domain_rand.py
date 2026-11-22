"""Integration test for the Domain Randomization trajectory multiplier."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sim.domain_rand import run_domain_randomization
from sim.executor import default_panda_waypoints


def test_run_domain_randomization_exports_npz_and_sample_video(tmp_path: Path) -> None:
    report = run_domain_randomization(
        waypoints=default_panda_waypoints(),
        robot_id="franka_emika_panda",
        task_type="pick_and_lift",
        base_seed=42,
        object_shape="cylinder",
        num_episodes=3,
        render_samples=1,
        output_dir=tmp_path,
    )

    assert report.total_episodes == 3
    assert report.passed_episodes >= 2
    assert len(report.exported_paths) == report.passed_episodes
    assert report.transfer_rate >= 2 / 3 - 1e-4

    # Verify .npz trajectories exist and load cleanly via np.load
    for npz_str in report.exported_paths:
        npz_path = Path(npz_str)
        assert npz_path.is_file()
        with np.load(npz_path) as data:
            assert "observation_qpos" in data
            assert "privileged_obj_pos" in data
            assert "action_ctrl" in data
            assert data["observation_qpos"].shape[0] > 5

    # Verify sample MP4 video was rendered
    assert len(report.sample_video_paths) == 1
    sample_mp4 = tmp_path / "trajectories" / f"dr_{report.batch_id}" / "dr_sample_1.mp4"
    assert sample_mp4.is_file()
    assert sample_mp4.stat().st_size > 1000
