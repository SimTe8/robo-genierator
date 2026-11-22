"""Physics integration tests for IK rollouts, EGL MP4 rendering, reference frames, spindle checkpointing, and Wuji hand."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from library.scene import build_task_scene
from schemas import Waypoint
from sim.executor import (
    SimState,
    _execute_in_hand_rotation,
    default_panda_waypoints,
    execute_rollout,
)
from sim.scene_builder import build_scene


def test_execute_rollout_panda_default_waypoints() -> None:
    bundle = build_scene(robot_id="franka_emika_panda", scene_seed=42)
    res = execute_rollout(bundle, default_panda_waypoints(), render_video=False)

    assert res.sim_success is True
    assert float(res.metrics["final_lift_cm"]) >= 8.0
    assert res.trajectory["observation_qpos"].shape[0] > 5
    assert res.trajectory["privileged_obj_pos"].shape[0] == res.trajectory["observation_qpos"].shape[0]
    assert res.trajectory["action_ctrl"].shape[0] == res.trajectory["observation_qpos"].shape[0]


def test_execute_rollout_renders_mp4_video(tmp_path: Path) -> None:
    bundle = build_scene(robot_id="franka_emika_panda", scene_seed=42)
    out_mp4 = tmp_path / "test.mp4"
    short_waypoints = [
        Waypoint(phase="pre_grasp", position_offset=[0.0, 0.0, 0.14], gripper_target=0.0, duration_s=0.4),
    ]
    res = execute_rollout(
        bundle,
        short_waypoints,
        video_path=out_mp4,
        width=320,
        height=240,
        fps=8,
        render_video=True,
    )

    assert res.video_path == out_mp4
    assert out_mp4.is_file()
    assert out_mp4.stat().st_size > 1000


def test_waypoint_reference_frames_execute_cleanly() -> None:
    bundle = build_task_scene("mug", "place_on_pad", robot_id="franka_emika_panda")
    waypoints = [
        Waypoint(
            phase="pre_grasp",
            position_offset=[0.0, 0.0, 0.12],
            reference_frame="target_object",
            gripper_target=0.0,
            duration_s=0.4,
        ),
        Waypoint(
            phase="lift",
            position_offset=[0.0, 0.0, 0.12],
            reference_frame="target_pad",
            gripper_target=0.0,
            duration_s=0.4,
        ),
    ]
    res = execute_rollout(bundle, waypoints, render_video=False)

    assert res.trajectory["observation_qpos"].shape[0] > 5
    assert np.all(np.isfinite(res.trajectory["observation_qpos"]))
    assert "sim_success=" in res.telemetry_summary

    # Verify the final waypoint anchored to 'target_pad' moved ee_site above zone_center
    ee_sid = mujoco.mj_name2id(bundle.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    ee_xy = bundle.data.site_xpos[ee_sid][:2]
    pad_xy = np.asarray(bundle.extras["task"]["zone_center"][:2], dtype=np.float64)
    assert float(np.linalg.norm(ee_xy - pad_xy)) < 0.03


def test_in_hand_spindle_rotation_and_checkpoint_resume() -> None:
    gait = [
        Waypoint(
            phase="pre_grasp",
            position_offset=[1.25, 0.20, 0.0],
            orientation_rpy=[-0.25, -0.25, -0.25],
            gripper_target=0.65,
            duration_s=0.25,
            finger_targets={"index": 0.65, "middle": 0.65, "ring": 0.65, "thumb": 0.70},
        ),
        Waypoint(
            phase="grasp",
            position_offset=[1.25, 0.85, 0.0],
            orientation_rpy=[0.30, 0.30, 0.30],
            gripper_target=0.80,
            duration_s=0.25,
            finger_targets={"index": 0.80, "middle": 0.80, "ring": 0.80, "thumb": 0.80},
        ),
        Waypoint(
            phase="close",
            position_offset=[1.15, 0.85, 0.0],
            orientation_rpy=[0.30, 0.30, 0.30],
            gripper_target=0.20,
            duration_s=0.25,
            finger_targets={"index": 0.15, "middle": 0.15, "ring": 0.15, "thumb": 0.55},
        ),
        Waypoint(
            phase="lift",
            position_offset=[1.15, 0.20, 0.0],
            orientation_rpy=[-0.25, -0.25, -0.25],
            gripper_target=0.20,
            duration_s=0.25,
            finger_targets={"index": 0.15, "middle": 0.15, "ring": 0.15, "thumb": 0.25},
        ),
    ]

    bundle1 = build_scene(robot_id="wonik_allegro", scene_seed=42, task_type="in_hand_spindle")
    res1 = _execute_in_hand_rotation(
        bundle=bundle1,
        waypoints=gait,
        render_video=False,
        target_rotation_deg=15.0,
        gait_cycles=2,
    )

    assert res1.end_state is not None
    assert isinstance(res1.end_state, SimState)
    assert res1.end_state.qpos.shape == (bundle1.model.nq,)
    assert res1.end_state.qvel.shape == (bundle1.model.nv,)
    assert res1.end_state.ctrl.shape == (bundle1.model.nu,)
    assert res1.metrics["dropped"] is False

    # Resume from saved SimState checkpoint on a fresh scene bundle
    bundle2 = build_scene(robot_id="wonik_allegro", scene_seed=42, task_type="in_hand_spindle")
    res2 = _execute_in_hand_rotation(
        bundle=bundle2,
        waypoints=gait,
        render_video=False,
        target_rotation_deg=15.0,
        initial_state=res1.end_state,
        gait_cycles=2,
    )

    assert res2.end_state is not None
    expected_cum = res1.end_state.cum_yaw_deg + float(res2.metrics["chunk_yaw_deg"])
    assert float(res2.metrics["cum_yaw_deg"]) == pytest.approx(expected_cum, abs=0.25)


def test_wuji_hand_five_finger_targets_rollout() -> None:
    bundle = build_task_scene("cube", "pick_lift", robot_id="wuji_hand")
    ctrl_before = bundle.data.ctrl.copy()
    qpos_before = bundle.data.qpos.copy()

    five_finger_targets = {
        "thumb": 0.9,
        "index": 0.9,
        "middle": 0.9,
        "ring": 0.9,
        "little": 0.9,
    }
    waypoints = [
        Waypoint(
            phase="pre_grasp",
            position_offset=[-0.04, 0.0, 0.08],
            orientation_rpy=[1.57, 1.57, -1.57],
            gripper_target=0.0,
            duration_s=0.4,
        ),
        Waypoint(
            phase="close",
            position_offset=[-0.02, 0.0, 0.04],
            orientation_rpy=[1.57, 1.57, -1.57],
            gripper_target=0.9,
            finger_targets=five_finger_targets,
            duration_s=0.4,
        ),
    ]
    res = execute_rollout(bundle, waypoints, render_video=False)

    assert res.trajectory["action_ctrl"].shape == (res.trajectory["observation_qpos"].shape[0], 20)
    assert np.linalg.norm(bundle.data.ctrl - ctrl_before) > 0.5
    assert np.linalg.norm(bundle.data.qpos - qpos_before) > 0.1

    # Confirm every one of the 5 fingers (right_finger1..right_finger5) received active flexion commands
    for finger_idx in range(1, 6):
        aid = mujoco.mj_name2id(
            bundle.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            f"right_finger{finger_idx}_joint1_actuator",
        )
        assert aid >= 0
        assert abs(float(bundle.data.ctrl[aid])) > 0.1
