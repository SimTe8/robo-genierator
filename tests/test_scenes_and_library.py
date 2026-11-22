"""Unit and physics tests for procedural scenes, library scenes, fault injection, and kinematics summaries."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from library.scene import _robot_touches_objects, build_task_scene, describe
from sim.scene_builder import build_scene, summarize_kinematics


@pytest.mark.parametrize(
    ("robot_id", "task_type"),
    [
        ("franka_emika_panda", "pick_and_lift"),
        ("franka_emika_panda", "obstacle_grasp"),
        ("wonik_allegro", "pick_and_lift"),
        ("wonik_allegro", "in_hand_rotation"),
        ("wonik_allegro", "in_hand_spindle"),
    ],
)
def test_build_scene_robots_and_tasks(robot_id: str, task_type: str) -> None:
    bundle = build_scene(robot_id=robot_id, scene_seed=42, task_type=task_type)  # type: ignore[arg-type]
    m, d = bundle.model, bundle.data

    assert bundle.robot_id == robot_id
    assert bundle.task_type == task_type
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_object") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "side_cam") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "top_cam") >= 0
    assert np.all(np.isfinite(d.qpos))

    if task_type == "obstacle_grasp":
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_roof") >= 0
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_post") >= 0
    elif task_type == "in_hand_spindle":
        spindle_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "target_object_spindle")
        assert spindle_jid >= 0
        assert m.jnt_type[spindle_jid] == mujoco.mjtJoint.mjJNT_HINGE
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "spindle_pin") >= 0
    elif task_type == "in_hand_rotation":
        free_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "target_object_freejoint")
        assert free_jid >= 0
        assert m.jnt_type[free_jid] == mujoco.mjtJoint.mjJNT_FREE
    elif robot_id == "wonik_allegro":
        mocap_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm_mocap")
        assert mocap_bid >= 0
        assert m.body_mocapid[mocap_bid] >= 0


@pytest.mark.parametrize(
    ("shape", "expected_mjt"),
    [
        ("box", mujoco.mjtGeom.mjGEOM_BOX),
        ("cylinder", mujoco.mjtGeom.mjGEOM_CYLINDER),
        ("sphere", mujoco.mjtGeom.mjGEOM_SPHERE),
    ],
)
def test_build_scene_object_shapes(shape: str, expected_mjt: int) -> None:
    bundle = build_scene(robot_id="franka_emika_panda", scene_seed=7, object_shape=shape)
    gid = mujoco.mj_name2id(bundle.model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom")
    assert gid >= 0
    assert bundle.obj_spec.shape == shape
    assert bundle.obj_spec.half_height > 0.0
    assert bundle.model.geom_type[gid] == expected_mjt


@pytest.mark.parametrize("robot_id", ["franka_emika_panda", "wonik_allegro", "wuji_hand"])
def test_build_task_scene_across_all_three_robots(robot_id: str) -> None:
    bundle = build_task_scene("cube", "pick_lift", robot_id=robot_id)
    m = bundle.model
    assert bundle.robot_id == robot_id
    assert bundle.extras["robot_id"] == robot_id
    assert bundle.extras["object_id"] == "cube"
    assert bundle.extras["task_id"] == "pick_lift"
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_object") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site") >= 0
    if robot_id == "wuji_hand":
        assert m.nu == 20
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm_mocap") >= 0


@pytest.mark.parametrize(
    ("object_id", "task_id", "params", "expected_joint_or_body", "obj_type"),
    [
        ("cube", "pick_lift", {}, "target_object_freejoint", mujoco.mjtObj.mjOBJ_JOINT),
        ("mug", "place_on_pad", {}, "fixture_pad", mujoco.mjtObj.mjOBJ_BODY),
        ("button_panel", "press_button", {"button": 1}, "button1_joint", mujoco.mjtObj.mjOBJ_JOINT),
        ("drawer", "open_drawer", {}, "target_object_joint", mujoco.mjtObj.mjOBJ_JOINT),
        ("key", "insert_key", {}, "lock_plug_joint", mujoco.mjtObj.mjOBJ_JOINT),
    ],
)
def test_build_task_scene_multiple_objects_and_fixtures(
    object_id: str,
    task_id: str,
    params: dict[str, float],
    expected_joint_or_body: str,
    obj_type: int,
) -> None:
    bundle = build_task_scene(object_id, task_id, params, robot_id="wonik_allegro")
    assert mujoco.mj_name2id(bundle.model, obj_type, expected_joint_or_body) >= 0
    desc = describe(bundle)
    assert desc["robot"] == "wonik_allegro"
    assert len(desc["object_position"]) == 3


def test_build_task_scene_clutter_and_deterministic_layout_seed() -> None:
    b1 = build_task_scene(
        "cube",
        "pick_lift",
        robot_id="wonik_allegro",
        clutter=["can", "ball"],
        layout_seed=123,
    )
    b2 = build_task_scene(
        "cube",
        "pick_lift",
        robot_id="wonik_allegro",
        clutter=["can", "ball"],
        layout_seed=123,
    )

    assert mujoco.mj_name2id(b1.model, mujoco.mjtObj.mjOBJ_BODY, "clutter_1") >= 0
    assert mujoco.mj_name2id(b1.model, mujoco.mjtObj.mjOBJ_BODY, "clutter_2") >= 0
    extra_ids = [e["object_id"] for e in b1.extras["task"]["extras"]]
    assert extra_ids == ["can", "ball"]

    # Deterministic layout_seed=123 produces identical body positions & qpos across calls
    assert np.allclose(b1.data.xpos, b2.data.xpos)
    assert np.allclose(b1.data.qpos, b2.data.qpos)

    # Zero initial robot-object collisions
    assert _robot_touches_objects(b1) is False


def test_live_fault_injection_panda_and_wuji() -> None:
    # 1. Panda joint4 fault injection
    panda_bundle = build_scene(robot_id="franka_emika_panda", scene_seed=42, disabled_joints=["joint4"])
    pm = panda_bundle.model
    p_jid = mujoco.mj_name2id(pm, mujoco.mjtObj.mjOBJ_JOINT, "joint4")
    assert p_jid >= 0
    assert pm.jnt_limited[p_jid] == 1
    assert 0.0 < (pm.jnt_range[p_jid, 1] - pm.jnt_range[p_jid, 0]) <= 2.5e-4
    p_acts = [aid for aid in range(pm.nu) if pm.actuator_trnid[aid, 0] == p_jid]
    assert len(p_acts) >= 1
    for aid in p_acts:
        assert np.allclose(pm.actuator_gainprm[aid, :], 0.0)
        assert np.allclose(pm.actuator_biasprm[aid, :], 0.0)

    # 2. Wuji right_finger3_joint1 fault injection
    wuji_bundle = build_task_scene(
        "cube",
        "pick_lift",
        robot_id="wuji_hand",
        disabled_joints=["right_finger3_joint1"],
    )
    wm = wuji_bundle.model
    w_jid = mujoco.mj_name2id(wm, mujoco.mjtObj.mjOBJ_JOINT, "right_finger3_joint1")
    assert w_jid >= 0
    assert wm.jnt_limited[w_jid] == 1
    assert 0.0 < (wm.jnt_range[w_jid, 1] - wm.jnt_range[w_jid, 0]) <= 2.5e-4
    w_acts = [aid for aid in range(wm.nu) if wm.actuator_trnid[aid, 0] == w_jid]
    assert len(w_acts) >= 1
    for aid in w_acts:
        assert np.allclose(wm.actuator_gainprm[aid, :], 0.0)
        assert np.allclose(wm.actuator_biasprm[aid, :], 0.0)


def test_summarize_kinematics_contains_robot_object_and_actuators() -> None:
    bundle = build_scene(robot_id="franka_emika_panda", scene_seed=42)
    summary = summarize_kinematics(bundle)

    assert isinstance(summary, str) and len(summary) > 100
    assert "franka_emika_panda" in summary
    assert "Target object:" in summary
    assert f"shape={bundle.obj_spec.shape}" in summary
    assert f"Actuators ({bundle.model.nu}):" in summary
    assert "ctrlrange=[" in summary
