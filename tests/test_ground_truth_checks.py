"""Ground-truth simulator check tests across lift, zone placement, button press, drawer slide, and block stacking."""

from __future__ import annotations

import mujoco
import pytest

from library.scene import build_task_scene
from schemas import CheckResult
from sim.check import evaluate, snapshot


def test_ground_truth_cube_pick_lift() -> None:
    bundle = build_task_scene("cube", "pick_lift", {"lift_cm": 10.0}, robot_id="franka_emika_panda")
    m, d = bundle.model, bundle.data
    before = snapshot(bundle)

    # 1. Unmoved -> False
    chk0 = evaluate(bundle, before)
    assert isinstance(chk0, CheckResult)
    assert chk0.success is False
    assert chk0.checks["lifted"] is False

    # 2. Lifted between Panda's closed fingertips -> CheckResult structure & True verified
    ee_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    ee_pos = d.site_xpos[ee_sid].copy()

    # Close Panda finger slide joints slightly into the 6cm cube (half_extent=0.03 -> 0.029 creates firm contact)
    f1_qadr = m.jnt_qposadr[m.joint("finger_joint1").id]
    f2_qadr = m.jnt_qposadr[m.joint("finger_joint2").id]
    d.qpos[f1_qadr] = 0.029
    d.qpos[f2_qadr] = 0.029

    obj_qadr = m.jnt_qposadr[m.joint("target_object_freejoint").id]
    d.qpos[obj_qadr : obj_qadr + 3] = ee_pos
    d.qpos[obj_qadr + 3 : obj_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)

    chk1 = evaluate(bundle, before)
    assert isinstance(chk1, CheckResult)
    assert set(chk1.checks.keys()) == {"lifted", "held"}
    assert chk1.checks["lifted"] is True
    assert chk1.checks["held"] is True
    assert chk1.success is True
    assert chk1.metrics["lift_cm"] >= 10.0
    assert chk1.metrics["fingers_in_contact"] >= 1.0


def test_ground_truth_mug_place_on_pad_zone() -> None:
    bundle = build_task_scene("mug", "place_on_pad", robot_id="franka_emika_panda")
    m, d = bundle.model, bundle.data
    assert bundle.extras["task"]["goal"] == "zone"
    before = snapshot(bundle)

    # Initial pose -> False
    chk0 = evaluate(bundle, before)
    assert chk0.success is False
    assert chk0.checks["on_pad"] is False

    # Move mug freejoint onto zone_center at resting height + mj_forward -> True
    zc = bundle.extras["task"]["zone_center"]
    resting_z = bundle.extras["task"]["table_top_z"] + bundle.extras["task"]["half_height"]
    qadr = m.jnt_qposadr[m.joint("target_object_freejoint").id]
    d.qpos[qadr : qadr + 3] = [zc[0], zc[1], resting_z]
    d.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)

    chk1 = evaluate(bundle, before)
    assert chk1.success is True
    assert chk1.checks == {"on_pad": True, "resting": True, "released": True}


def test_ground_truth_button_panel_press_button() -> None:
    bundle = build_task_scene("button_panel", "press_button", {"button": 1}, robot_id="wonik_allegro")
    m, d = bundle.model, bundle.data
    assert bundle.extras["task"]["goal"] == "button"
    before = snapshot(bundle)

    # Initial qpos=0 -> False
    chk0 = evaluate(bundle, before)
    assert chk0.success is False
    assert chk0.checks["pressed"] is False

    # Depressing button1_joint to 85% travel -> True
    travel = bundle.extras["task"]["child_travel"]["button1"]
    b1_qadr = m.jnt_qposadr[m.joint("button1_joint").id]
    d.qpos[b1_qadr] = 0.85 * travel
    mujoco.mj_forward(m, d)

    chk1 = evaluate(bundle, before)
    assert chk1.success is True
    assert chk1.checks == {"pressed": True, "others_released": True}
    assert chk1.metrics["button1_press"] == pytest.approx(0.85, abs=0.02)


def test_ground_truth_drawer_open() -> None:
    bundle = build_task_scene("drawer", "open_drawer", {"open_cm": 6.0}, robot_id="franka_emika_panda")
    m, d = bundle.model, bundle.data
    assert bundle.extras["task"]["goal"] == "drawer_open"
    before = snapshot(bundle)

    # Initial 0 -> False
    chk0 = evaluate(bundle, before)
    assert chk0.success is False
    assert chk0.checks["drawer_open"] is False

    # Slide target_object_joint open to 7 cm -> True
    j_qadr = m.jnt_qposadr[m.joint("target_object_joint").id]
    d.qpos[j_qadr] = 0.07
    mujoco.mj_forward(m, d)

    chk1 = evaluate(bundle, before)
    assert chk1.success is True
    assert chk1.checks["drawer_open"] is True
    assert chk1.metrics["drawer_open_cm"] == pytest.approx(7.0, abs=0.05)


def test_ground_truth_cube_stack_on_block() -> None:
    bundle = build_task_scene("cube", "stack_on_block", robot_id="franka_emika_panda")
    m, d = bundle.model, bundle.data
    assert bundle.extras["task"]["goal"] == "stack_on"
    before = snapshot(bundle)

    # Initial -> False
    chk0 = evaluate(bundle, before)
    assert chk0.success is False
    assert chk0.checks["on_block"] is False

    # Place target_object centered on top of base_block + mj_forward -> True
    base_pos = d.xpos[m.body("base_block").id].copy()
    bh = bundle.extras["task"]["base_half"]
    half_h = bundle.extras["task"]["half_height"]
    qadr = m.jnt_qposadr[m.joint("target_object_freejoint").id]
    d.qpos[qadr : qadr + 3] = [base_pos[0], base_pos[1], base_pos[2] + bh[2] + half_h]
    d.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)

    chk1 = evaluate(bundle, before)
    assert chk1.success is True
    assert chk1.checks == {"on_block": True, "released": True, "block_upright": True}
