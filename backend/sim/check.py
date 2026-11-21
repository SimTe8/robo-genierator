"""Simulator ground truth for every task preset. The critic's verdict never overrides this.

    before = snapshot(bundle)                       # right before executing
    ...execute the trajectory (mutates bundle.data)...
    result = evaluate(bundle, before, metrics)      # metrics: optional executor metrics (max_lift_cm)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from config import load_config
from schemas import CheckResult
from sim.scene_builder import SceneBundle

# Body-name prefix -> finger, per robot (thumb/index naming kept the same across robots)
FINGERS = {
    "wonik_allegro": {"palm": "palm", "ff_": "index", "mf_": "middle", "rf_": "ring", "th_": "thumb"},
    "wuji_hand": {"right_palm": "palm", "right_finger1": "thumb", "right_finger2": "index", "right_finger3": "middle",
                  "right_finger4": "ring", "right_finger5": "little"},
    # Panda: the two gripper fingers count as "thumb" and "index" so pinch tasks read the same
    "franka_emika_panda": {"hand": "palm", "left_finger": "thumb", "right_finger": "index"},
}


@dataclass
class Snapshot:
    pos: np.ndarray
    mat: np.ndarray
    joint: float | None  # target_object_joint (hinge angle or slide distance)
    nail_z: float | None
    lock: float | None
    others: dict[str, np.ndarray] = field(default_factory=dict)  # multi-object scenes: every extra body's position


def _qpos(bundle: SceneBundle, joint: str) -> float | None:
    m = bundle.model
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint) < 0:
        return None
    return float(bundle.data.qpos[m.jnt_qposadr[m.joint(joint).id]])


def _bpos(bundle: SceneBundle, body: str) -> np.ndarray:
    return bundle.data.xpos[bundle.model.body(body).id].copy()


def snapshot(bundle: SceneBundle) -> Snapshot:
    m, d = bundle.model, bundle.data
    bid = m.body("target_object").id
    nail = None
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "fixture_nail") >= 0:
        nail = float(d.xpos[m.body("fixture_nail").id][2])
    others = {e["body"]: _bpos(bundle, e["body"]) for e in bundle.extras["task"].get("extras", [])}
    return Snapshot(d.xpos[bid].copy(), d.xmat[bid].reshape(3, 3).copy(), _qpos(bundle, "target_object_joint"), nail,
                    _qpos(bundle, "lock_plug_joint"), others)


def hand_contacts(bundle: SceneBundle, body: str = "target_object") -> tuple[set[str], set[str]]:
    """(fingers touching `body` or its children, geom names of `body` touched by the hand)."""
    m, d = bundle.model, bundle.data
    root = m.body(body).id
    finger_map = FINGERS[bundle.extras["robot_id"]]

    def in_object(b: int) -> bool:
        while b > 0:
            if b == root:
                return True
            b = m.body_parentid[b]
        return False

    fingers: set[str] = set()
    parts: set[str] = set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if in_object(b1) == in_object(b2):
            continue
        other_geom, obj_geom = (c.geom2, c.geom1) if in_object(b1) else (c.geom1, c.geom2)
        other = m.body(m.geom_bodyid[other_geom]).name
        prefix = next((p for p in finger_map if other.startswith(p)), None)
        if prefix:
            fingers.add(finger_map[prefix])
            parts.add(m.geom(obj_geom).name)
    return fingers, parts


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1))))


def _yaw(mat: np.ndarray) -> float:
    return math.degrees(math.atan2(mat[1, 0], mat[0, 0]))


def _yaw_err(delta: float, target: float, symmetry: float) -> float:
    sym = symmetry if symmetry and symmetry < 360 else 360.0
    return abs((delta - target + sym / 2) % sym - sym / 2)


def _quat_to_mat(q_wxyz: list[float]) -> np.ndarray:
    w, x, y, z = q_wxyz
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def _press_fraction(bundle: SceneBundle, child: str) -> float:
    q = _qpos(bundle, f"{child}_joint") or 0.0
    return q / bundle.extras["task"]["child_travel"][child]


def _local_point(bundle: SceneBundle, local: list[float]) -> np.ndarray:
    m, d = bundle.model, bundle.data
    bid = m.body("target_object").id
    return d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ np.array(local)


def evaluate(bundle: SceneBundle, before: Snapshot, metrics: dict | None = None) -> CheckResult:
    th = {k: float(v) for k, v in load_config()["checks"].items()}
    t = bundle.extras["task"]
    params = bundle.extras["params"]
    goal = t["goal"]
    metrics = metrics or {}
    now = snapshot(bundle)
    pos, mat = now.pos, now.mat
    top, half_h = t["table_top_z"], t["half_height"]
    fingers, touched = hand_contacts(bundle)
    held = bool(fingers) and (pos[2] - top - half_h) * 100 >= th["held_min_height_cm"]
    released = not fingers
    resting = abs(pos[2] - (top + half_h)) * 100 <= th["rest_tolerance_cm"]
    lift_cm = (pos[2] - before.pos[2]) * 100
    tilt = _angle_between(mat[:, 2], np.array([0, 0, 1.0]))
    sym = t["yaw_symmetry_deg"]

    def lifted_to(target_cm: float) -> bool:
        return lift_cm >= target_cm - th["lift_tolerance_cm"]

    def in_zone(margin_m: float = 0.0, p: np.ndarray | None = None) -> bool:
        p = pos if p is None else p
        c, h = t["zone_center"], t["zone_half"]
        return abs(p[0] - c[0]) <= h[0] + margin_m and abs(p[1] - c[1]) <= h[1] + margin_m

    def none_held(bodies: list[str]) -> bool:
        return all(not hand_contacts(bundle, b)[0] for b in bodies)

    checks: dict[str, bool] = {}
    out = {"lift_cm": round(lift_cm, 2), "tilt_deg": round(tilt, 1), "fingers_in_contact": float(len(fingers))}

    if goal == "lift":
        checks.update(lifted=lifted_to(params["lift_cm"]), held=bool(fingers))
    elif goal == "pinch":
        checks.update(lifted=lifted_to(params["lift_cm"]), thumb_and_index=fingers == {"thumb", "index"})
    elif goal == "part_lift":
        part_geom = f"target_object_{t['part']}"
        checks.update(lifted=lifted_to(params["lift_cm"]), holding_part=part_geom in touched)
    elif goal == "rotate_held":
        err = _yaw_err(_yaw(mat) - _yaw(before.mat), params["angle_deg"], sym)
        checks.update(held=held, rotated=err <= th["yaw_tolerance_deg"])
        out["yaw_error_deg"] = round(err, 1)
    elif goal == "reorient_held":
        target = _quat_to_mat(t["target_quat"])
        if sym == 0:  # round about z: only the z axis is observable
            err = _angle_between(mat[:, 2], target[:, 2])
        else:
            err = math.degrees(Rotation.from_matrix(mat @ target.T).magnitude())
        checks.update(held=held, matches_target=err <= th["orientation_tolerance_deg"])
        out["orientation_error_deg"] = round(err, 1)
    elif goal == "flip":
        flip = _angle_between(mat[:, 2], before.mat[:, 2])
        checks.update(flipped=flip >= th["min_flip_deg"], resting=resting, released=released)
        out["flip_deg"] = round(flip, 1)
    elif goal == "zone":
        checks.update(on_pad=in_zone(), resting=resting, released=released)
    elif goal == "inside":
        checks.update(in_bin=in_zone() and pos[2] < t["rim_z"], released=released)
    elif goal == "push":
        checks["on_pad"] = in_zone()
        if "max_lift_cm" in metrics:
            checks["never_lifted"] = float(metrics["max_lift_cm"]) <= th["max_lift_cm"]
    elif goal == "pose":
        target = _quat_to_mat(t["target_quat"])
        dist_cm = float(np.linalg.norm(pos[:2] - np.array(t["target_pos"][:2]))) * 100
        err = _yaw_err(_yaw(mat) - _yaw(target), 0.0, sym)
        checks.update(at_outline=dist_cm <= th["xy_tolerance_cm"], rotation=err <= th["yaw_tolerance_deg"])
        if "max_lift_cm" in metrics:
            checks["never_lifted"] = float(metrics["max_lift_cm"]) <= th["max_lift_cm"]
        out.update(distance_cm=round(dist_cm, 2), yaw_error_deg=round(err, 1))
    elif goal == "tip":
        drift_cm = float(np.linalg.norm(pos[:2] - before.pos[:2])) * 100
        checks.update(on_its_side=th["min_tilt_deg"] <= tilt <= th["max_tilt_deg"], near_start=drift_cm <= 10, released=released)
    elif goal == "upright":
        upright_rest = abs(pos[2] - (top + t["upright_half_height"])) * 100 <= th["rest_tolerance_cm"]
        checks.update(upright=tilt <= th["upright_tolerance_deg"], resting=upright_rest, released=released)
    elif goal == "face_part":
        heading = mat[:, 0]  # parts like the mug handle sit on the object's +x side
        err = _angle_between(np.array([heading[0], heading[1], 0.0]), np.array([-1.0, 0.0, 0.0]))
        checks.update(part_faces_hand=err <= th["facing_tolerance_deg"], upright=tilt <= th["upright_tolerance_deg"])
        out["facing_error_deg"] = round(err, 1)
    elif goal == "joint":
        assert now.joint is not None and before.joint is not None
        turned = math.degrees(now.joint - before.joint)
        checks["turned"] = turned >= params["angle_deg"] - th["angle_tolerance_deg"]
        out["turned_deg"] = round(turned, 1)
    elif goal in ("key_in", "key_turn"):
        tip = _local_point(bundle, t["key_tip_local"])
        depth_cm = (t["lock_top_z"] - tip[2]) * 100
        off_cm = float(np.linalg.norm(tip[:2] - np.array(t["lock_center"]))) * 100
        checks["key_inserted"] = depth_cm >= th["key_depth_cm"] and off_cm <= 1.0
        out.update(key_depth_cm=round(depth_cm, 2), key_offset_cm=round(off_cm, 2))
        if goal == "key_turn":
            assert now.lock is not None and before.lock is not None
            turned = math.degrees(now.lock - before.lock)
            checks["lock_turned"] = turned >= params["angle_deg"] - th["angle_tolerance_deg"]
            out["lock_turned_deg"] = round(turned, 1)
    elif goal == "peg_hole":
        hole = np.array(t["hole_center"])
        off_cm = float(np.linalg.norm(pos[:2] - hole)) * 100
        depth_cm = (t["board_top_z"] - (pos[2] - half_h)) * 100
        checks.update(in_matching_hole=off_cm <= th["hole_tolerance_cm"], deep_enough=depth_cm >= th["peg_depth_cm"])
        out.update(hole_offset_cm=round(off_cm, 2), peg_depth_cm=round(depth_cm, 2))
    elif goal == "tool_button":
        frac = _press_fraction(bundle, "button")
        checks.update(held=held, button_pressed=frac >= th["button_press_fraction"])
        out["button_press"] = round(frac, 2)
    elif goal == "screw":
        frac = _press_fraction(bundle, "button")
        tip = _local_point(bundle, t["bit_tip_local"])
        dist_cm = float(np.linalg.norm(tip - np.array(t["screw_head"]))) * 100
        checks.update(bit_on_screw=dist_cm <= th["bit_tolerance_cm"], button_pressed=frac >= th["button_press_fraction"])
        out.update(bit_distance_cm=round(dist_cm, 2), button_press=round(frac, 2))
    elif goal in ("button", "two_buttons"):
        fr = {c: _press_fraction(bundle, c) for c in t["children"]}
        want = {f"button{int(params['button'])}"} if goal == "button" else {"button2", "button3"}
        checks.update(pressed=all(fr[c] >= th["button_press_fraction"] for c in want),
                      others_released=all(fr[c] < 0.3 for c in fr if c not in want))
        out.update({f"{c}_press": round(v, 2) for c, v in fr.items()})
    elif goal in ("drawer_open", "drawer_place"):
        assert now.joint is not None
        open_cm = now.joint * 100
        need = params["open_cm"] if goal == "drawer_open" else th["drawer_min_open_cm"]
        checks["drawer_open"] = open_cm >= need
        out["drawer_open_cm"] = round(open_cm, 2)
        if goal == "drawer_place":
            cargo = _bpos(bundle, "cargo")
            local = now.mat.T @ (cargo - pos)  # cargo in the drawer's frame
            checks["block_in_drawer"] = abs(local[0]) < 0.042 and abs(local[1]) < 0.039 and local[2] < 0.02
    elif goal == "pour":
        c, rad = np.array(t["zone_center"][:2]), t["zone_radius"]
        beads = [_bpos(bundle, b) for b in t["beads"]]
        inside = sum(1 for b in beads if np.linalg.norm(b[:2] - c) < rad and b[2] < t["rim_z"] + 0.01)
        checks["poured"] = inside / len(beads) >= th["pour_fraction"]
        out["beads_in_bowl"] = float(inside)
    elif goal == "heaviest":
        lifted_ok = lifted_to(params["lift_cm"])
        others_down = all(abs(_bpos(bundle, n)[2] - (top + t["distractor_half_height"])) * 100 <= th["rest_tolerance_cm"] for n in t["distractors"])
        checks.update(heaviest_lifted=lifted_ok, light_cans_on_table=others_down)
    # ---- multi-object ---------------------------------------------------------------
    elif goal == "tower":
        bodies = ["target_object", *t["stack"]]
        column = sorted((_bpos(bundle, b) for b in bodies), key=lambda v: v[2])
        tol, max_offset = th["stack_tolerance_cm"] / 100, 0.6 * min(t["half_xy"])
        levels = int(abs(column[0][2] - (top + half_h)) <= tol)  # bottom piece rests on the table
        for below, above in zip(column, column[1:]):
            if levels and abs(above[2] - below[2] - 2 * half_h) <= tol and np.linalg.norm(above[:2] - below[:2]) <= max_offset:
                levels += 1
            else:
                break
        checks.update(tower_complete=levels == len(bodies), nothing_held=none_held(bodies))
        out["tower_levels"] = float(levels)
    elif goal == "stack_on":
        base, bh = _bpos(bundle, "base_block"), t["base_half"]
        base_tilt = _angle_between(bundle.data.xmat[bundle.model.body("base_block").id].reshape(3, 3)[:, 2], np.array([0, 0, 1.0]))
        on_top = (abs(pos[0] - base[0]) <= bh[0] and abs(pos[1] - base[1]) <= bh[1]
                  and abs(pos[2] - (base[2] + bh[2] + half_h)) * 100 <= th["rest_tolerance_cm"])
        checks.update(on_block=on_top, released=released, block_upright=base_tilt <= th["upright_tolerance_deg"])
    elif goal == "all_inside":
        bodies = ["target_object", *t["bin_items"]]
        inside = [b for b in bodies if in_zone(p=_bpos(bundle, b)) and _bpos(bundle, b)[2] < t["rim_z"]]
        checks.update(all_in_bin=len(inside) == len(bodies), nothing_held=none_held(bodies))
        out["in_bin"] = float(len(inside))
    elif goal == "sort":
        pads, ph = t["sort_pads"], t["sort_pad_half"]
        sorted_ok = 0
        for body, colour in t["sort_items"].items():
            q, (cx, cy, _) = _bpos(bundle, body), pads[colour]
            sorted_ok += abs(q[0] - cx) <= ph and abs(q[1] - cy) <= ph  # stacked on the right pad also counts
        checks.update(all_sorted=sorted_ok == len(t["sort_items"]), nothing_held=none_held(list(t["sort_items"])))
        out["sorted"] = float(sorted_ok)
    elif goal == "clutter_pick":
        moved = max((float(np.linalg.norm(_bpos(bundle, b) - before.others[b])) for b in t["clutter_distractors"]), default=0.0)
        checks.update(lifted=lifted_to(params["lift_cm"]), held=bool(fingers), others_undisturbed=moved * 100 <= th["clutter_disturb_cm"])
        out["max_other_moved_cm"] = round(moved * 100, 2)
    elif goal == "nail":
        assert now.nail_z is not None and before.nail_z is not None
        depth_cm = (before.nail_z - now.nail_z) * 100
        checks["nail_driven"] = depth_cm >= params["depth_cm"]
        out["nail_depth_cm"] = round(depth_cm, 2)
    else:
        raise ValueError(f"no success rule for goal '{goal}'")

    return CheckResult(success=all(checks.values()), checks=checks, metrics=out)
