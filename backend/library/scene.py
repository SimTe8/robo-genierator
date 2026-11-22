"""Build the MuJoCo scene for (robot, object, task).

The UI only picks robot + object + task. Conditions (pose, scale, mass, friction, faults) are the
PLANNER's job: it calls `build_task_scene(...)` with its own values. Defaults give a fixed scene;
`layout_seed=N` reproducibly shuffles where the target and the other objects stand (UI: "Shuffle layout").

    bundle = build_task_scene("cube", "rotate_in_hand", {"angle_deg": 90}, robot_id="wuji_hand",
                              object_xy=(0.01, -0.02), object_yaw_deg=30, size_scale=1.1)
    bundle.extras["task"]      # goal + resolved params + targets, used by sim/check.py

All positions below are relative to the robot's workspace origin (ORIGIN) on its table.
Robots: Allegro and Franka Panda come from mujoco_menagerie via sim/scene_builder.py (unchanged);
the Wuji hand (MIT, wuji-technology/wuji-hand-description) is mounted here the same way as the
Allegro: a free palm welded to a mocap body named `palm_mocap`, so the executor can move it.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from config import REPO_ROOT, load_config
from library.generated import refresh_generated
from library.multi import (BASE_BLOCK, Layout, BLUE, LOOKALIKE_DY, MAX_CLUTTER, PUSH_OFFSET, RED, SORT_PAD_HALF, SORT_PADS, add_extra,
                           can_be_extra, describe_extras, plan_layout, recolour_main)
from library.objects import OBJECTS_BY_ID, ObjectDef, Part, add_parts, default_part
from library.tasks import TASKS_BY_ID, TaskPreset
from sim.check import FINGERS
from sim.scene_builder import ObjectSpec, SceneBundle, _add_cameras, _add_table, build_scene

ROBOTS = ("wonik_allegro", "wuji_hand", "franka_emika_panda")
ORIGIN = {"wonik_allegro": (0.0, 0.0), "wuji_hand": (0.0, 0.0), "franka_emika_panda": (0.50, 0.0)}

FIXTURE_XY = {"target_pad": (0.0, -0.10), "bin": (0.0, -0.115), "nail_board": (0.0, -0.08), "lock": (0.0, -0.085),
              "shape_board": (0.0, -0.09), "screw_board": (0.0, -0.085), "cargo": (-0.02, 0.09), "bowl": (0.0, -0.10)}
MOVABLE_ZONE_OUTER_HALF = {"target_pad": 0.045, "bin": 0.060, "bowl": 0.054}
TABLE_GEOM_BY_ROBOT = {
    "franka_emika_panda": ((0.52, 0.0), (0.28, 0.35)),
    "wonik_allegro": ((0.0, 0.0), (0.20, 0.20)),
    "wuji_hand": ((0.0, 0.0), (0.20, 0.20)),
}
GHOST_AIR_POS = (0.07, 0.12, 0.09)  # beside the hand, height above the table

# Lock geometry (keyhole on top, vertical insertion)
LOCK_H = 0.035          # housing height above the table
SLOT_HALF_GAP = 0.0022  # keyhole half-width (key blade is 2.4 mm thick)
# Collision bits: key = 3, plug = 2, everything else = 1 -> the plug only touches the key, never the housing or hand
PLUG_BITS = 2


def _quat_wxyz(r: Rotation) -> list[float]:
    x, y, z, w = r.as_quat()
    return [float(w), float(x), float(y), float(z)]


def resolve_params(task: TaskPreset, obj: ObjectDef, params: dict[str, float] | None) -> dict[str, float]:
    """Preset defaults < object-specific defaults < explicit params."""
    ref = next((r for r in obj.tasks if r.task_id == task.id), None)
    out = {p.key: p.default for p in task.params}
    out.update(ref.params if ref else {})
    out.update(params or {})
    return {k: float(v) for k, v in out.items()}


def check_combo(robot_id: str, object_id: str, task_id: str, clutter: list[str] | None = None) -> str | None:
    """Why this combination is not allowed (None = fine)."""
    refresh_generated()
    obj, task = OBJECTS_BY_ID.get(object_id), TASKS_BY_ID.get(task_id)
    if robot_id not in ROBOTS:
        return f"Unknown robot '{robot_id}'."
    if obj is None or task is None:
        return f"Unknown object '{object_id}' or task '{task_id}'."
    if task_id not in {r.task_id for r in obj.tasks}:
        return f"'{task.name}' is not a preset for the {obj.name.lower()}."
    if robot_id not in task.robots:
        return f"'{task.name}' needs a multi-finger hand."
    clutter = clutter or []
    if len(clutter) > MAX_CLUTTER:
        return f"At most {MAX_CLUTTER} extra objects."
    for cid in clutter:
        extra = OBJECTS_BY_ID.get(cid)
        if extra is None:
            return f"Unknown extra object '{cid}'."
        if not can_be_extra(extra):
            return f"The {extra.name.lower()} can't be added as an extra object (articulated or has its own parts)."
    return None


def build_task_scene(
    object_id: str,
    task_id: str,
    params: dict[str, float] | None = None,
    *,
    robot_id: str = "wonik_allegro",
    object_xy: tuple[float, float] | None = None,
    object_yaw_deg: float | None = None,
    size_scale: float = 1.0,
    mass_scale: float = 1.0,
    friction: float | None = None,
    disabled_joints: list[str] | None = None,
    clutter: list[str] | None = None,
    layout_seed: int | None = None,
    fixture_xy_jitter: tuple[float, float] | None = None,
) -> SceneBundle:
    """`clutter`: extra library object ids to put on the table (multi-object scenes, any task).
    `layout_seed`: None = default layout; an int = reproducible shuffle of positions (+ yaw where safe).
    `fixture_xy_jitter`: optional (dx, dy) shift in meters for movable tabletop zones (`target_pad`, `bin`, `bowl`).
    Explicit `object_xy` / `object_yaw_deg` override the shuffle for the target."""
    problem = check_combo(robot_id, object_id, task_id, clutter)
    if problem:
        raise ValueError(problem)
    obj, task = OBJECTS_BY_ID[object_id], TASKS_BY_ID[task_id]
    p = resolve_params(task, obj, params)
    ox, oy = ORIGIN[robot_id]

    t_center, t_half = TABLE_GEOM_BY_ROBOT[robot_id]
    shifted_fixture_xy: tuple[float, float] | None = None
    fx_outer_half = MOVABLE_ZONE_OUTER_HALF.get(task.fixture or "", 0.0)
    if fixture_xy_jitter is not None and task.fixture in MOVABLE_ZONE_OUTER_HALF:
        base_fx, base_fy = FIXTURE_XY[task.fixture]
        margin = fx_outer_half + 0.005
        fx_cx = float(
            np.clip(
                ox + base_fx + float(fixture_xy_jitter[0]),
                t_center[0] - t_half[0] + margin,
                t_center[0] + t_half[0] - margin,
            )
        )
        fx_cy = float(
            np.clip(
                oy + base_fy + float(fixture_xy_jitter[1]),
                t_center[1] - t_half[1] + margin,
                t_center[1] + t_half[1] - margin,
            )
        )
        shifted_fixture_xy = (fx_cx, fx_cy)

    obj_horiz_r = max(
        obj.half_extents[0],
        obj.half_extents[2] if task.initial_pose == "lying" else obj.half_extents[1],
    ) * size_scale

    def _nudge_rel_clear_of_fixture(rel_xy: tuple[float, float]) -> tuple[float, float]:
        if shifted_fixture_xy is None or object_xy is not None:
            return rel_xy
        wx, wy = ox + rel_xy[0], oy + rel_xy[1]
        fx_cx, fx_cy = shifted_fixture_xy
        min_sep = fx_outer_half + obj_horiz_r + 0.015
        if abs(wx - fx_cx) < min_sep and abs(wy - fx_cy) < min_sep:
            obj_margin = obj_horiz_r + 0.012
            # Push along +Y (or -Y if +Y has less room on table) so target starts outside the shifted zone
            cand_up = fx_cy + min_sep
            cand_dn = fx_cy - min_sep
            y_hi = t_center[1] + t_half[1] - obj_margin
            y_lo = t_center[1] - t_half[1] + obj_margin
            if cand_up <= y_hi:
                wy = cand_up
            elif cand_dn >= y_lo:
                wy = cand_dn
            else:
                wx = float(
                    np.clip(
                        fx_cx + (min_sep if wx >= fx_cx else -min_sep),
                        t_center[0] - t_half[0] + obj_margin,
                        t_center[0] + t_half[0] - obj_margin,
                    )
                )
            return (wx - ox, wy - oy)
        return rel_xy

    def assemble(layout: Layout, force_clear_fixture: bool = False) -> SceneBundle:
        rel, extras, multi_info = layout.target_xy, layout.extras, layout.info
        if force_clear_fixture:
            rel = _nudge_rel_clear_of_fixture(rel)
        xy = (ox + rel[0], oy + rel[1])

        def populate(spec: mujoco.MjSpec, _robot: str, top_z: float, _center, _half):
            r0 = Rotation.from_euler("z", layout.target_yaw_deg, degrees=True)
            hz = obj.half_extents[2] * size_scale
            if obj.articulation is not None:
                obj_spec = _add_articulated(spec, obj, xy, top_z, r0, size_scale)
            else:
                if task.initial_pose == "lying":
                    r0 = r0 * Rotation.from_euler("x", 90, degrees=True)
                    hz = obj.half_extents[1] * size_scale
                body = spec.worldbody.add_body()
                body.name = "target_object"
                body.pos = [xy[0], xy[1], top_z + hz + 0.001]
                body.quat = _quat_wxyz(r0)
                j = body.add_joint()
                j.name = "target_object_freejoint"
                j.type = mujoco.mjtJoint.mjJNT_FREE
                rgba = tuple(multi_info.get("target_rgba", obj.rgba))
                parts = recolour_main(obj.parts) if "target_rgba" in multi_info else obj.parts
                add_parts(body, parts, rgba, obj.mass_kg, size_scale)
                _add_children(body, obj, size_scale)
                hx, hy, _ = (v * size_scale for v in obj.half_extents)
                obj_spec = ObjectSpec("box", [hx, hy, hz], list(body.pos), _quat_wxyz(r0), list(obj.rgba), obj.mass_kg * size_scale**3)

            info = _add_fixture_and_targets(
                spec,
                obj,
                task,
                p,
                xy,
                (ox, oy),
                top_z,
                r0,
                size_scale,
                fixture_xy_override=shifted_fixture_xy,
            )
            if shifted_fixture_xy is not None and fixture_xy_jitter is not None:
                info["fixture_xy_jitter"] = [round(float(fixture_xy_jitter[0]), 4), round(float(fixture_xy_jitter[1]), 4)]
            info.update(_add_special(spec, obj, xy, top_z, size_scale))
            info.update(multi_info)
            info.update(layout_seed=layout.seed, target_xy_rel=[round(v, 4) for v in rel], target_yaw_deg=round(layout.target_yaw_deg, 1))
            if "sort_pads" in info:
                info["sort_pads"] = {c: [ox + x, oy + y, top_z] for c, (x, y) in info["sort_pads"].items()}
            if any(e.name == "base_block" for e in extras):
                info["base_half"] = list(BASE_BLOCK.half_extents)
            info["extras"] = describe_extras(extras)
            for e in extras:
                add_extra(spec, e, (ox, oy), top_z)
            info.update(goal=task.goal, part=default_part(obj), yaw_symmetry_deg=obj.yaw_symmetry_deg,
                        start_quat=_quat_wxyz(r0), table_top_z=top_z, half_height=hz, origin=[ox, oy],
                        upright_half_height=obj.half_extents[2] * size_scale,
                        half_xy=[obj.half_extents[0] * size_scale, obj.half_extents[1] * size_scale], children=[c.name for c in obj.children],
                        child_travel={c.name: c.range[1] * size_scale for c in obj.children})
            return obj_spec, {"robot_id": robot_id, "object_id": obj.id, "task_id": task.id, "params": p, "task": info}

        if robot_id == "wuji_hand":
            bundle = _build_wuji_scene(populate, disabled_joints)
        else:
            bundle = build_scene(robot_id=robot_id, scene_seed=0, disabled_joints=disabled_joints, populate=populate)  # type: ignore[arg-type]

        m = bundle.model
        if mass_scale != 1.0 or friction is not None:
            bid = m.body("target_object").id
            m.body_mass[bid] *= mass_scale
            m.body_inertia[bid] *= mass_scale
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == bid and friction is not None:
                    m.geom_friction[g, 0] = friction
        mujoco.mj_forward(m, bundle.data)
        return bundle

    def layout_for(seed: int | None, attempt: int = 0) -> Layout:
        return plan_layout(task, obj, p, list(clutter or []), size_scale, seed=seed, attempt=attempt,
                           object_xy=object_xy, object_yaw_deg=object_yaw_deg)

    if layout_seed is None:
        return assemble(layout_for(None), force_clear_fixture=True)
    # Shuffled: the sampler only knows table/fixture/object footprints, not the robot's start pose or shifted fixture,
    # so re-draw (deterministically: next draw of the same seed) if anything starts in contact or overlaps shifted zone.
    base_tries = int(load_config().get("layout", {}).get("max_redraws", 8))
    tries = max(base_tries, 20) if shifted_fixture_xy is not None else base_tries
    for attempt in range(tries):
        bundle = assemble(layout_for(layout_seed, attempt), force_clear_fixture=False)
        if not _robot_touches_objects(bundle) and not _shifted_fixture_overlaps(
            bundle, shifted_fixture_xy, fx_outer_half, obj_horiz_r
        ):
            return bundle
    bundle = assemble(layout_for(None), force_clear_fixture=True)  # give up: default layout (nudged clear of shifted fixture)
    bundle.extras["task"].update(layout_seed=layout_seed, layout_note=f"shuffle failed after {tries} draws; default layout used")
    return bundle


def _shifted_fixture_overlaps(
    bundle: SceneBundle,
    shifted_fixture_xy: tuple[float, float] | None,
    fx_outer_half: float,
    obj_horiz_r: float,
) -> bool:
    """True if target_object or any extra/distractor body overlaps the shifted fixture zone."""
    if shifted_fixture_xy is None:
        return False
    m, d = bundle.model, bundle.data
    fx_cx, fx_cy = shifted_fixture_xy
    t = bundle.extras["task"]

    obj_bid = m.body("target_object").id
    ox_w, oy_w = float(d.xpos[obj_bid, 0]), float(d.xpos[obj_bid, 1])
    min_obj_sep = fx_outer_half + obj_horiz_r + 0.012
    if abs(ox_w - fx_cx) < min_obj_sep and abs(oy_w - fx_cy) < min_obj_sep:
        return True

    other_bodies = [*(e["body"] for e in t.get("extras", [])), *t.get("distractors", [])]
    for bname in other_bodies:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid < 0:
            continue
        bx, by = float(d.xpos[bid, 0]), float(d.xpos[bid, 1])
        b_r = max(
            (float(max(m.geom_size[g, 0], m.geom_size[g, 1])) for g in range(m.ngeom) if m.geom_bodyid[g] == bid),
            default=0.025,
        )
        if abs(bx - fx_cx) < fx_outer_half + b_r + 0.010 and abs(by - fx_cy) < fx_outer_half + b_r + 0.010:
            return True

    for i in range(d.ncon):
        c = d.contact[i]
        bnames = [m.body(int(m.geom_bodyid[g])).name for g in (c.geom1, c.geom2)]
        if any(b.startswith(("fixture_bin", "fixture_bowl")) for b in bnames):
            if any(b in ("target_object", *other_bodies) for b in bnames):
                return True
    return False


def _robot_touches_objects(bundle: SceneBundle) -> bool:
    """True if any robot body touches the target or an extra object right after building."""
    m, d = bundle.model, bundle.data
    robot_roots = tuple(FINGERS[bundle.extras["robot_id"]]) + ("link",)  # hands: palm-rooted; Panda: link0-rooted
    t = bundle.extras["task"]
    ours = {"target_object", "target_base", *(e["body"] for e in t.get("extras", [])), *t.get("distractors", [])}
    for i in range(d.ncon):
        c = d.contact[i]
        roots = [m.body(int(m.body_rootid[m.geom_bodyid[g]])).name for g in (c.geom1, c.geom2)]
        if any(r.startswith(robot_roots) for r in roots) and any(r in ours for r in roots):
            return True
    return False


# ---------------------------------------------------------------------------------------------
# Robots
# ---------------------------------------------------------------------------------------------


def _build_wuji_scene(populate, disabled_joints: list[str] | None) -> SceneBundle:
    """Wuji right hand on a mocap-welded free palm over a table (same layout as the Allegro scene)."""
    path = REPO_ROOT / load_config()["wuji_mjcf"]
    if not path.exists():
        raise FileNotFoundError(f"Wuji hand model not found at {path}. See README: download wuji-hand-description.")
    spec = mujoco.MjSpec.from_file(str(path))
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    tex = spec.add_texture()
    tex.name = "grid"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.rgb1, tex.rgb2 = [0.2, 0.3, 0.4], [0.1, 0.2, 0.3]
    tex.width = tex.height = 300
    mat = spec.add_material()
    mat.name = "grid"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    mat.texrepeat = [5, 5]
    mat.reflectance = 0.2
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    floor.material = "grid"
    floor.pos = [0, 0, -0.3]
    light = spec.worldbody.add_light()
    light.pos = [0, 0, 1.5]
    light.dir = [0, 0, -1]

    # Palm frame: fingers along +z, fingers flex toward +x. Point the fingers forward (+x, world)
    # and roll so they flex downward: palm faces the table, like the Allegro.
    pose = Rotation.from_euler("x", 90, degrees=True) * Rotation.from_euler("y", 90, degrees=True) * Rotation.from_euler("z", -90, degrees=True)
    palm_pos = [-0.13, 0.0, 0.10]
    palm = spec.body("right_palm_link")
    palm.pos = palm_pos
    palm.quat = _quat_wxyz(pose)
    fj = palm.add_joint()
    fj.name = "palm_freejoint"
    fj.type = mujoco.mjtJoint.mjJNT_FREE
    site = palm.add_site()
    site.name = "ee_site"
    # Place TCP at the opposed thumb-and-fingers pinch center so [0, 0, 0] centers fingers on the object
    site.pos = [0.055, 0.005, 0.115]
    site.size = [0.008, 0.008, 0.008]
    site.rgba = [0.1, 0.9, 0.1, 0.6]
    mocap = spec.worldbody.add_body()
    mocap.name = "palm_mocap"
    mocap.mocap = True
    mocap.pos = palm_pos
    mocap.quat = _quat_wxyz(pose)
    weld = spec.add_equality()
    weld.type = mujoco.mjtEq.mjEQ_WELD
    weld.name1, weld.name2 = "palm_mocap", "right_palm_link"
    weld.objtype = mujoco.mjtObj.mjOBJ_BODY
    weld.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
    weld.solref = [0.01, 1.0]

    top_z = -0.08
    center, half = (0.0, 0.0), (0.20, 0.20, 0.01)
    _add_table(spec, center, half, top_z)
    obj_spec, extras = populate(spec, "wuji_hand", top_z, center, half)
    _add_cameras(spec, "wonik_allegro")  # same workspace layout as the Allegro scene
    model = spec.compile()
    # Give Wuji finger servos realistic torque authority & 4D torsional pad friction
    model.actuator_gainprm[:, 0] *= 4.5
    model.actuator_biasprm[:, 1] *= 4.5
    for gid in range(model.ngeom):
        if "right_finger" in (model.body(int(model.geom_bodyid[gid])).name or ""):
            model.geom_condim[gid] = 4
            model.geom_friction[gid] = [1.3, 0.01, 0.0005]
    data = mujoco.MjData(model)
    # Initialize thumb CMC joint1 in opposed downward posture so hand starts ready to pinch
    th_opp_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_finger1_joint1")
    if th_opp_jid >= 0:
        lo, hi = model.jnt_range[th_opp_jid]
        q_opp = lo + 0.88 * (hi - lo)
        data.qpos[model.jnt_qposadr[th_opp_jid]] = q_opp
        if model.nu > 0:
            data.ctrl[0] = q_opp
    for jname in disabled_joints or []:  # lock like scene_builder does
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            q0 = float(data.qpos[model.jnt_qposadr[jid]])
            model.jnt_limited[jid] = 1
            model.jnt_range[jid] = [q0 - 1e-4, q0 + 1e-4]
            for aid in range(model.nu):
                if model.actuator_trnid[aid, 0] == jid:
                    model.actuator_gainprm[aid, :] = 0.0
                    model.actuator_biasprm[aid, :] = 0.0
    mujoco.mj_forward(model, data)
    return SceneBundle(model=model, data=data, robot_id="wuji_hand", scene_seed=0, obj_spec=obj_spec,  # type: ignore[arg-type]
                       table_top_z=top_z, disabled_joints=list(disabled_joints or []), extras=extras)


# ---------------------------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------------------------


def _add_children(body, obj: ObjectDef, scale: float) -> None:
    for c in obj.children:
        cb = body.add_body()
        cb.name = f"target_object_{c.name}"
        cb.pos = [v * scale for v in c.pos]
        j = cb.add_joint()
        j.name = f"{c.name}_joint"
        j.type = mujoco.mjtJoint.mjJNT_SLIDE if c.kind == "slide" else mujoco.mjtJoint.mjJNT_HINGE
        j.axis = list(c.axis)
        j.range = [c.range[0] * scale, c.range[1] * scale]
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        j.stiffness = [c.stiffness, 0.0, 0.0]  # MuJoCo 4: 3-vector; spring pulls back to qpos 0
        j.damping = [c.damping, 0.0, 0.0]
        add_parts(cb, c.parts, c.rgba, 0.005, scale, name_prefix=f"target_object_{c.name}")


def _add_articulated(spec, obj: ObjectDef, xy, top_z, r0: Rotation, scale: float) -> ObjectSpec:
    art = obj.articulation
    assert art is not None
    pivot = [xy[0], xy[1], top_z + art.pivot_z * scale]
    if art.base_parts:
        base = spec.worldbody.add_body()
        base.name = "target_base"
        base.pos = pivot
        add_parts(base, art.base_parts, art.base_rgba, 0.0, scale, name_prefix="target_base")
    body = spec.worldbody.add_body()
    body.name = "target_object"
    body.pos = pivot
    body.quat = _quat_wxyz(r0)
    if art.kind != "static":
        j = body.add_joint()
        j.name = "target_object_joint"
        j.type = mujoco.mjtJoint.mjJNT_HINGE if art.kind == "hinge" else mujoco.mjtJoint.mjJNT_SLIDE
        j.axis = list(art.axis)
        j.damping = [art.damping, 0.0, 0.0]
        j.frictionloss = art.frictionloss
        if art.range:
            j.range = list(art.range)
            j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    add_parts(body, obj.parts, obj.rgba, obj.mass_kg if art.kind != "static" else 0.0, scale)
    _add_children(body, obj, scale)
    hx, hy, hz = (v * scale for v in obj.half_extents)
    return ObjectSpec("box", [hx, hy, hz], pivot, _quat_wxyz(r0), list(obj.rgba), obj.mass_kg)


def _add_special(spec, obj: ObjectDef, xy, top_z, scale) -> dict[str, Any]:
    """Extra bodies that belong to the object (beads inside the cup, look-alike cans)."""
    if obj.special == "beads":
        names = []
        r = 0.0065
        for i in range(10):
            b = spec.worldbody.add_body()
            b.name = f"bead_{i}"
            ang = i * 2.4
            rad = 0.008 + 0.004 * (i % 3)
            b.pos = [xy[0] + rad * math.cos(ang), xy[1] + rad * math.sin(ang), top_z + 0.012 + 0.014 * (i // 4)]
            j = b.add_joint()
            j.type = mujoco.mjtJoint.mjJNT_FREE
            g = b.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size = [r, 0, 0]
            g.mass = 0.003
            g.rgba = [0.95, 0.35 + 0.05 * (i % 4), 0.2, 1]
            names.append(b.name)
        return {"beads": names}
    if obj.special == "lookalike_cans":
        names = []
        for i, dy in enumerate((LOOKALIKE_DY, -LOOKALIKE_DY)):
            b = spec.worldbody.add_body()
            b.name = f"distractor_{i}"
            b.pos = [xy[0], xy[1] + dy, top_z + obj.half_extents[2] * scale + 0.001]
            j = b.add_joint()
            j.type = mujoco.mjtJoint.mjJNT_FREE
            add_parts(b, obj.parts, obj.rgba, 0.05, scale, name_prefix=f"distractor_{i}")  # empty: 50 g vs 400 g
            names.append(b.name)
        return {"distractors": names, "distractor_half_height": obj.half_extents[2] * scale}
    return {}


# ---------------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------------


def _static_box(parent, name: str, pos, half, rgba, collide: bool = True, bits: int = 1, quat=None):
    b = parent.add_body()
    b.name = name
    b.pos = list(pos)
    if quat is not None:
        b.quat = quat
    g = b.add_geom()
    g.name = f"{name}_geom"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = list(half)
    g.rgba = list(rgba)
    g.contype = g.conaffinity = bits if collide else 0
    return b


def _ring(parent, prefix: str, center, inner: float, wall: float, height: float, z0: float, rgba, n: int = 8, bits: int = 1):
    """Approximate a round hole/bowl wall with `n` boxes around `center` (inner radius `inner`)."""
    for k in range(n):
        a = 2 * math.pi * k / n
        r = inner + wall / 2
        half_len = (inner + wall) * math.tan(math.pi / n) + 0.0005
        _static_box(parent, f"{prefix}{k}", (center[0] + r * math.cos(a), center[1] + r * math.sin(a), z0 + height / 2),
                    (wall / 2, half_len, height / 2), rgba, bits=bits, quat=_quat_wxyz(Rotation.from_euler("z", a)))


def _add_fixture_and_targets(
    spec,
    obj: ObjectDef,
    task: TaskPreset,
    p,
    xy,
    origin,
    top_z,
    r0: Rotation,
    scale,
    fixture_xy_override: tuple[float, float] | None = None,
) -> dict[str, Any]:
    info: dict[str, Any] = {"fixture": task.fixture}
    fx = task.fixture
    wb = spec.worldbody

    def at(name: str) -> tuple[float, float]:
        if fixture_xy_override is not None and name in MOVABLE_ZONE_OUTER_HALF:
            return fixture_xy_override
        return origin[0] + FIXTURE_XY[name][0], origin[1] + FIXTURE_XY[name][1]

    if fx == "target_pad":
        cx, cy = at("target_pad")
        half = 0.045
        _static_box(wb, "fixture_pad", (cx, cy, top_z + 0.0005), (half, half, 0.0005), (0.15, 0.85, 0.35, 0.55), collide=False)
        info.update(zone_center=[cx, cy, top_z], zone_half=[half, half])
    elif fx == "sort_pads":
        for colour, (dx, dy) in SORT_PADS.items():
            rgba = (*(RED if colour == "red" else BLUE)[:3], 0.45)
            _static_box(wb, f"fixture_pad_{colour}", (origin[0] + dx, origin[1] + dy, top_z + 0.0005),
                        (SORT_PAD_HALF, SORT_PAD_HALF, 0.0005), rgba, collide=False)
    elif fx == "bin":
        cx, cy = at("bin")
        inner, wall, h = 0.055, 0.005, 0.035
        rgba = (0.55, 0.60, 0.68, 1)
        _static_box(wb, "fixture_bin_floor", (cx, cy, top_z + 0.002), (inner + wall, inner + wall, 0.002), rgba)
        for i, (dx, dy, sx, sy) in enumerate([(inner + wall / 2, 0, wall / 2, inner + wall), (-(inner + wall / 2), 0, wall / 2, inner + wall),
                                              (0, inner + wall / 2, inner, wall / 2), (0, -(inner + wall / 2), inner, wall / 2)]):
            _static_box(wb, f"fixture_bin_wall{i}", (cx + dx, cy + dy, top_z + h / 2), (sx, sy, h / 2), rgba)
        info.update(zone_center=[cx, cy, top_z], zone_half=[inner, inner], rim_z=top_z + h)
    elif fx == "bowl":
        cx, cy = at("bowl")
        inner, h = 0.05, 0.025
        rgba = (0.92, 0.92, 0.88, 1)
        _static_box(wb, "fixture_bowl_floor", (cx, cy, top_z + 0.002), (inner + 0.004, inner + 0.004, 0.002), rgba)
        _ring(wb, "fixture_bowl_wall", (cx, cy), inner, 0.004, h, top_z, rgba, n=12)
        info.update(zone_center=[cx, cy, top_z], zone_radius=inner, rim_z=top_z + h)
    elif fx == "nail_board":
        cx, cy = at("nail_board")
        board_h = 0.03
        _static_box(wb, "fixture_board", (cx, cy, top_z + board_h / 2), (0.04, 0.04, board_h / 2), (0.62, 0.48, 0.30, 1))
        nail = wb.add_body()
        nail.name = "fixture_nail"
        nail.pos = [cx, cy, top_z + board_h]
        j = nail.add_joint()
        j.name = "nail_slide"
        j.type = mujoco.mjtJoint.mjJNT_SLIDE
        j.axis = [0, 0, 1]
        j.range = [-0.03, 0.0]
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        j.damping = [30.0, 0.0, 0.0]
        j.frictionloss = 3.0
        for name, size, pos, collide in (("shaft", [0.003, 0.015, 0], [0, 0, 0.015], False), ("head", [0.009, 0.002, 0], [0, 0, 0.032], True)):
            g = nail.add_geom()
            g.name = f"fixture_nail_{name}"
            g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            g.size, g.pos = size, pos
            g.rgba = [0.75, 0.75, 0.8, 1]
            g.mass = 0.005
            if not collide:
                g.contype = g.conaffinity = 0
        info.update(nail_start_z=top_z + board_h)
    elif fx == "lock":
        cx, cy = at("lock")
        housing = (0.30, 0.30, 0.34, 1)
        plug_r_half = 0.012
        # Housing: base plate + four walls around the plug (bits 1: collides with key and hand, not the plug)
        _static_box(wb, "fixture_lock_base", (cx, cy, top_z + 0.004), (0.03, 0.03, 0.004), housing)
        for i, (dx, dy, sx, sy) in enumerate([(0.022, 0, 0.008, 0.03), (-0.022, 0, 0.008, 0.03), (0, 0.022, 0.014, 0.008), (0, -0.022, 0.014, 0.008)]):
            _static_box(wb, f"fixture_lock_wall{i}", (cx + dx, cy + dy, top_z + LOCK_H / 2), (sx, sy, LOCK_H / 2), housing)
        # Plug: hinged about z, two half-blocks with the keyhole slot between them
        plug = wb.add_body()
        plug.name = "fixture_lock_plug"
        plug.pos = [cx, cy, top_z + 0.008]
        j = plug.add_joint()
        j.name = "lock_plug_joint"
        j.type = mujoco.mjtJoint.mjJNT_HINGE
        j.axis = [0, 0, 1]
        # Hinge ranges follow the spec's angle unit: menagerie scenes compile in degrees, the Wuji MJCF in radians.
        j.range = [-10.0, 125.0] if spec.compiler.degree else [math.radians(-10), math.radians(125)]
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        # Light resistance: a 15 g key only produces ~1e-4 N·m through soft contacts, so friction must stay below that
        j.damping = [0.0005, 0.0, 0.0]
        j.frictionloss = 0.00005
        j.armature = 0.00002
        plug_h = LOCK_H - 0.008
        half_block = (plug_r_half - SLOT_HALF_GAP) / 2
        for side in (1, -1):
            g = plug.add_geom()
            g.name = f"fixture_lock_plug_{'a' if side > 0 else 'b'}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [plug_r_half, half_block, plug_h / 2]
            g.pos = [0, side * (SLOT_HALF_GAP + half_block), plug_h / 2]
            g.rgba = [0.80, 0.68, 0.25, 1]
            g.contype = g.conaffinity = PLUG_BITS
            g.mass = 0.01
            g.solref = [0.004, 1.0]  # stiff walls so a turning key blade drives the plug instead of sinking into it
            g.solimp = [0.95, 0.99, 0.0005, 0.5, 2.0]
        info.update(lock_center=[cx, cy], lock_top_z=top_z + LOCK_H)
    elif fx == "shape_board":
        cx, cy = at("shape_board")
        h, rgba = 0.03, (0.62, 0.48, 0.30, 1)
        round_c, square_c = (cx, cy - 0.032), (cx, cy + 0.032)
        _ring(wb, "fixture_round_hole", round_c, 0.0145, 0.012, h, top_z, rgba, n=8)
        hs = 0.0145
        for i, (dx, dy, sx, sy) in enumerate([(hs + 0.006, 0, 0.006, hs + 0.012), (-(hs + 0.006), 0, 0.006, hs + 0.012),
                                              (0, hs + 0.006, hs, 0.006), (0, -(hs + 0.006), hs, 0.006)]):
            _static_box(wb, f"fixture_square_hole{i}", (square_c[0] + dx, square_c[1] + dy, top_z + h / 2), (sx, sy, h / 2), rgba)
        _static_box(wb, "fixture_round_label", (round_c[0], round_c[1], top_z + 0.0005), (0.008, 0.008, 0.0005), (0.35, 0.45, 0.95, 1), collide=False)
        _static_box(wb, "fixture_square_label", (square_c[0], square_c[1], top_z + 0.0005), (0.008, 0.008, 0.0005), (0.92, 0.26, 0.21, 1), collide=False)
        holes = {"round": round_c, "square": square_c}
        info.update(hole_center=list(holes["square" if obj.id == "square_peg" else "round"]), board_top_z=top_z + h,
                    wrong_hole_center=list(holes["round" if obj.id == "square_peg" else "square"]))
    elif fx == "screw_board":
        cx, cy = at("screw_board")
        h = 0.025
        _static_box(wb, "fixture_screw_board", (cx, cy, top_z + h / 2), (0.04, 0.04, h / 2), (0.62, 0.48, 0.30, 1))
        screw = wb.add_body()
        screw.name = "fixture_screw"
        screw.pos = [cx, cy, top_z + h]
        g = screw.add_geom()
        g.name = "fixture_screw_head"
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [0.006, 0.0015, 0]
        g.pos = [0, 0, 0.0015]
        g.rgba = [0.8, 0.8, 0.85, 1]
        mark = screw.add_geom()
        mark.type = mujoco.mjtGeom.mjGEOM_BOX
        mark.size = [0.005, 0.0008, 0.0002]
        mark.pos = [0, 0, 0.0031]
        mark.rgba = [0.1, 0.1, 0.1, 1]
        mark.contype = mark.conaffinity = 0
        info.update(screw_head=[cx, cy, top_z + h + 0.003])
    elif fx == "cargo":
        cx, cy = at("cargo")
        b = wb.add_body()
        b.name = "cargo"
        b.pos = [cx, cy, top_z + 0.0151]
        j = b.add_joint()
        j.type = mujoco.mjtJoint.mjJNT_FREE
        g = b.add_geom()
        g.name = "cargo_geom"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.015, 0.015, 0.015]
        g.rgba = [0.2, 0.4, 0.95, 1]
        g.mass = 0.03
        info.update(cargo="cargo")
    elif fx == "ghost_table":
        yaw = p.get("yaw_deg", 90.0)
        target_r = Rotation.from_euler("z", yaw, degrees=True) * r0
        pos = [xy[0] + PUSH_OFFSET[0], xy[1] + PUSH_OFFSET[1], top_z + obj.half_extents[2] * scale + 0.001]
        g = wb.add_body()
        g.name = "fixture_ghost"
        g.pos = pos
        g.quat = _quat_wxyz(target_r)
        add_parts(g, obj.parts, obj.rgba, 0.0, scale, visual_only=True, name_prefix="fixture_ghost")
        info.update(target_pos=pos, target_quat=_quat_wxyz(target_r))
    elif fx == "ghost_air":
        axis = "xyz"[int(p.get("axis", 0))]
        target_r = Rotation.from_euler(axis, p.get("angle_deg", 90.0), degrees=True) * r0
        pos = [origin[0] + GHOST_AIR_POS[0], origin[1] + GHOST_AIR_POS[1], top_z + GHOST_AIR_POS[2]]
        g = wb.add_body()
        g.name = "fixture_ghost"
        g.pos = pos
        g.quat = _quat_wxyz(target_r)
        add_parts(g, obj.parts, obj.rgba, 0.0, scale, visual_only=True, name_prefix="fixture_ghost")
        info.update(target_quat=_quat_wxyz(target_r), axis=axis)
    if obj.id == "key":
        info["key_tip_local"] = [0.030 * scale, 0, 0]
    if obj.id == "power_screwdriver":
        info["bit_tip_local"] = [0.065 * scale, 0, 0]
    return info


def describe(bundle: SceneBundle) -> dict[str, Any]:
    """Plain-numbers description of the built scene (for the UI and the planner brief)."""
    m, d = bundle.model, bundle.data
    bid = m.body("target_object").id
    t = bundle.extras["task"]
    pos = d.xpos[bid]
    yaw = math.degrees(math.atan2(d.xmat[bid][3], d.xmat[bid][0]))
    out: dict[str, Any] = {
        "robot": bundle.extras["robot_id"],
        "object_position": [round(float(v), 4) for v in pos],
        "object_yaw_deg": round(yaw, 1),
        "object_mass_kg": round(float(m.body_subtreemass[bid]), 3),
        "table_top_z": round(t["table_top_z"], 4),
        "fixture": t.get("fixture"),
        "layout_seed": t.get("layout_seed"),
    }
    if t.get("layout_note"):
        out["layout_note"] = t["layout_note"]
    for key in ("zone_center", "zone_half", "zone_radius", "rim_z", "target_pos", "target_quat", "nail_start_z", "lock_center",
                "lock_top_z", "hole_center", "board_top_z", "screw_head"):
        if key in t:
            v = t[key]
            out[key] = [round(float(x), 4) for x in v] if isinstance(v, list) else round(float(v), 4)
    if "distractors" in t:
        out["distractors"] = t["distractors"]
    if t.get("extras"):  # multi-object scenes: every other free body, with its live position
        out["other_objects"] = [{**e, "position": [round(float(v), 4) for v in d.xpos[m.body(e["body"]).id]]} for e in t["extras"]]
    if "sort_pads" in t:
        out["sort_pads"] = {c: [round(float(v), 4) for v in xyz] for c, xyz in t["sort_pads"].items()}
        out["sort_pad_half"] = t["sort_pad_half"]
        out["target_colour"] = "red"
    return out


def object_thumbnail_scene(object_id: str) -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray]:
    """A tiny stand-alone scene (floor + object) for library thumbnails. Returns model, data, look-at point."""
    refresh_generated()
    obj = OBJECTS_BY_ID[object_id]
    spec = mujoco.MjSpec()
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    floor = spec.worldbody.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [5, 5, 0.1]
    floor.rgba = [0.16, 0.19, 0.24, 1]
    light = spec.worldbody.add_light()
    light.pos = [0.2, -0.3, 0.8]
    light.dir = [-0.2, 0.3, -0.8]
    light.castshadow = True
    z = obj.articulation.pivot_z if obj.articulation else obj.half_extents[2] + 0.001
    body = spec.worldbody.add_body()
    body.pos = [0, 0, z]
    if obj.articulation:
        add_parts(body, obj.articulation.base_parts, obj.articulation.base_rgba, 0.0, name_prefix=None)
    add_parts(body, obj.parts, obj.rgba, 0.0, name_prefix=None)
    for c in obj.children:
        cb = body.add_body()
        cb.pos = list(c.pos)
        add_parts(cb, c.parts, c.rgba, 0.0, name_prefix=None)
    if obj.special == "lookalike_cans":
        for dy in (0.06, -0.06):
            b = spec.worldbody.add_body()
            b.pos = [0, dy, z]
            add_parts(b, obj.parts, obj.rgba, 0.0, name_prefix=None)
    if obj.special == "beads":
        for i in range(6):
            b = spec.worldbody.add_body()
            b.pos = [0.008 * math.cos(i), 0.008 * math.sin(i), 0.03 + 0.004 * i]
            add_parts(b, (Part("sphere", (0.0065,)),), (0.95, 0.4, 0.2, 1), 0.0, name_prefix=None)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, np.array([0, 0, z * 0.7])


__all__ = ["ROBOTS", "ORIGIN", "build_task_scene", "check_combo", "describe", "object_thumbnail_scene", "resolve_params"]
