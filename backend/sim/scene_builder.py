"""Procedural MuJoCo scene builder for cross-embodiment manipulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MENAGERIE_DIR = REPO_ROOT / "assets" / "mujoco_menagerie"

RobotId = Literal["franka_emika_panda", "wonik_allegro"]
ObjectShape = Literal["box", "cylinder", "sphere"]


@dataclass
class ObjectSpec:
    shape: ObjectShape
    size: list[float]  # MuJoCo geom size (half-extents / radius, half-height)
    pos: list[float]
    quat: list[float]
    rgba: list[float]
    mass: float

    @property
    def half_height(self) -> float:
        if self.shape == "box":
            return self.size[2]
        if self.shape == "cylinder":
            return self.size[1]
        return self.size[0]


@dataclass
class SceneBundle:
    model: mujoco.MjModel
    data: mujoco.MjData
    robot_id: RobotId
    scene_seed: int
    obj_spec: ObjectSpec
    table_top_z: float
    disabled_joints: list[str] = field(default_factory=list)


def _yaw_to_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def generate_object_spec(
    seed: int,
    robot_id: RobotId,
    table_top_z: float,
) -> ObjectSpec:
    """Deterministically generate a graspable object from a seed."""
    rng = np.random.default_rng(seed)
    shapes: list[ObjectShape] = ["box", "cylinder", "sphere"]
    shape: ObjectShape = shapes[int(rng.integers(0, len(shapes)))]

    # High-contrast colors for clear visual diagnosis by Gemini 4
    palette = [
        [0.92, 0.26, 0.21, 1.0],  # bright red-orange
        [0.10, 0.68, 0.62, 1.0],  # teal
        [0.96, 0.65, 0.14, 1.0],  # amber
        [0.35, 0.45, 0.95, 1.0],  # cobalt blue
    ]
    rgba = palette[int(rng.integers(0, len(palette)))]

    if shape == "box":
        hx = float(rng.uniform(0.020, 0.032))
        hy = float(rng.uniform(0.020, 0.032))
        hz = float(rng.uniform(0.030, 0.050))
        size = [hx, hy, hz]
        half_h = hz
    elif shape == "cylinder":
        radius = float(rng.uniform(0.020, 0.032))
        half_h = float(rng.uniform(0.032, 0.052))
        size = [radius, half_h, 0.0]
    else:  # sphere
        radius = float(rng.uniform(0.024, 0.034))
        size = [radius, 0.0, 0.0]
        half_h = radius

    if robot_id == "franka_emika_panda":
        x = float(rng.uniform(0.42, 0.60))
        y = float(rng.uniform(-0.16, 0.16))
    else:
        x = float(rng.uniform(-0.04, 0.04))
        y = float(rng.uniform(-0.04, 0.04))

    z = table_top_z + half_h + 0.002
    yaw = float(rng.uniform(-math.pi, math.pi))
    mass = float(rng.uniform(0.08, 0.18))

    return ObjectSpec(
        shape=shape,
        size=size,
        pos=[x, y, z],
        quat=_yaw_to_quat(yaw),
        rgba=rgba,
        mass=mass,
    )


def _lookat_quat(
    eye: list[float],
    target: list[float],
    up: list[float] = [0.0, 0.0, 1.0],
) -> list[float]:
    """Compute MuJoCo camera quaternion [w, x, y, z] looking from eye to target."""
    eye_v = np.asarray(eye, dtype=np.float64)
    tgt_v = np.asarray(target, dtype=np.float64)
    up_v = np.asarray(up, dtype=np.float64)

    # MuJoCo camera looks along -Z, so +Z points from target to eye
    z_axis = eye_v - tgt_v
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up_v, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    rot = np.column_stack([x_axis, y_axis, z_axis])
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, rot.ravel())
    return quat.tolist()


def _add_table_and_object(
    spec: mujoco.MjSpec,
    table_center_xy: tuple[float, float],
    table_half_size: tuple[float, float, float],
    table_top_z: float,
    obj_spec: ObjectSpec,
) -> None:
    """Add a clean tabletop surface and the procedural target object to an MjSpec."""
    table_body = spec.worldbody.add_body()
    table_body.name = "work_table"
    table_body.pos = [
        table_center_xy[0],
        table_center_xy[1],
        table_top_z - table_half_size[2],
    ]
    table_geom = table_body.add_geom()
    table_geom.name = "table_top"
    table_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    table_geom.size = list(table_half_size)
    table_geom.rgba = [0.32, 0.35, 0.40, 1.0]
    table_geom.friction = [0.8, 0.005, 0.0001]

    # Add procedural target object with a freejoint
    obj_body = spec.worldbody.add_body()
    obj_body.name = "target_object"
    obj_body.pos = obj_spec.pos
    obj_body.quat = obj_spec.quat

    free_jnt = obj_body.add_joint()
    free_jnt.name = "target_object_freejoint"
    free_jnt.type = mujoco.mjtJoint.mjJNT_FREE

    obj_geom = obj_body.add_geom()
    obj_geom.name = "target_object_geom"
    if obj_spec.shape == "box":
        obj_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        obj_geom.size = obj_spec.size
    elif obj_spec.shape == "cylinder":
        obj_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        obj_geom.size = [obj_spec.size[0], obj_spec.size[1], 0.0]
    else:
        obj_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        obj_geom.size = [obj_spec.size[0], 0.0, 0.0]

    obj_geom.rgba = obj_spec.rgba
    obj_geom.mass = obj_spec.mass
    obj_geom.condim = 4
    obj_geom.friction = [1.2, 0.01, 0.0005]


def _add_cameras(spec: mujoco.MjSpec, robot_id: RobotId) -> None:
    """Add fixed side_cam and top_cam views tailored to each robot workspace."""
    if robot_id == "franka_emika_panda":
        side_cam = spec.worldbody.add_camera()
        side_cam.name = "side_cam"
        side_cam.pos = [1.05, -0.85, 0.58]
        side_cam.quat = _lookat_quat(side_cam.pos, [0.42, 0.0, 0.22])
        side_cam.fovy = 42.0

        top_cam = spec.worldbody.add_camera()
        top_cam.name = "top_cam"
        top_cam.pos = [0.55, 0.0, 1.35]
        top_cam.quat = _lookat_quat(top_cam.pos, [0.52, 0.0, 0.02], up=[1.0, 0.0, 0.0])
        top_cam.fovy = 42.0
    else:
        side_cam = spec.worldbody.add_camera()
        side_cam.name = "side_cam"
        side_cam.pos = [0.34, -0.34, 0.22]
        side_cam.quat = _lookat_quat(side_cam.pos, [0.0, 0.0, 0.02])
        side_cam.fovy = 42.0

        top_cam = spec.worldbody.add_camera()
        top_cam.name = "top_cam"
        top_cam.pos = [0.0, 0.0, 0.62]
        top_cam.quat = _lookat_quat(top_cam.pos, [0.0, 0.0, -0.05], up=[1.0, 0.0, 0.0])
        top_cam.fovy = 42.0


def build_scene(
    robot_id: RobotId = "franka_emika_panda",
    scene_seed: int = 42,
    disabled_joints: list[str] | None = None,
) -> SceneBundle:
    """Build a MuJoCo model & data with the requested robot and seeded object."""
    disabled_joints = list(disabled_joints or [])

    if robot_id == "franka_emika_panda":
        scene_xml = MENAGERIE_DIR / "franka_emika_panda" / "scene.xml"
        spec = mujoco.MjSpec.from_file(str(scene_xml))
        table_top_z = 0.02
        obj_spec = generate_object_spec(scene_seed, robot_id, table_top_z)

        # Add an end-effector site between the Panda fingertips for IK
        hand_body = spec.body("hand")
        if hand_body is not None:
            ee_site = hand_body.add_site()
            ee_site.name = "ee_site"
            ee_site.pos = [0.0, 0.0, 0.1034]
            ee_site.size = [0.008, 0.008, 0.008]
            ee_site.rgba = [0.1, 0.9, 0.1, 0.6]

        _add_table_and_object(
            spec=spec,
            table_center_xy=(0.52, 0.0),
            table_half_size=(0.28, 0.35, 0.01),
            table_top_z=table_top_z,
            obj_spec=obj_spec,
        )
        _add_cameras(spec, robot_id)

        model = spec.compile()
        data = mujoco.MjData(model)

        # Initialize Panda to its 'home' keyframe if available
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id >= 0:
            # Copy only robot joints/ctrl from keyframe so target_object stays at obj_spec.pos
            key_qpos = model.key_qpos[home_id]
            key_ctrl = model.key_ctrl[home_id]
            n_robot_q = min(9, model.nq)
            data.qpos[:n_robot_q] = key_qpos[:n_robot_q]
            data.ctrl[: len(key_ctrl)] = key_ctrl

    elif robot_id == "wonik_allegro":
        scene_xml = MENAGERIE_DIR / "wonik_allegro" / "scene_right.xml"
        spec = mujoco.MjSpec.from_file(str(scene_xml))

        # Remove the default menagerie ellipsoid 'object' body
        default_obj = spec.body("object")
        if default_obj is not None:
            spec.delete(default_obj)

        # Give Allegro palm a freejoint + welded mocap body so it can move over the table
        palm_body = spec.body("palm")
        if palm_body is not None:
            palm_body.pos = [-0.10, 0.0, 0.14]
            # Orient palm downward/forward toward the table workspace
            palm_body.quat = [0.7071, 0.0, 0.7071, 0.0]
            palm_free = palm_body.add_joint()
            palm_free.name = "palm_freejoint"
            palm_free.type = mujoco.mjtJoint.mjJNT_FREE

            palm_site = palm_body.add_site()
            palm_site.name = "ee_site"
            palm_site.pos = [0.0, 0.0, 0.0]
            palm_site.size = [0.008, 0.008, 0.008]
            palm_site.rgba = [0.1, 0.9, 0.1, 0.6]

            mocap_body = spec.worldbody.add_body()
            mocap_body.name = "palm_mocap"
            mocap_body.mocap = True
            mocap_body.pos = [-0.10, 0.0, 0.14]
            mocap_body.quat = [0.7071, 0.0, 0.7071, 0.0]

            weld = spec.add_equality()
            weld.type = mujoco.mjtEq.mjEQ_WELD
            weld.name1 = "palm_mocap"
            weld.name2 = "palm"
            weld.objtype = mujoco.mjtObj.mjOBJ_BODY
            weld.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
            weld.solref = [0.01, 1.0]

        table_top_z = -0.08
        obj_spec = generate_object_spec(scene_seed, robot_id, table_top_z)

        _add_table_and_object(
            spec=spec,
            table_center_xy=(0.0, 0.0),
            table_half_size=(0.20, 0.20, 0.01),
            table_top_z=table_top_z,
            obj_spec=obj_spec,
        )
        _add_cameras(spec, robot_id)

        model = spec.compile()
        data = mujoco.MjData(model)
    else:
        raise ValueError(f"Unsupported robot_id: {robot_id}")

    # Apply live fault injection if requested: lock disabled joints at their initial pose
    mujoco.mj_forward(model, data)
    for jname in disabled_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            qadr = model.jnt_qposadr[jid]
            q0 = float(data.qpos[qadr])
            model.jnt_limited[jid] = 1
            model.jnt_range[jid, 0] = q0 - 1e-4
            model.jnt_range[jid, 1] = q0 + 1e-4
            for aid in range(model.nu):
                if model.actuator_trnid[aid, 0] == jid:
                    model.actuator_gainprm[aid, :] = 0.0
                    model.actuator_biasprm[aid, :] = 0.0

    mujoco.mj_forward(model, data)

    return SceneBundle(
        model=model,
        data=data,
        robot_id=robot_id,
        scene_seed=scene_seed,
        obj_spec=obj_spec,
        table_top_z=table_top_z,
        disabled_joints=disabled_joints,
    )


def summarize_kinematics(bundle: SceneBundle) -> str:
    """Return a concise text summary of the robot joints, actuators, and object."""
    m = bundle.model
    d = bundle.data
    lines = [
        f"Robot: {bundle.robot_id} (seed={bundle.scene_seed})",
        f"Disabled joints: {bundle.disabled_joints or 'none'}",
        f"Table top z: {bundle.table_top_z:.3f} m",
    ]
    obj_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    if obj_id >= 0:
        pos = d.xpos[obj_id]
        lines.append(
            f"Target object: shape={bundle.obj_spec.shape}, "
            f"size={[round(v, 4) for v in bundle.obj_spec.size]}, "
            f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], "
            f"mass={bundle.obj_spec.mass:.3f} kg"
        )

    lines.append(f"Actuators ({m.nu}):")
    for aid in range(m.nu):
        aname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or f"act_{aid}"
        lo, hi = m.actuator_ctrlrange[aid]
        lines.append(f"  - {aname}: ctrlrange=[{lo:.3f}, {hi:.3f}]")

    return "\n".join(lines)
