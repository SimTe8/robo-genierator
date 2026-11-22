"""Damped least-squares IK executor, slow-motion renderer, and ground-truth verifier."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio.v3 as iio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from schemas import Waypoint
from sim.scene_builder import REPO_ROOT, SceneBundle


@dataclass
class SimState:
    """Snapshot of MuJoCo physics state for checkpointed closed-loop planning."""

    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    cum_yaw_deg: float = 0.0
    obj_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


@dataclass
class RolloutResult:
    sim_success: bool
    telemetry_summary: str
    video_path: Path | None
    metrics: dict[str, float | bool | int | str]
    trajectory: dict[str, np.ndarray] = field(default_factory=dict)
    end_state: SimState | None = None
    frames: list[np.ndarray] = field(default_factory=list)


def _rpy_to_mat(rpy: Sequence[float]) -> np.ndarray:
    return Rotation.from_euler("xyz", rpy, degrees=False).as_matrix()


def _rpy_to_quat_wxyz(rpy: Sequence[float]) -> np.ndarray:
    xyzw = Rotation.from_euler("xyz", rpy, degrees=False).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def _orientation_error(current_mat: np.ndarray, target_mat: np.ndarray) -> np.ndarray:
    """Compute 3D axis-angle orientation error vector in world frame."""
    r_err = target_mat @ current_mat.T
    rotvec = Rotation.from_matrix(r_err).as_rotvec()
    return rotvec


def _label_frame(frame: np.ndarray, text: str, badge_color: tuple[int, int, int] = (15, 20, 28)) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (8, 8), (8 + len(text) * 9 + 18, 34), badge_color, -1)
    cv2.putText(
        out,
        text,
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (240, 245, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def _solve_panda_ik(
    model: mujoco.MjModel,
    qpos_init: np.ndarray,
    site_id: int,
    target_pos: np.ndarray,
    target_mat: np.ndarray,
    disabled_dof_mask: np.ndarray,
    max_iters: int = 80,
    damping: float = 0.03,
    ori_weight: float = 0.45,
) -> np.ndarray:
    """Solve damped least-squares IK on a scratch MjData to reach target_pos & target_mat."""
    ik_data = mujoco.MjData(model)
    ik_data.qpos[:] = qpos_init
    q_locked = qpos_init[:7].copy()

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    home_q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

    for _ in range(max_iters):
        mujoco.mj_kinematics(model, ik_data)
        mujoco.mj_comPos(model, ik_data)

        cur_pos = ik_data.site_xpos[site_id]
        cur_mat = ik_data.site_xmat[site_id].reshape(3, 3)

        pos_err = target_pos - cur_pos
        ori_err = _orientation_error(cur_mat, target_mat)
        if np.linalg.norm(pos_err) < 1.5e-3 and np.linalg.norm(ori_err) < 0.03:
            break

        mujoco.mj_jacSite(model, ik_data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, :7], jacr[:, :7] * ori_weight])

        for idx in range(7):
            if disabled_dof_mask[idx]:
                J[:, idx] = 0.0

        err = np.concatenate([pos_err, ori_err * ori_weight])
        jjt = J @ J.T + (damping**2) * np.eye(6)
        dq = J.T @ np.linalg.solve(jjt, err)

        # Gentle nullspace pull toward home posture
        null_proj = np.eye(7) - J.T @ np.linalg.solve(jjt, J)
        dq_null = 0.03 * (null_proj @ (home_q - ik_data.qpos[:7]))
        dq_null[disabled_dof_mask[:7]] = 0.0
        dq = dq + dq_null

        step_norm = np.linalg.norm(dq)
        if step_norm > 0.25:
            dq *= 0.25 / step_norm

        q_next = ik_data.qpos[:7] + dq
        for i in range(7):
            if disabled_dof_mask[i]:
                q_next[i] = q_locked[i]
            else:
                lo, hi = model.actuator_ctrlrange[i]
                q_next[i] = np.clip(q_next[i], lo, hi)
        ik_data.qpos[:7] = q_next

    return ik_data.qpos[:7].copy()


def _set_allegro_fingers(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gripper_target: float,
    finger_targets: dict[str, float] | None,
    disabled_joints: set[str],
) -> None:
    """Map normalized closure [0=open, 1=closed] to Allegro (16-DOF) or Wuji (20-DOF) finger actuators."""
    # Allegro prefixes: ff (index), mf (middle), rf (ring), th (thumb)
    # Wuji prefixes: right_finger1 (thumb), right_finger2 (index), right_finger3 (middle), right_finger4 (ring), right_finger5 (little)
    prefix_map = {
        "thumb": ("th", "right_finger1"),
        "index": ("ff", "right_finger2"),
        "middle": ("mf", "right_finger3"),
        "ring": ("rf", "right_finger4"),
        "little": ("right_finger5",),
    }
    defaults = {k: gripper_target for k in prefix_map}
    if finger_targets:
        for k, v in finger_targets.items():
            if k in defaults:
                defaults[k] = float(np.clip(v, 0.0, 1.0))

    for aid in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
        jid = model.actuator_trnid[aid, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if jname in disabled_joints or aname in disabled_joints:
            continue

        lo, hi = model.actuator_ctrlrange[aid]
        closure = gripper_target
        for fname, prefs in prefix_map.items():
            if any(aname.startswith(p) for p in prefs):
                closure = defaults[fname]
                break

        # Wuji hand: thumb joint1 is CMC opposition (keep opposed downward even when open!),
        # finger2-5 joint2 is abduction (keep 0.0), remaining joints flex to pinch/cage
        if aname.startswith("right_finger"):
            if aname == "right_finger1_joint1_actuator":
                data.ctrl[aid] = lo + (0.85 + 0.10 * closure) * (hi - lo)
            elif aname == "right_finger1_joint2_actuator":
                data.ctrl[aid] = lo + (0.12 + 0.65 * closure) * (hi - lo)
            elif aname.startswith("right_finger1_"):
                data.ctrl[aid] = lo + (0.04 + 0.62 * closure) * (hi - lo)
            elif "joint2" in aname:
                data.ctrl[aid] = 0.0
            else:
                data.ctrl[aid] = lo + (hi - lo) * (0.05 + 0.78 * closure)
        # Allegro hand: a0 is abduction, a1/a2/a3 flex
        elif aname.endswith("0"):
            if aname.startswith("th"):
                data.ctrl[aid] = lo + 0.75 * (hi - lo) * closure
            else:
                data.ctrl[aid] = 0.0
        else:
            data.ctrl[aid] = lo + (hi - lo) * (0.10 + 0.75 * closure)


def _find_palm_body_id(model: mujoco.MjModel) -> int:
    for name in ("palm", "right_palm_link", "hand"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            return bid
    return -1


def _read_ee_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_site_id: int,
    palm_body_id: int,
) -> np.ndarray:
    """Return [x, y, z, qw, qx, qy, qz] of ee_site if present, else palm body."""
    if ee_site_id >= 0:
        pos = data.site_xpos[ee_site_id]
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, data.site_xmat[ee_site_id])
        return np.concatenate([pos, quat]).astype(np.float32)
    if palm_body_id >= 0:
        pos = data.xpos[palm_body_id]
        quat = data.xquat[palm_body_id]
        return np.concatenate([pos, quat]).astype(np.float32)
    return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _pack_trajectory(
    bundle: SceneBundle,
    obs_qpos: list[np.ndarray],
    obs_qvel: list[np.ndarray],
    priv_obj_pos: list[np.ndarray],
    actions_ctrl: list[np.ndarray],
    ee_poses: list[np.ndarray],
    gripper_states: list[list[float]],
    timestamps_s: list[float],
) -> dict[str, np.ndarray]:
    """Pack backwards-compatible + researcher-grade trajectory arrays."""
    m = bundle.model
    t_len = len(obs_qpos)
    joint_names = np.asarray(
        [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"j_{j}" for j in range(m.njnt)]
    )
    actuator_names = np.asarray(
        [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"a_{a}" for a in range(m.nu)]
    )
    return {
        "observation_qpos": np.asarray(obs_qpos, dtype=np.float32).reshape(t_len, m.nq),
        "observation_qvel": np.asarray(obs_qvel, dtype=np.float32).reshape(t_len, m.nv),
        "privileged_obj_pos": np.asarray(priv_obj_pos, dtype=np.float32).reshape(t_len, 3),
        "action_ctrl": np.asarray(actions_ctrl, dtype=np.float32).reshape(t_len, m.nu),
        "ee_pose": np.asarray(ee_poses, dtype=np.float32).reshape(t_len, 7),
        "gripper_state": np.asarray(gripper_states, dtype=np.float32).reshape(t_len, 1),
        "timestamps_s": np.asarray(timestamps_s, dtype=np.float32).reshape(t_len),
        "dt": np.float32(20.0 * m.opt.timestep),
        "robot_id": np.asarray(bundle.robot_id),
        "joint_names": joint_names,
        "actuator_names": actuator_names,
    }


def _execute_in_hand_rotation(
    bundle: SceneBundle,
    waypoints: Sequence[Waypoint],
    video_path: Path | None = None,
    width: int = 384,
    height: int = 288,
    fps: int = 15,
    render_video: bool = True,
    target_rotation_deg: float = 60.0,
    initial_state: SimState | None = None,
    gait_cycles: int | None = None,
    total_goal_deg: float | None = None,
) -> RolloutResult:
    """Execute a cyclic 4-phase dexterous finger gait for Allegro in-hand cube rotation."""
    m, d = bundle.model, bundle.data
    disabled_set = set(bundle.disabled_joints)

    if initial_state is not None:
        d.qpos[:] = initial_state.qpos
        d.qvel[:] = initial_state.qvel
        d.ctrl[:] = initial_state.ctrl
        mujoco.mj_forward(m, d)
        start_cum_yaw = float(initial_state.cum_yaw_deg)
    else:
        # Settle cube gently into the cupped palm
        for _ in range(120):
            mujoco.mj_step(m, d)
        start_cum_yaw = 0.0

    obj_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    obj_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom")
    spindle_jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "target_object_spindle")
    is_spindle = spindle_jnt_id >= 0
    if is_spindle:
        obj_dof_adr = m.jnt_dofadr[spindle_jnt_id]
        spindle_qadr = m.jnt_qposadr[spindle_jnt_id]
        start_spindle_rad = float(d.qpos[spindle_qadr])
    else:
        obj_jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "target_object_freejoint")
        obj_dof_adr = m.jnt_dofadr[obj_jnt_id]
        spindle_qadr = -1
        start_spindle_rad = 0.0
    floor_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    ee_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    palm_body_id = _find_palm_body_id(m)

    palm_center_xy = np.array([0.015, 0.0], dtype=np.float64)
    init_pos = d.xpos[obj_body_id].copy()

    # Map actuator names to IDs for clean gait control
    act_ids = {
        (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""): aid
        for aid in range(m.nu)
    }

    def _waypoint_to_allegro_ctrl(wp: Waypoint) -> np.ndarray:
        ctrl = d.ctrl.copy()
        ft = wp.finger_targets or {}
        curl = {
            "ff": float(np.clip(ft.get("index", wp.gripper_target), 0.0, 1.0)),
            "mf": float(np.clip(ft.get("middle", wp.gripper_target), 0.0, 1.0)),
            "rf": float(np.clip(ft.get("ring", wp.gripper_target), 0.0, 1.0)),
            "th": float(np.clip(ft.get("thumb", wp.gripper_target), 0.0, 1.0)),
        }
        sweeps = {
            "ff": float(np.clip(wp.orientation_rpy[0], -0.47, 0.47)),
            "mf": float(np.clip(wp.orientation_rpy[1], -0.47, 0.47)),
            "rf": float(np.clip(wp.orientation_rpy[2], -0.47, 0.47)),
        }
        th_opp = float(np.clip(wp.position_offset[0], 0.263, 1.396))
        th_swp = float(np.clip(wp.position_offset[1], -0.105, 1.163))

        for aname, aid in act_ids.items():
            jid = m.actuator_trnid[aid, 0]
            jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            if jname in disabled_set or aname in disabled_set:
                continue
            lo, hi = m.actuator_ctrlrange[aid]
            pref = aname[:2]
            if pref in ("ff", "mf", "rf"):
                if aname.endswith("0"):
                    ctrl[aid] = np.clip(sweeps[pref], lo, hi)
                elif aname.endswith("1"):
                    # Keep knuckle raised enough to cage cube (0.35..1.15 rad) instead of opening flat
                    ctrl[aid] = np.clip(0.35 + 0.85 * curl[pref], lo, hi)
                elif aname.endswith("2"):
                    # Medial joint curves fingertips onto cube side faces (0.40..1.30 rad)
                    ctrl[aid] = np.clip(0.40 + 0.90 * curl[pref], lo, hi)
                else:
                    # Distal tip stays gently angled for fingertip tangential contact (0.25..0.95 rad)
                    ctrl[aid] = np.clip(0.25 + 0.70 * curl[pref], lo, hi)
            elif pref == "th":
                if aname == "tha0":
                    ctrl[aid] = np.clip(th_opp, lo, hi)
                elif aname == "tha1":
                    ctrl[aid] = np.clip(th_swp, lo, hi)
                else:
                    ctrl[aid] = np.clip(0.30 + 0.85 * curl["th"], lo, hi)
        return ctrl

    cycles = (
        gait_cycles
        if gait_cycles is not None
        else max(3, min(8, int(round(target_rotation_deg / 25.0))))
    )
    wp_steps = [
        max(1, int(round(float(np.clip(wp.duration_s, 0.2, 2.0)) / m.opt.timestep)))
        for wp in waypoints
    ]
    total_sim_steps = cycles * sum(wp_steps)
    # Keep each rendered clip compact (~60 frames = 4s at 15fps)
    render_every = max(1, total_sim_steps // 60)

    renderer = mujoco.Renderer(m, height=height, width=width) if render_video else None
    frames: list[np.ndarray] = []
    obs_qpos: list[np.ndarray] = []
    obs_qvel: list[np.ndarray] = []
    priv_obj_pos: list[np.ndarray] = []
    actions_ctrl: list[np.ndarray] = []
    ee_poses: list[np.ndarray] = []
    gripper_states: list[list[float]] = []
    timestamps_s: list[float] = []

    chunk_yaw_deg = 0.0
    max_abs_chunk_yaw = 0.0
    obj_contact_steps = 0
    dropped = False
    global_step = 0
    display_goal = total_goal_deg if total_goal_deg is not None else target_rotation_deg

    try:
        for cycle in range(cycles):
            for idx, (wp, n_steps) in enumerate(zip(waypoints, wp_steps)):
                start_ctrl = d.ctrl.copy()
                goal_ctrl = _waypoint_to_allegro_ctrl(wp)

                for step in range(n_steps):
                    frac = min(1.0, (step + 1) / (0.8 * n_steps))
                    d.ctrl[:] = (1.0 - frac) * start_ctrl + frac * goal_ctrl
                    mujoco.mj_step(m, d)
                    global_step += 1

                    cur_pos = d.xpos[obj_body_id]
                    if is_spindle:
                        chunk_yaw_deg = float(np.degrees(float(d.qpos[spindle_qadr]) - start_spindle_rad))
                        if abs(chunk_yaw_deg) > max_abs_chunk_yaw:
                            max_abs_chunk_yaw = abs(chunk_yaw_deg)
                    else:
                        if float(cur_pos[2]) < -0.02:
                            dropped = True
                        if not dropped:
                            wz = float(d.qvel[obj_dof_adr + 5])
                            chunk_yaw_deg += float(np.degrees(wz * m.opt.timestep))
                            if abs(chunk_yaw_deg) > max_abs_chunk_yaw:
                                max_abs_chunk_yaw = abs(chunk_yaw_deg)

                    for c_idx in range(d.ncon):
                        con = d.contact[c_idx]
                        if obj_geom_id in (con.geom1, con.geom2) and floor_geom_id not in (con.geom1, con.geom2):
                            obj_contact_steps += 1
                            break

                    if global_step % 20 == 0:
                        obs_qpos.append(d.qpos.copy())
                        obs_qvel.append(d.qvel.copy())
                        priv_obj_pos.append(cur_pos.copy())
                        actions_ctrl.append(d.ctrl.copy())
                        ee_poses.append(_read_ee_pose(m, d, ee_site_id, palm_body_id))
                        gripper_states.append([float(wp.gripper_target)])
                        timestamps_s.append(float(d.time))

                    if renderer is not None and global_step % render_every == 0:
                        cum_now = start_cum_yaw + chunk_yaw_deg
                        status = (
                            "DROPPED!"
                            if dropped
                            else f"YAW={cum_now:+.0f}/{display_goal:.0f}deg (d={chunk_yaw_deg:+.0f})"
                        )
                        badge = (140, 25, 25) if dropped else (20, 90, 45)
                        renderer.update_scene(d, camera="side_cam")
                        side_img = _label_frame(
                            renderer.render(),
                            f"CYC {cycle + 1}/{cycles} S{idx + 1} | {status}",
                            badge_color=badge,
                        )
                        renderer.update_scene(d, camera="top_cam")
                        drift_cm = float(np.linalg.norm(cur_pos[:2] - palm_center_xy)) * 100.0
                        top_img = _label_frame(
                            renderer.render(),
                            f"TOP | drift={drift_cm:.1f}cm",
                        )
                        frames.append(np.concatenate([side_img, top_img], axis=1))

            if dropped:
                break

        if renderer is not None and video_path is not None and frames:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(
                video_path,
                np.stack(frames, axis=0),
                fps=fps,
                codec="libx264",
                pixelformat="yuv420p",
                output_params=["-profile:v", "baseline", "-level", "3.0", "-movflags", "+faststart"],
            )
    finally:
        if renderer is not None:
            renderer.close()

    final_pos = d.xpos[obj_body_id].copy()
    if (not is_spindle) and float(final_pos[2]) < -0.02:
        dropped = True

    final_chunk_rot = abs(chunk_yaw_deg)
    cum_total_yaw = start_cum_yaw + chunk_yaw_deg
    drift_xy = final_pos[:2] - palm_center_xy
    drift_cm = float(np.linalg.norm(drift_xy)) * 100.0

    # Keep consistent spin direction when continuing from a prior checkpoint
    same_dir_progress = (
        chunk_yaw_deg * (1.0 if start_cum_yaw >= 0 else -1.0)
        if abs(start_cum_yaw) > 15.0
        else final_chunk_rot
    )
    sim_success = bool((not dropped) and (same_dir_progress >= target_rotation_deg))

    end_state = SimState(
        qpos=d.qpos.copy(),
        qvel=d.qvel.copy(),
        ctrl=d.ctrl.copy(),
        cum_yaw_deg=cum_total_yaw,
        obj_pos=final_pos.copy(),
    )

    metrics: dict[str, float | bool | int | str] = {
        "sim_success": sim_success,
        "chunk_yaw_deg": round(chunk_yaw_deg, 1),
        "cum_yaw_deg": round(cum_total_yaw, 1),
        "final_rot_deg": round(final_chunk_rot, 1),
        "max_rot_deg": round(max_abs_chunk_yaw, 1),
        "target_rotation_deg": target_rotation_deg,
        "gait_cycles": cycles,
        "dropped": dropped,
        "final_obj_z_cm": round(float(final_pos[2]) * 100.0, 2),
        "palm_drift_cm": round(drift_cm, 2),
        "palm_drift_dx_cm": round(float(drift_xy[0]) * 100.0, 2),
        "palm_drift_dy_cm": round(float(drift_xy[1]) * 100.0, 2),
        "obj_contact_steps": obj_contact_steps,
        "video_frames": len(frames),
    }

    telemetry_summary = (
        f"sim_success={sim_success} | "
        f"chunk_yaw={chunk_yaw_deg:+.1f}deg (cum={cum_total_yaw:+.1f}deg, chunk_target>={target_rotation_deg:.0f}deg, cycles={cycles}) | "
        f"dropped={dropped} | final_z={final_pos[2] * 100:.1f}cm | "
        f"palm_drift=({drift_xy[0] * 100:+.1f}cm X, {drift_xy[1] * 100:+.1f}cm Y, total={drift_cm:.1f}cm) | "
        f"obj_contacts={obj_contact_steps}"
    )

    trajectory = _pack_trajectory(
        bundle=bundle,
        obs_qpos=obs_qpos,
        obs_qvel=obs_qvel,
        priv_obj_pos=priv_obj_pos,
        actions_ctrl=actions_ctrl,
        ee_poses=ee_poses,
        gripper_states=gripper_states,
        timestamps_s=timestamps_s,
    )

    return RolloutResult(
        sim_success=sim_success,
        telemetry_summary=telemetry_summary,
        video_path=video_path if (render_video and video_path is not None) else None,
        metrics=metrics,
        trajectory=trajectory,
        end_state=end_state,
        frames=frames,
    )


def execute_rollout(
    bundle: SceneBundle,
    waypoints: Sequence[Waypoint],
    video_path: Path | None = None,
    width: int = 384,
    height: int = 288,
    fps: int = 15,
    render_video: bool = True,
    target_rotation_deg: float = 60.0,
    initial_state: SimState | None = None,
    gait_cycles: int | None = None,
    total_goal_deg: float | None = None,
) -> RolloutResult:
    """Execute waypoints in MuJoCo, render slow-mo multi-view MP4, and verify success."""
    if bundle.task_type in ("in_hand_rotation", "in_hand_spindle"):
        return _execute_in_hand_rotation(
            bundle=bundle,
            waypoints=waypoints,
            video_path=video_path,
            width=width,
            height=height,
            fps=fps,
            render_video=render_video,
            target_rotation_deg=target_rotation_deg,
            initial_state=initial_state,
            gait_cycles=gait_cycles,
            total_goal_deg=total_goal_deg,
        )
    m, d = bundle.model, bundle.data
    robot_id = bundle.robot_id
    disabled_set = set(bundle.disabled_joints)

    # Settle object briefly before reading its reference position
    for _ in range(100):
        mujoco.mj_step(m, d)

    if "task" in bundle.extras and "object_id" in bundle.extras:
        from sim.check import snapshot as snap_library_task
        bundle.extras["_before_snapshot"] = snap_library_task(bundle)

    obj_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    obj_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom")
    table_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    obs_roof_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_roof")
    obs_post_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_post")
    obstacle_geom_ids = {gid for gid in (obs_roof_id, obs_post_id) if gid >= 0}
    ee_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    palm_body_id = _find_palm_body_id(m)
    mocap_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm_mocap")
    mocap_id = m.body_mocapid[mocap_body_id] if mocap_body_id >= 0 else -1

    initial_obj_pos = d.xpos[obj_body_id].copy()
    initial_obj_z = float(initial_obj_pos[2])

    # MimicGen-style multi-frame anchor lookup (built right after initial settle)
    anchor_frames: dict[str, np.ndarray] = {}
    for bid in range(m.nbody):
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname:
            anchor_frames[bname] = d.xpos[bid].copy()
    anchor_frames["target_object"] = initial_obj_pos.copy()
    anchor_frames["world"] = np.zeros(3, dtype=np.float64)

    task_info = bundle.extras.get("task", {})
    if isinstance(task_info, dict):
        if "zone_center" in task_info:
            zc = np.asarray(task_info["zone_center"], dtype=np.float64)
            for alias in ("target_pad", "pad", "zone", "bin", "bowl"):
                anchor_frames[alias] = zc.copy()
        if "lock_center" in task_info and "lock_top_z" in task_info:
            lc = task_info["lock_center"]
            anchor_frames["lock"] = np.array(
                [lc[0], lc[1], task_info["lock_top_z"]], dtype=np.float64
            )
        if "hole_center" in task_info and "board_top_z" in task_info:
            hc = task_info["hole_center"]
            anchor_frames["hole"] = np.array(
                [hc[0], hc[1], task_info["board_top_z"]], dtype=np.float64
            )
        if "sort_pads" in task_info and isinstance(task_info["sort_pads"], dict):
            sort_pads = task_info["sort_pads"]
            if "red" in sort_pads:
                anchor_frames["red_pad"] = np.asarray(sort_pads["red"], dtype=np.float64)
            if "blue" in sort_pads:
                anchor_frames["blue_pad"] = np.asarray(sort_pads["blue"], dtype=np.float64)
        if "screw_head" in task_info:
            anchor_frames["screw"] = np.asarray(task_info["screw_head"], dtype=np.float64)

    # Build disabled DOF mask for Panda
    disabled_dof_mask = np.zeros(m.nv, dtype=bool)
    for jname in disabled_set:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            disabled_dof_mask[m.jnt_dofadr[jid]] = True

    # Phase durations (sim seconds) and slow-motion playback factors
    # Slower playback around grasp & close gives ~1fps video models 3-4 clear frames of jaw closing & lift-off
    phase_config = {
        "pre_grasp": {"duration": 0.8, "slowmo": 1.0},
        "grasp": {"duration": 0.8, "slowmo": 0.35},
        "close": {"duration": 0.65, "slowmo": 0.25},
        "lift": {"duration": 1.1, "slowmo": 0.60},
    }

    renderer = mujoco.Renderer(m, height=height, width=width) if render_video else None
    frames: list[np.ndarray] = []

    # Telemetry & trajectory buffers
    max_obj_z = initial_obj_z
    ee_min_dist = float("inf")
    obj_contact_steps = 0
    table_collision_steps = 0
    obstacle_collision_steps = 0
    lift_held_steps = 0
    lift_threshold = 0.08  # 8 cm above initial resting height

    obs_qpos: list[np.ndarray] = []
    obs_qvel: list[np.ndarray] = []
    priv_obj_pos: list[np.ndarray] = []
    actions_ctrl: list[np.ndarray] = []
    ee_poses: list[np.ndarray] = []
    gripper_states: list[list[float]] = []
    timestamps_s: list[float] = []

    try:
        num_wps = len(waypoints)
        ee_local_offset = (
            m.site_pos[ee_site_id].copy()
            if (mocap_id >= 0 and ee_site_id >= 0)
            else np.zeros(3, dtype=np.float64)
        )
        for wp_idx, wp in enumerate(waypoints):
            cfg = phase_config.get(wp.phase, {"duration": 0.8, "slowmo": 0.6})
            dur_s = float(np.clip(wp.duration_s, 0.4, 1.6)) if wp.duration_s else cfg["duration"]
            if wp.phase == "lift":
                dur_s = max(dur_s, 0.85)
            is_final_release = (wp_idx == num_wps - 1) and (num_wps >= 5) and (wp.gripper_target <= 0.20)
            if is_final_release:
                dur_s = max(dur_s, 0.80)
            n_steps = max(1, int(round(dur_s / m.opt.timestep)))
            sim_dt_per_frame = (1.0 / fps) * cfg["slowmo"]
            render_every = max(1, int(round(sim_dt_per_frame / m.opt.timestep)))

            # Multi-frame target pose anchored to wp.reference_frame (default: "target_object")
            ref_name = getattr(wp, "reference_frame", "target_object") or "target_object"
            anchor_pos = anchor_frames.get(ref_name, initial_obj_pos)
            target_pos = anchor_pos + np.asarray(wp.position_offset, dtype=np.float64)
            target_mat = _rpy_to_mat(wp.orientation_rpy)
            target_quat = _rpy_to_quat_wxyz(wp.orientation_rpy)
            retract_pos = target_pos + np.array([0.0, 0.0, 0.06], dtype=np.float64)

            # Pre-solve K=4 Cartesian sub-waypoints for Panda (linear XYZ + geodesic SLERP orientation)
            q_knots: list[np.ndarray] | None = None
            q_retract: np.ndarray | None = None
            n_sub = 4
            if robot_id == "franka_emika_panda":
                mujoco.mj_kinematics(m, d)
                q_start = d.ctrl[:7].copy()
                cur_ee_pos = d.site_xpos[ee_site_id].copy()
                cur_ee_mat = d.site_xmat[ee_site_id].reshape(3, 3).copy()
                delta_rotvec = Rotation.from_matrix(target_mat @ cur_ee_mat.T).as_rotvec()

                q_warm = d.qpos.copy()
                q_knots = [q_start]
                for k in range(1, n_sub + 1):
                    alpha = k / float(n_sub)
                    if k < n_sub:
                        sub_pos = (1.0 - alpha) * cur_ee_pos + alpha * target_pos
                        sub_mat = Rotation.from_rotvec(alpha * delta_rotvec).as_matrix() @ cur_ee_mat
                        iters = 35
                    else:
                        sub_pos = target_pos
                        sub_mat = target_mat
                        iters = 80
                    q_sol = _solve_panda_ik(
                        m,
                        q_warm,
                        ee_site_id,
                        sub_pos,
                        sub_mat,
                        disabled_dof_mask,
                        max_iters=iters,
                    )
                    q_knots.append(q_sol)
                    q_warm[:7] = q_sol
                if is_final_release:
                    q_retract = _solve_panda_ik(
                        m,
                        q_warm,
                        ee_site_id,
                        retract_pos,
                        target_mat,
                        disabled_dof_mask,
                        max_iters=50,
                    )

            # For Allegro/Wuji mocap smooth interpolation (shift mocap so ee_site TCP lands on target_pos)
            mocap_goal_pos = target_pos - target_mat @ ee_local_offset
            mocap_retract_pos = retract_pos - target_mat @ ee_local_offset
            start_mocap_pos = d.mocap_pos[mocap_id].copy() if mocap_id >= 0 else None
            start_mocap_quat = d.mocap_quat[mocap_id].copy() if mocap_id >= 0 else None

            for step in range(n_steps):
                if is_final_release:
                    # First 60%: move to release target & open jaws cleanly; last 40%: lift open jaws +6cm clear
                    split = int(0.60 * n_steps)
                    if step < split:
                        frac = min(1.0, (step + 1) / max(1.0, 0.45 * n_steps))
                        ret_frac = 0.0
                    else:
                        frac = 1.0
                        ret_frac = min(1.0, (step - split + 1) / max(1.0, 0.35 * n_steps))
                else:
                    # Smoothly ramp over first 75% of phase, hold last 25% for settling
                    frac = min(1.0, (step + 1) / (0.75 * n_steps))
                    ret_frac = 0.0

                if robot_id == "franka_emika_panda" and q_knots is not None:
                    if ret_frac > 0.0 and q_retract is not None:
                        d.ctrl[:7] = (1.0 - ret_frac) * q_knots[-1] + ret_frac * q_retract
                    else:
                        scaled = frac * n_sub
                        seg = min(n_sub - 1, int(scaled))
                        local_frac = scaled - seg
                        d.ctrl[:7] = (1.0 - local_frac) * q_knots[seg] + local_frac * q_knots[seg + 1]
                    # Panda actuator8: 255 = open (0.04m), 0 = closed (0.0m)
                    d.ctrl[7] = float(np.clip((1.0 - wp.gripper_target) * 255.0, 0.0, 255.0))
                else:
                    if mocap_id >= 0 and start_mocap_pos is not None and start_mocap_quat is not None:
                        if ret_frac > 0.0:
                            d.mocap_pos[mocap_id] = (1.0 - ret_frac) * mocap_goal_pos + ret_frac * mocap_retract_pos
                        else:
                            d.mocap_pos[mocap_id] = (1.0 - frac) * start_mocap_pos + frac * mocap_goal_pos
                        q_interp = (1.0 - frac) * start_mocap_quat + frac * target_quat
                        d.mocap_quat[mocap_id] = q_interp / (np.linalg.norm(q_interp) + 1e-8)
                    _set_allegro_fingers(m, d, wp.gripper_target, wp.finger_targets, disabled_set)

                mujoco.mj_step(m, d)

                # Track ground-truth metrics
                cur_obj_pos = d.xpos[obj_body_id]
                cur_obj_z = float(cur_obj_pos[2])
                if cur_obj_z > max_obj_z:
                    max_obj_z = cur_obj_z

                if ee_site_id >= 0:
                    dist = float(np.linalg.norm(d.site_xpos[ee_site_id] - cur_obj_pos))
                    if dist < ee_min_dist:
                        ee_min_dist = dist

                # Check contacts
                touching_obj = False
                touching_obs = False
                for c_idx in range(d.ncon):
                    con = d.contact[c_idx]
                    g1, g2 = con.geom1, con.geom2
                    if obj_geom_id in (g1, g2) and table_geom_id not in (g1, g2) and g1 not in obstacle_geom_ids and g2 not in obstacle_geom_ids:
                        touching_obj = True
                    if table_geom_id in (g1, g2) and obj_geom_id not in (g1, g2) and g1 not in obstacle_geom_ids and g2 not in obstacle_geom_ids:
                        table_collision_steps += 1
                    if obstacle_geom_ids and ((g1 in obstacle_geom_ids) or (g2 in obstacle_geom_ids)) and table_geom_id not in (g1, g2):
                        touching_obs = True
                        obstacle_collision_steps += 1
                if touching_obj:
                    obj_contact_steps += 1

                if wp.phase == "lift" and (cur_obj_z - initial_obj_z) >= lift_threshold:
                    lift_held_steps += 1

                # Record trajectory at 25 Hz
                if step % 20 == 0:
                    obs_qpos.append(d.qpos.copy())
                    obs_qvel.append(d.qvel.copy())
                    priv_obj_pos.append(cur_obj_pos.copy())
                    actions_ctrl.append(d.ctrl.copy())
                    ee_poses.append(_read_ee_pose(m, d, ee_site_id, palm_body_id))
                    gripper_states.append([float(wp.gripper_target)])
                    timestamps_s.append(float(d.time))

                # Render frame
                if renderer is not None and step % render_every == 0:
                    renderer.update_scene(d, camera="side_cam")
                    speed_tag = f"{cfg['slowmo']:.2f}x" if cfg["slowmo"] < 1.0 else "1.0x"
                    if touching_obs:
                        status_tag = " | OBSTACLE HIT!"
                        badge_col = (155, 25, 25)
                    elif touching_obj:
                        status_tag = " | CONTACT"
                        badge_col = (20, 90, 45)
                    else:
                        status_tag = ""
                        badge_col = (15, 20, 28)
                    grip_state = "CLOSED" if wp.gripper_target >= 0.5 else "OPEN"
                    frame_tag = f" @{ref_name}" if ref_name != "target_object" else ""
                    side_img = _label_frame(
                        renderer.render(),
                        f"{wp.phase.upper()}{frame_tag} ({speed_tag}) | GRIP:{grip_state}{status_tag}",
                        badge_color=badge_col,
                    )
                    renderer.update_scene(d, camera="top_cam")
                    lift_cm = (cur_obj_z - initial_obj_z) * 100.0
                    top_img = _label_frame(
                        renderer.render(),
                        f"TOP | dz={lift_cm:+.1f}cm",
                    )
                    frames.append(np.concatenate([side_img, top_img], axis=1))

        if renderer is not None and video_path is not None and frames:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(
                video_path,
                np.stack(frames, axis=0),
                fps=fps,
                codec="libx264",
                pixelformat="yuv420p",
                output_params=["-profile:v", "baseline", "-level", "3.0", "-movflags", "+faststart"],
            )
    finally:
        if renderer is not None:
            renderer.close()

    final_obj_pos = d.xpos[obj_body_id].copy()
    final_lift = float(final_obj_pos[2] - initial_obj_z)
    max_lift = float(max_obj_z - initial_obj_z)
    held_seconds = lift_held_steps * m.opt.timestep
    dropped = (max_lift >= 0.05) and (final_lift < 0.04)
    severe_table_collision = table_collision_steps > int(0.35 / m.opt.timestep)
    severe_obstacle_collision = obstacle_collision_steps > 25

    # Ground-truth success criterion (judged strictly by the simulator)
    # If this bundle was built from library.scene.build_task_scene(), use sim.check.evaluate()!
    if "task" in bundle.extras and "object_id" in bundle.extras:
        from sim.check import evaluate as eval_library_task, snapshot as snap_library_task

        if "_before_snapshot" in bundle.extras:
            chk = eval_library_task(
                bundle,
                bundle.extras["_before_snapshot"],
                metrics={"max_lift_cm": round(max_lift * 100.0, 2)},
            )
            sim_success = bool(chk.success)
            checks_str = ", ".join(f"{k}={v}" for k, v in chk.checks.items())
            metrics_str = ", ".join(f"{k}={v:.2f}" for k, v in chk.metrics.items())
            telemetry_summary = (
                f"sim_success={sim_success} | checks=({checks_str}) | "
                f"metrics=({metrics_str}) | ee_min_dist={ee_min_dist * 100:.1f}cm | "
                f"obj_contacts={obj_contact_steps}"
            )
            metrics = {
                "sim_success": sim_success,
                **chk.metrics,
                "ee_min_dist_cm": round(ee_min_dist * 100.0, 2),
                "obj_contact_steps": obj_contact_steps,
                "video_frames": len(frames),
            }
            trajectory = _pack_trajectory(
                bundle=bundle,
                obs_qpos=obs_qpos,
                obs_qvel=obs_qvel,
                priv_obj_pos=priv_obj_pos,
                actions_ctrl=actions_ctrl,
                ee_poses=ee_poses,
                gripper_states=gripper_states,
                timestamps_s=timestamps_s,
            )
            return RolloutResult(
                sim_success=sim_success,
                telemetry_summary=telemetry_summary,
                video_path=video_path if (render_video and video_path is not None) else None,
                metrics=metrics,
                trajectory=trajectory,
            )

    sim_success = bool(
        (final_lift >= lift_threshold)
        and (held_seconds >= 0.30)
        and (not dropped)
        and (not severe_table_collision)
        and (not severe_obstacle_collision)
    )

    metrics: dict[str, float | bool | int | str] = {
        "sim_success": sim_success,
        "initial_obj_z_m": round(initial_obj_z, 4),
        "max_lift_cm": round(max_lift * 100.0, 2),
        "final_lift_cm": round(final_lift * 100.0, 2),
        "held_seconds": round(held_seconds, 2),
        "dropped": dropped,
        "ee_min_dist_cm": round(ee_min_dist * 100.0, 2),
        "obj_contact_steps": obj_contact_steps,
        "table_collision_steps": table_collision_steps,
        "obstacle_collision_steps": obstacle_collision_steps,
        "severe_table_collision": severe_table_collision,
        "severe_obstacle_collision": severe_obstacle_collision,
        "video_frames": len(frames),
    }

    obs_str = (
        f" | obstacle_collisions={obstacle_collision_steps}"
        if obstacle_geom_ids
        else ""
    )
    telemetry_summary = (
        f"sim_success={sim_success} | "
        f"final_lift={final_lift * 100:.1f}cm (max={max_lift * 100:.1f}cm, held={held_seconds:.2f}s) | "
        f"dropped={dropped} | ee_min_dist={ee_min_dist * 100:.1f}cm | "
        f"obj_contacts={obj_contact_steps} | table_collisions={table_collision_steps}{obs_str}"
    )

    trajectory = _pack_trajectory(
        bundle=bundle,
        obs_qpos=obs_qpos,
        obs_qvel=obs_qvel,
        priv_obj_pos=priv_obj_pos,
        actions_ctrl=actions_ctrl,
        ee_poses=ee_poses,
        gripper_states=gripper_states,
        timestamps_s=timestamps_s,
    )

    return RolloutResult(
        sim_success=sim_success,
        telemetry_summary=telemetry_summary,
        video_path=video_path if (render_video and video_path is not None) else None,
        metrics=metrics,
        trajectory=trajectory,
    )


def default_panda_waypoints() -> list[Waypoint]:
    """Hand-written baseline waypoints for Panda top-down grasp & 15cm lift."""
    return [
        Waypoint(
            phase="pre_grasp",
            position_offset=[0.0, 0.0, 0.14],
            orientation_rpy=[3.14159, 0.0, 1.5708],
            gripper_target=0.0,
        ),
        Waypoint(
            phase="grasp",
            position_offset=[0.0, 0.0, 0.0],
            orientation_rpy=[3.14159, 0.0, 1.5708],
            gripper_target=0.0,
        ),
        Waypoint(
            phase="close",
            position_offset=[0.0, 0.0, 0.0],
            orientation_rpy=[3.14159, 0.0, 1.5708],
            gripper_target=1.0,
        ),
        Waypoint(
            phase="lift",
            position_offset=[0.0, 0.0, 0.18],
            orientation_rpy=[3.14159, 0.0, 1.5708],
            gripper_target=1.0,
            duration_s=0.9,
        ),
    ]
