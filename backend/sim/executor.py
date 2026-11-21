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
class RolloutResult:
    sim_success: bool
    telemetry_summary: str
    video_path: Path | None
    metrics: dict[str, float | bool | int | str]
    trajectory: dict[str, np.ndarray] = field(default_factory=dict)


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
    """Map normalized closure [0=open, 1=closed] to Allegro's 16 finger actuators."""
    # Finger actuator prefixes: ff (index), mf (middle), rf (ring), th (thumb)
    prefix_map = {
        "index": "ff",
        "middle": "mf",
        "ring": "rf",
        "thumb": "th",
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
        for fname, pref in prefix_map.items():
            if aname.startswith(pref):
                closure = defaults[fname]
                break

        # Abduction joints (a0) stay centered; flexion joints (a1, a2, a3) close with closure
        if aname.endswith("0"):
            if aname.startswith("th"):
                data.ctrl[aid] = lo + 0.75 * (hi - lo) * closure
            else:
                data.ctrl[aid] = 0.0
        else:
            data.ctrl[aid] = lo + (hi - lo) * (0.10 + 0.75 * closure)


def _execute_in_hand_rotation(
    bundle: SceneBundle,
    waypoints: Sequence[Waypoint],
    video_path: Path | None = None,
    width: int = 384,
    height: int = 288,
    fps: int = 15,
    render_video: bool = True,
    target_rotation_deg: float = 60.0,
) -> RolloutResult:
    """Execute a cyclic 4-phase dexterous finger gait for Allegro in-hand cube rotation."""
    m, d = bundle.model, bundle.data
    disabled_set = set(bundle.disabled_joints)

    # Settle cube gently into the cupped palm
    for _ in range(120):
        mujoco.mj_step(m, d)

    obj_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    obj_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom")
    obj_jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "target_object_freejoint")
    obj_dof_adr = m.jnt_dofadr[obj_jnt_id]
    floor_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

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
                else:
                    ctrl[aid] = lo + (hi - lo) * (0.08 + 0.82 * curl[pref])
            elif pref == "th":
                if aname == "tha0":
                    ctrl[aid] = np.clip(th_opp, lo, hi)
                elif aname == "tha1":
                    ctrl[aid] = np.clip(th_swp, lo, hi)
                else:
                    ctrl[aid] = lo + (hi - lo) * (0.08 + 0.82 * curl["th"])
        return ctrl

    # Scale cyclic repetitions to the target rotation (e.g. 60deg -> 2 cycles, 360deg -> 8 cycles)
    gait_cycles = max(2, min(10, int(round(target_rotation_deg / 45.0))))
    step_dur_s = 0.55
    n_steps_per_wp = int(round(step_dur_s / m.opt.timestep))
    total_sim_steps = gait_cycles * len(waypoints) * n_steps_per_wp
    # Keep rendered video compact (~75 frames = 5s at 15fps) regardless of cycle count
    render_every = max(1, total_sim_steps // 75)

    renderer = mujoco.Renderer(m, height=height, width=width) if render_video else None
    frames: list[np.ndarray] = []
    obs_qpos: list[np.ndarray] = []
    priv_obj_pos: list[np.ndarray] = []
    actions_ctrl: list[np.ndarray] = []

    cum_yaw_deg = 0.0
    max_abs_yaw_deg = 0.0
    obj_contact_steps = 0
    dropped = False
    global_step = 0

    try:
        for cycle in range(gait_cycles):
            for idx, wp in enumerate(waypoints):
                start_ctrl = d.ctrl.copy()
                goal_ctrl = _waypoint_to_allegro_ctrl(wp)

                for step in range(n_steps_per_wp):
                    frac = min(1.0, (step + 1) / (0.8 * n_steps_per_wp))
                    d.ctrl[:] = (1.0 - frac) * start_ctrl + frac * goal_ctrl
                    mujoco.mj_step(m, d)
                    global_step += 1

                    # Integrate unwrapped Z-axis angular velocity (handles 360+ deg cleanly!)
                    # Only accumulate while cube is still in-hand (not after dropping)
                    cur_pos = d.xpos[obj_body_id]
                    if float(cur_pos[2]) < -0.02:
                        dropped = True
                    if not dropped:
                        wz = float(d.qvel[obj_dof_adr + 5])
                        cum_yaw_deg += float(np.degrees(wz * m.opt.timestep))
                        if abs(cum_yaw_deg) > max_abs_yaw_deg:
                            max_abs_yaw_deg = abs(cum_yaw_deg)

                    for c_idx in range(d.ncon):
                        con = d.contact[c_idx]
                        if obj_geom_id in (con.geom1, con.geom2) and floor_geom_id not in (con.geom1, con.geom2):
                            obj_contact_steps += 1
                            break

                    if global_step % 25 == 0:
                        obs_qpos.append(d.qpos.copy())
                        priv_obj_pos.append(cur_pos.copy())
                        actions_ctrl.append(d.ctrl.copy())

                    if renderer is not None and global_step % render_every == 0:
                        status = "DROPPED!" if dropped else f"YAW={cum_yaw_deg:+.0f}/{target_rotation_deg:.0f}deg"
                        badge = (140, 25, 25) if dropped else (20, 90, 45)
                        renderer.update_scene(d, camera="side_cam")
                        side_img = _label_frame(
                            renderer.render(),
                            f"CYC {cycle + 1}/{gait_cycles} S{idx + 1} | {status}",
                            badge_color=badge,
                        )
                        renderer.update_scene(d, camera="top_cam")
                        top_img = _label_frame(
                            renderer.render(),
                            f"TOP | max={max_abs_yaw_deg:.0f}deg",
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
            )
    finally:
        if renderer is not None:
            renderer.close()

    final_pos = d.xpos[obj_body_id].copy()
    if float(final_pos[2]) < -0.02:
        dropped = True

    final_rot_deg = abs(cum_yaw_deg)
    sim_success = bool((not dropped) and (final_rot_deg >= target_rotation_deg))

    metrics: dict[str, float | bool | int | str] = {
        "sim_success": sim_success,
        "signed_yaw_deg": round(cum_yaw_deg, 1),
        "final_rot_deg": round(final_rot_deg, 1),
        "max_rot_deg": round(max_abs_yaw_deg, 1),
        "target_rotation_deg": target_rotation_deg,
        "gait_cycles": gait_cycles,
        "dropped": dropped,
        "final_obj_z_cm": round(float(final_pos[2]) * 100.0, 2),
        "obj_displacement_cm": round(float(np.linalg.norm(final_pos[:2] - init_pos[:2])) * 100.0, 2),
        "obj_contact_steps": obj_contact_steps,
        "video_frames": len(frames),
    }

    telemetry_summary = (
        f"sim_success={sim_success} | "
        f"net_yaw={cum_yaw_deg:+.1f}deg (abs={final_rot_deg:.1f}deg, max={max_abs_yaw_deg:.1f}deg, "
        f"target>={target_rotation_deg:.0f}deg, cycles={gait_cycles}) | "
        f"dropped={dropped} | final_z={final_pos[2] * 100:.1f}cm | "
        f"obj_contacts={obj_contact_steps}"
    )

    trajectory = {
        "observation_qpos": np.asarray(obs_qpos, dtype=np.float32),
        "privileged_obj_pos": np.asarray(priv_obj_pos, dtype=np.float32),
        "action_ctrl": np.asarray(actions_ctrl, dtype=np.float32),
    }

    return RolloutResult(
        sim_success=sim_success,
        telemetry_summary=telemetry_summary,
        video_path=video_path if (render_video and video_path is not None) else None,
        metrics=metrics,
        trajectory=trajectory,
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
) -> RolloutResult:
    """Execute waypoints in MuJoCo, render slow-mo multi-view MP4, and verify success."""
    if bundle.task_type == "in_hand_rotation":
        return _execute_in_hand_rotation(
            bundle=bundle,
            waypoints=waypoints,
            video_path=video_path,
            width=width,
            height=height,
            fps=fps,
            render_video=render_video,
            target_rotation_deg=target_rotation_deg,
        )
    m, d = bundle.model, bundle.data
    robot_id = bundle.robot_id
    disabled_set = set(bundle.disabled_joints)

    # Settle object briefly before reading its reference position
    for _ in range(100):
        mujoco.mj_step(m, d)

    obj_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    obj_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom")
    table_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    ee_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    mocap_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm_mocap")
    mocap_id = m.body_mocapid[mocap_body_id] if mocap_body_id >= 0 else -1

    initial_obj_pos = d.xpos[obj_body_id].copy()
    initial_obj_z = float(initial_obj_pos[2])

    # Build disabled DOF mask for Panda
    disabled_dof_mask = np.zeros(m.nv, dtype=bool)
    for jname in disabled_set:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            disabled_dof_mask[m.jnt_dofadr[jid]] = True

    # Phase durations (sim seconds) and slow-motion playback factors
    # Keep total video ~5-6s at 15fps (~80 frames) for <8s rollout latency
    phase_config = {
        "pre_grasp": {"duration": 0.8, "slowmo": 1.0},
        "grasp": {"duration": 0.8, "slowmo": 0.50},
        "close": {"duration": 0.5, "slowmo": 0.40},
        "lift": {"duration": 1.1, "slowmo": 0.70},
    }

    renderer = mujoco.Renderer(m, height=height, width=width) if render_video else None
    frames: list[np.ndarray] = []

    # Telemetry & trajectory buffers
    max_obj_z = initial_obj_z
    ee_min_dist = float("inf")
    obj_contact_steps = 0
    table_collision_steps = 0
    lift_held_steps = 0
    lift_threshold = 0.08  # 8 cm above initial resting height

    obs_qpos: list[np.ndarray] = []
    priv_obj_pos: list[np.ndarray] = []
    actions_ctrl: list[np.ndarray] = []

    try:
        for wp in waypoints:
            cfg = phase_config.get(wp.phase, {"duration": 0.8, "slowmo": 1.0})
            n_steps = max(1, int(round(cfg["duration"] / m.opt.timestep)))
            sim_dt_per_frame = (1.0 / fps) * cfg["slowmo"]
            render_every = max(1, int(round(sim_dt_per_frame / m.opt.timestep)))

            # Object-centric target pose anchored to initial object position
            target_pos = initial_obj_pos + np.asarray(wp.position_offset, dtype=np.float64)
            target_mat = _rpy_to_mat(wp.orientation_rpy)
            target_quat = _rpy_to_quat_wxyz(wp.orientation_rpy)

            # Pre-solve target joint configuration for Panda
            q_start = d.ctrl[:7].copy() if robot_id == "franka_emika_panda" else None
            q_goal = (
                _solve_panda_ik(m, d.qpos, ee_site_id, target_pos, target_mat, disabled_dof_mask)
                if robot_id == "franka_emika_panda"
                else None
            )

            # For Allegro mocap smooth interpolation
            start_mocap_pos = d.mocap_pos[mocap_id].copy() if mocap_id >= 0 else None
            start_mocap_quat = d.mocap_quat[mocap_id].copy() if mocap_id >= 0 else None

            for step in range(n_steps):
                # Smoothly ramp over first 75% of phase, hold last 25% for settling
                frac = min(1.0, (step + 1) / (0.75 * n_steps))

                if robot_id == "franka_emika_panda" and q_start is not None and q_goal is not None:
                    d.ctrl[:7] = (1.0 - frac) * q_start + frac * q_goal
                    # Panda actuator8: 255 = open (0.04m), 0 = closed (0.0m)
                    d.ctrl[7] = float(np.clip((1.0 - wp.gripper_target) * 255.0, 0.0, 255.0))
                else:
                    if mocap_id >= 0 and start_mocap_pos is not None and start_mocap_quat is not None:
                        d.mocap_pos[mocap_id] = (1.0 - frac) * start_mocap_pos + frac * target_pos
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
                for c_idx in range(d.ncon):
                    con = d.contact[c_idx]
                    g1, g2 = con.geom1, con.geom2
                    if obj_geom_id in (g1, g2) and table_geom_id not in (g1, g2):
                        touching_obj = True
                    if table_geom_id in (g1, g2) and obj_geom_id not in (g1, g2):
                        # Robot-table collision
                        table_collision_steps += 1
                if touching_obj:
                    obj_contact_steps += 1

                if wp.phase == "lift" and (cur_obj_z - initial_obj_z) >= lift_threshold:
                    lift_held_steps += 1

                # Record trajectory at 25 Hz
                if step % 20 == 0:
                    obs_qpos.append(d.qpos.copy())
                    priv_obj_pos.append(cur_obj_pos.copy())
                    actions_ctrl.append(d.ctrl.copy())

                # Render frame
                if renderer is not None and step % render_every == 0:
                    renderer.update_scene(d, camera="side_cam")
                    speed_tag = f"{cfg['slowmo']:.2f}x" if cfg["slowmo"] < 1.0 else "1.0x"
                    contact_tag = " | CONTACT" if touching_obj else ""
                    side_img = _label_frame(
                        renderer.render(),
                        f"{wp.phase.upper()} ({speed_tag}){contact_tag}",
                        badge_color=(20, 90, 45) if touching_obj else (15, 20, 28),
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

    # Ground-truth success criterion (judged strictly by the simulator)
    sim_success = bool(
        (final_lift >= lift_threshold)
        and (held_seconds >= 0.35)
        and (not dropped)
        and (not severe_table_collision)
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
        "severe_table_collision": severe_table_collision,
        "video_frames": len(frames),
    }

    telemetry_summary = (
        f"sim_success={sim_success} | "
        f"final_lift={final_lift * 100:.1f}cm (max={max_lift * 100:.1f}cm, held={held_seconds:.2f}s) | "
        f"dropped={dropped} | ee_min_dist={ee_min_dist * 100:.1f}cm | "
        f"obj_contacts={obj_contact_steps} | table_collisions={table_collision_steps}"
    )

    trajectory = {
        "observation_qpos": np.asarray(obs_qpos, dtype=np.float32),
        "privileged_obj_pos": np.asarray(priv_obj_pos, dtype=np.float32),
        "action_ctrl": np.asarray(actions_ctrl, dtype=np.float32),
    }

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
        ),
    ]
