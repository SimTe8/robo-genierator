"""Phase 4 Evaluation Benchmark (§ 6 "Proof It Works").

Runs:
  1. 3-Arm Ablation (one_shot vs. text_only vs. video) on unseen evaluation seeds across
     obstacle_grasp (Panda), fault-injected pick_and_lift (Panda, joint4 locked), and
     in_hand_spindle (Allegro 16-DOF hand).
  2. Planner-Critic Model-Duo Comparison (Neon+Hydrogen, Carbon+Hydrogen, Carbon+Neon, Neon+Carbon).
  3. "Where It Breaks" stress cases (multi-joint fault, free-cube 360-deg Allegro) + DR transfer sweep.

Saves structured JSON incrementally to outputs/benchmark_results.json (and mirrors to
../gemini4nyc-ui/outputs/benchmark_results.json if the UI worktree is present) plus
a readable Markdown report at outputs/benchmark_summary.md.

Usage:
  python backend/agents/benchmark.py --workers 4
  python backend/agents/benchmark.py --quick --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from agents.loop import run_closed_loop
from schemas import RunRequest
from sim.domain_rand import run_domain_randomization


@dataclass
class BenchJob:
    suite: str  # "ablation" | "model_duo" | "stress"
    label: str
    request_dict: dict
    run_dr: bool = False


def _run_one_job(job: BenchJob) -> dict:
    """Worker entrypoint executed in its own process (clean EGL context per process)."""
    os.environ["MUJOCO_GL"] = "egl"
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    req = RunRequest.model_validate(job.request_dict)
    t0 = time.time()
    try:
        summary = run_closed_loop(req)
        events = summary.events

        # Compute critic-vs-simulator agreement & false accepts across all evaluated attempts
        eval_attempts = 0
        agreements = 0
        false_accepts = 0
        for ev in events:
            if req.feedback_mode == "one_shot":
                continue
            eval_attempts += 1
            critic_accept = ev.decision == "ACCEPT_TRAJECTORY"
            if critic_accept == ev.sim_success:
                agreements += 1
            if critic_accept and not ev.sim_success:
                false_accepts += 1

        dr_info = None
        if job.run_dr and summary.final_sim_success and events:
            dr = run_domain_randomization(
                waypoints=events[-1].waypoints,
                robot_id=req.robot_id,
                task_type=req.task_type,
                base_seed=req.scene_seed,
                object_shape=req.object_shape,
                disabled_joints=req.disabled_joints,
                target_rotation_deg=req.target_rotation_deg,
                num_episodes=20,
                library_object_id=req.library_object_id,
                library_task_id=req.library_task_id,
                library_task_params=req.library_task_params,
                library_clutter=req.library_clutter,
                base_layout_seed=req.library_layout_seed,
            )
            dr_info = {
                "passed": dr.passed_episodes,
                "total": dr.total_episodes,
                "transfer_rate": dr.transfer_rate,
                "elapsed_s": dr.elapsed_s,
                "exported_count": len(dr.exported_paths),
            }

        task_key = (
            f"wuji_{req.library_task_id}"
            if (req.robot_id == "wuji_hand" and req.library_task_id)
            else (req.library_task_id or req.task_type)
        )
        return {
            "suite": job.suite,
            "label": job.label,
            "run_id": summary.run_id,
            "robot_id": req.robot_id,
            "task_type": task_key,
            "scene_seed": req.scene_seed,
            "disabled_joints": req.disabled_joints,
            "feedback_mode": req.feedback_mode,
            "planner_model": req.planner_model,
            "critic_model": req.critic_model,
            "duo": f"{req.planner_model.split('/')[-1]} + {req.critic_model.split('/')[-1]}",
            "attempts": len(events),
            "sim_success": summary.final_sim_success,
            "final_decision": summary.final_decision,
            "total_latency_s": summary.total_latency_s,
            "avg_iter_latency_s": round(
                sum(e.latency_s for e in events) / max(1, len(events)), 2
            ),
            "eval_attempts": eval_attempts,
            "critic_agreements": agreements,
            "false_accepts": false_accepts,
            "final_telemetry": events[-1].telemetry_summary if events else "",
            "final_reason": events[-1].reason[:300] if events else "",
            "final_video_url": events[-1].video_url if events else "",
            "attempt_1_video_url": events[0].video_url if events else "",
            "exported_trajectory": summary.exported_trajectory_path,
            "dr_transfer": dr_info,
            "error": None,
        }
    except Exception as exc:
        return {
            "suite": job.suite,
            "label": job.label,
            "run_id": "error",
            "robot_id": req.robot_id,
            "task_type": req.task_type,
            "scene_seed": req.scene_seed,
            "disabled_joints": req.disabled_joints,
            "feedback_mode": req.feedback_mode,
            "planner_model": req.planner_model,
            "critic_model": req.critic_model,
            "duo": f"{req.planner_model.split('/')[-1]} + {req.critic_model.split('/')[-1]}",
            "attempts": 0,
            "sim_success": False,
            "final_decision": "GIVE_UP",
            "total_latency_s": round(time.time() - t0, 2),
            "avg_iter_latency_s": 0.0,
            "eval_attempts": 0,
            "critic_agreements": 0,
            "false_accepts": 0,
            "final_telemetry": "",
            "final_reason": f"ERROR: {exc}",
            "final_video_url": "",
            "attempt_1_video_url": "",
            "exported_trajectory": None,
            "dr_transfer": None,
            "error": str(exc),
        }


def _aggregate(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {}
    succ = sum(1 for r in records if r["sim_success"])
    eval_att = sum(r["eval_attempts"] for r in records)
    agrees = sum(r["critic_agreements"] for r in records)
    false_acc = sum(r["false_accepts"] for r in records)
    return {
        "episodes": n,
        "successes": succ,
        "success_rate": round(succ / n, 4),
        "avg_attempts": round(sum(r["attempts"] for r in records) / n, 2),
        "avg_total_latency_s": round(sum(r["total_latency_s"] for r in records) / n, 2),
        "avg_iter_latency_s": round(sum(r["avg_iter_latency_s"] for r in records) / n, 2),
        "critic_sim_agreement": round(agrees / eval_att, 4) if eval_att > 0 else None,
        "false_accept_rate": round(false_acc / eval_att, 4) if eval_att > 0 else None,
        "eval_attempts": eval_att,
        "false_accepts": false_acc,
    }


def build_jobs(quick: bool = False) -> list[BenchJob]:
    """Construct unseen-seed evaluation jobs for 3-arm ablation, model-duo study, and stress tests."""
    jobs: list[BenchJob] = []

    # Unseen evaluation seeds (never used during development; dev used seed=42)
    obs_seeds = [101, 102] if quick else [101, 102, 103, 104]
    fault_seeds = [201] if quick else [201, 202, 203]
    spindle_seeds = [301] if quick else [301, 302]

    # --- SUITE 1: 3-Arm Ablation (One-Shot vs Text-Only vs Video) across 3 Robots & Multi-Object Tasks ---
    for seed in obs_seeds:
        for mode in ("one_shot", "text_only", "video"):
            jobs.append(
                BenchJob(
                    suite="ablation",
                    label=f"obstacle_grasp_seed{seed}_{mode}",
                    request_dict=RunRequest(
                        task_prompt="Pick up the object and lift it at least 12 cm above the table while avoiding any workspace obstacles.",
                        task_type="obstacle_grasp",
                        robot_id="franka_emika_panda",
                        scene_seed=seed,
                        max_attempts=3,
                        feedback_mode=mode,
                        planner_model="models/gemini-4-neon",
                        critic_model="models/gemini-4-hydrogen",
                        thinking_level="medium",
                    ).model_dump(),
                    run_dr=(mode == "video" and seed == obs_seeds[0]),
                )
            )

    for seed in fault_seeds:
        for mode in ("one_shot", "text_only", "video"):
            jobs.append(
                BenchJob(
                    suite="ablation",
                    label=f"panda_fault_joint4_seed{seed}_{mode}",
                    request_dict=RunRequest(
                        task_prompt="Pick up the object and lift it 15 cm above the table while compensating for the locked elbow joint4.",
                        task_type="pick_and_lift",
                        robot_id="franka_emika_panda",
                        scene_seed=seed,
                        disabled_joints=["joint4"],
                        max_attempts=3,
                        feedback_mode=mode,
                        planner_model="models/gemini-4-neon",
                        critic_model="models/gemini-4-hydrogen",
                        thinking_level="medium",
                    ).model_dump(),
                    run_dr=(mode == "video" and seed == fault_seeds[0]),
                )
            )

    for seed in spindle_seeds:
        for mode in ("one_shot", "text_only", "video"):
            jobs.append(
                BenchJob(
                    suite="ablation",
                    label=f"allegro_spindle90_seed{seed}_{mode}",
                    request_dict=RunRequest(
                        task_prompt="Rotate the multi-colored cube on the spindle in-hand by at least 90 degrees using coordinated finger gaits.",
                        task_type="in_hand_spindle",
                        robot_id="wonik_allegro",
                        scene_seed=seed,
                        max_attempts=3,
                        target_rotation_deg=90.0,
                        feedback_mode=mode,
                        planner_model="models/gemini-4-neon",
                        critic_model="models/gemini-4-hydrogen",
                        thinking_level="medium",
                    ).model_dump(),
                    run_dr=(mode == "video" and seed == spindle_seeds[0]),
                )
            )

    # Multi-object 2-body stacking (MimicGen-style reference_frame="base_block")
    for mode in ("one_shot", "text_only", "video"):
        jobs.append(
            BenchJob(
                suite="ablation",
                label=f"panda_stack_on_block_seed401_{mode}",
                request_dict=RunRequest(
                    task_prompt="Pick up the cube and stack it cleanly centered on top of the base block, then release it.",
                    task_type="pick_and_lift",
                    robot_id="franka_emika_panda",
                    scene_seed=401,
                    library_object_id="cube",
                    library_task_id="stack_on_block",
                    max_attempts=3,
                    feedback_mode=mode,
                    planner_model="models/gemini-4-neon",
                    critic_model="models/gemini-4-hydrogen",
                    thinking_level="medium",
                ).model_dump(),
                run_dr=(mode == "video"),
            )
        )

    # Multi-frame pick-and-place on target pad (simultaneous mug + pad randomization)
    for mode in ("one_shot", "text_only", "video"):
        jobs.append(
            BenchJob(
                suite="ablation",
                label=f"panda_mug_place_on_pad_seed402_{mode}",
                request_dict=RunRequest(
                    task_prompt="Pick up the mug, place it upright on the green target pad, and release it.",
                    task_type="pick_and_lift",
                    robot_id="franka_emika_panda",
                    scene_seed=402,
                    library_object_id="mug",
                    library_task_id="place_on_pad",
                    max_attempts=3,
                    feedback_mode=mode,
                    planner_model="models/gemini-4-neon",
                    critic_model="models/gemini-4-hydrogen",
                    thinking_level="medium",
                ).model_dump(),
                run_dr=(mode == "video"),
            )
        )

    # 20-DoF 5-finger Wuji hand tabletop grasp & lift
    for mode in ("one_shot", "text_only", "video"):
        jobs.append(
            BenchJob(
                suite="ablation",
                label=f"wuji_20dof_pick_lift_seed501_{mode}",
                request_dict=RunRequest(
                    task_prompt="Pick up the cube with the 5-finger Wuji hand and lift it at least 10 cm above the table.",
                    task_type="pick_and_lift",
                    robot_id="wuji_hand",
                    scene_seed=501,
                    library_object_id="cube",
                    library_task_id="pick_lift",
                    max_attempts=3,
                    feedback_mode=mode,
                    planner_model="models/gemini-4-neon",
                    critic_model="models/gemini-4-hydrogen",
                    thinking_level="medium",
                ).model_dump(),
                run_dr=(mode == "video"),
            )
        )

    # --- SUITE 2: Planner-Critic Model-Duo Comparison (on unseen obstacle_grasp seeds) ---
    duo_seeds = [101] if quick else [101, 102]
    extra_duos = [
        ("models/gemini-4-carbon", "models/gemini-4-hydrogen"),
        ("models/gemini-4-carbon", "models/gemini-4-neon"),
        ("models/gemini-4-neon", "models/gemini-4-carbon"),
    ]
    for planner_m, critic_m in extra_duos:
        for seed in duo_seeds:
            jobs.append(
                BenchJob(
                    suite="model_duo",
                    label=f"duo_{planner_m.split('/')[-1]}_{critic_m.split('/')[-1]}_obs_seed{seed}",
                    request_dict=RunRequest(
                        task_prompt="Pick up the object and lift it at least 12 cm above the table while avoiding any workspace obstacles.",
                        task_type="obstacle_grasp",
                        robot_id="franka_emika_panda",
                        scene_seed=seed,
                        max_attempts=3,
                        feedback_mode="video",
                        planner_model=planner_m,
                        critic_model=critic_m,
                        thinking_level="medium",
                    ).model_dump(),
                )
            )

    # --- SUITE 3: "Where It Breaks" Stress Cases ---
    jobs.append(
        BenchJob(
            suite="stress",
            label="break_panda_double_fault_joint2_joint4_seed201",
            request_dict=RunRequest(
                task_prompt="Pick up the object and lift it 15 cm above the table with BOTH joint2 and joint4 locked.",
                task_type="pick_and_lift",
                robot_id="franka_emika_panda",
                scene_seed=201,
                disabled_joints=["joint2", "joint4"],
                max_attempts=3,
                feedback_mode="video",
                planner_model="models/gemini-4-neon",
                critic_model="models/gemini-4-hydrogen",
                thinking_level="medium",
            ).model_dump(),
        )
    )
    jobs.append(
        BenchJob(
            suite="stress",
            label="break_allegro_freecube_360deg_seed301",
            request_dict=RunRequest(
                task_prompt="Rotate the free multi-colored cube in-hand by a full 360 degrees without dropping it.",
                task_type="in_hand_rotation",
                robot_id="wonik_allegro",
                scene_seed=301,
                max_attempts=2,
                target_rotation_deg=360.0,
                feedback_mode="video",
                planner_model="models/gemini-4-neon",
                critic_model="models/gemini-4-hydrogen",
                thinking_level="medium",
            ).model_dump(),
        )
    )

    return jobs


def build_report(records: list[dict], elapsed_s: float) -> dict:
    ablation = [r for r in records if r["suite"] == "ablation"]
    summary_by_arm = {
        mode: _aggregate([r for r in ablation if r["feedback_mode"] == mode])
        for mode in ("one_shot", "text_only", "video")
    }

    summary_by_task_and_arm: dict[str, dict] = {}
    seen_tasks = list(dict.fromkeys(r["task_type"] for r in ablation))
    for task in seen_tasks:
        summary_by_task_and_arm[task] = {
            mode: _aggregate(
                [r for r in ablation if r["task_type"] == task and r["feedback_mode"] == mode]
            )
            for mode in ("one_shot", "text_only", "video")
        }

    # Combine Suite 2 records with matching Neon+Hydrogen obstacle_grasp video episodes (seeds 101, 102)
    duo_records = [r for r in records if r["suite"] == "model_duo"]
    duo_seeds_seen = {r["scene_seed"] for r in duo_records}
    if duo_seeds_seen:
        nh_matching = [
            r
            for r in ablation
            if r["task_type"] == "obstacle_grasp"
            and r["feedback_mode"] == "video"
            and r["scene_seed"] in duo_seeds_seen
        ]
        all_duo = duo_records + nh_matching
    else:
        all_duo = duo_records

    summary_by_duo: dict[str, dict] = {}
    for duo_name in sorted({r["duo"] for r in all_duo}):
        summary_by_duo[duo_name] = _aggregate([r for r in all_duo if r["duo"] == duo_name])

    stress_records = [r for r in records if r["suite"] == "stress"]
    dr_records = [
        {"label": r["label"], "task_type": r["task_type"], "dr_transfer": r["dr_transfer"]}
        for r in records
        if r.get("dr_transfer") is not None
    ]

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_wall_time_s": round(elapsed_s, 1),
        "total_episodes": len(records),
        "decisions_model": "gemini-decision-1",
        "summary_by_arm": summary_by_arm,
        "summary_by_task_and_arm": summary_by_task_and_arm,
        "summary_by_duo": summary_by_duo,
        "where_it_breaks": stress_records,
        "domain_randomization": dr_records,
        "episodes": records,
    }


def format_markdown(report: dict) -> str:
    lines = [
        "# Benchmark Report (§ 6 Proof It Works)",
        f"- **Generated:** {report['generated_at']} | **Wall Time:** {report['total_wall_time_s']}s | **Total Episodes:** {report['total_episodes']}",
        f"- **Decisions API Gate:** `{report['decisions_model']}`",
        "",
        "## 1. Three-Arm Ablation (All Unseen Evaluation Seeds)",
        "| Arm | Episodes | Sim Success Rate | Avg Attempts | Avg Loop Latency | Critic–Sim Agreement | False Accept Rate |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for arm, s in report["summary_by_arm"].items():
        if not s:
            continue
        agree = f"{s['critic_sim_agreement'] * 100:.1f}%" if s["critic_sim_agreement"] is not None else "—"
        fa = f"{s['false_accept_rate'] * 100:.1f}% ({s['false_accepts']}/{s['eval_attempts']})" if s["false_accept_rate"] is not None else "—"
        lines.append(
            f"| `{arm}` | {s['episodes']} | **{s['success_rate'] * 100:.1f}%** ({s['successes']}/{s['episodes']}) | "
            f"{s['avg_attempts']} | {s['avg_iter_latency_s']}s | {agree} | {fa} |"
        )

    lines.extend([
        "",
        "## 2. Breakdown by Task × Arm",
        "| Task | One-Shot | Text-Only | Video (Ours) |",
        "| :--- | :---: | :---: | :---: |",
    ])
    for task, arms in report["summary_by_task_and_arm"].items():
        cells = []
        for arm in ("one_shot", "text_only", "video"):
            s = arms.get(arm)
            cells.append(f"{s['success_rate'] * 100:.0f}% ({s['successes']}/{s['episodes']})" if s else "—")
        lines.append(f"| `{task}` | {cells[0]} | {cells[1]} | **{cells[2]}** |")

    if report["summary_by_duo"]:
        lines.extend([
            "",
            "## 3. Planner–Critic Model Duo Comparison (`obstacle_grasp`, Video Mode)",
            "| Planner + Critic Duo | Episodes | Sim Success Rate | Avg Attempts | Avg Iter Latency | Total Latency | Critic–Sim Agreement |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for duo, s in report["summary_by_duo"].items():
            agree = f"{s['critic_sim_agreement'] * 100:.1f}%" if s["critic_sim_agreement"] is not None else "—"
            lines.append(
                f"| `{duo}` | {s['episodes']} | **{s['success_rate'] * 100:.1f}%** ({s['successes']}/{s['episodes']}) | "
                f"{s['avg_attempts']} | {s['avg_iter_latency_s']}s | {s['avg_total_latency_s']}s | {agree} |"
            )

    if report["where_it_breaks"]:
        lines.extend([
            "",
            "## 4. Where It Breaks (Stress Cases)",
            "| Stress Case | Sim Success | Decision | Telemetry |",
            "| :--- | :---: | :---: | :--- |",
        ])
        for r in report["where_it_breaks"]:
            lines.append(
                f"| `{r['label']}` | `{r['sim_success']}` | `{r['final_decision']}` | `{r['final_telemetry']}` |"
            )

    if report["domain_randomization"]:
        lines.extend(["", "## 5. Domain Randomization Zero-Shot Transfer (10 Randomized Scenes / Verified Skill)"])
        for d in report["domain_randomization"]:
            dr = d["dr_transfer"]
            lines.append(
                f"- **`{d['label']}`**: **{dr['passed']}/{dr['total']} ({dr['transfer_rate'] * 100:.0f}%)** passed in `{dr['elapsed_s']}s`"
            )

    return "\n".join(lines) + "\n"


def _save_outputs(report: dict) -> None:
    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_results.json"
    md_path = out_dir / "benchmark_summary.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(format_markdown(report), encoding="utf-8")

    # Mirror to sibling UI worktree if present so the React UI can read it immediately
    ui_out = REPO_ROOT.parent / "gemini4nyc-ui" / "outputs"
    if ui_out.is_dir():
        try:
            shutil.copy2(json_path, ui_out / "benchmark_results.json")
        except OSError:
            pass


def build_post_fix_jobs() -> list[BenchJob]:
    """Targeted post-failure-fix verification + 20-episode DR statistics per verified skill."""
    jobs: list[BenchJob] = []
    # Uses Carbon + Hydrogen (fastest duo from initial study: 55s/iter) + post-fix pitch safety guard
    for seed in (101, 102, 103, 104):
        for mode in ("one_shot", "text_only", "video"):
            jobs.append(
                BenchJob(
                    suite="ablation",
                    label=f"postfix_obstacle_grasp_seed{seed}_{mode}",
                    request_dict=RunRequest(
                        task_prompt="Pick up the object and lift it at least 12 cm above the table while avoiding any workspace obstacles.",
                        task_type="obstacle_grasp",
                        robot_id="franka_emika_panda",
                        scene_seed=seed,
                        max_attempts=4,
                        feedback_mode=mode,
                        planner_model="models/gemini-4-carbon",
                        critic_model="models/gemini-4-hydrogen",
                    ).model_dump(),
                    run_dr=True,
                )
            )

    for seed in (301, 302):
        for mode in ("one_shot", "video"):
            jobs.append(
                BenchJob(
                    suite="ablation",
                    label=f"postfix_allegro_spindle90_seed{seed}_{mode}",
                    request_dict=RunRequest(
                        task_prompt="Rotate the multi-colored cube on the spindle in-hand by at least 90 degrees using coordinated finger gaits.",
                        task_type="in_hand_spindle",
                        robot_id="wonik_allegro",
                        scene_seed=seed,
                        max_attempts=3,
                        target_rotation_deg=90.0,
                        feedback_mode=mode,
                        planner_model="models/gemini-4-neon",
                        critic_model="models/gemini-4-hydrogen",
                    ).model_dump(),
                    run_dr=True,
                )
            )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 § 6 Benchmark Suite.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes.")
    parser.add_argument("--quick", action="store_true", help="Smaller fast suite (~3-5 min).")
    parser.add_argument(
        "--post-fix",
        action="store_true",
        help="Run post-failure-fix verification with 20-episode DR randomization on every verified skill.",
    )
    args = parser.parse_args()

    jobs = build_post_fix_jobs() if args.post_fix else build_jobs(quick=args.quick)
    print(f"Launching § 6 Benchmark: {len(jobs)} episodes across {args.workers} parallel processes...")
    t0 = time.time()
    records: list[dict] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fut_to_job = {pool.submit(_run_one_job, j): j for j in jobs}
        for i, fut in enumerate(as_completed(fut_to_job), start=1):
            rec = fut.result()
            records.append(rec)
            status_icon = "PASS" if rec["sim_success"] else "FAIL"
            print(
                f"[{i:02d}/{len(jobs):02d}] {status_icon} | {rec['label']} | "
                f"att={rec['attempts']} | dec={rec['final_decision']} | {rec['total_latency_s']:.1f}s",
                flush=True,
            )
            _save_outputs(build_report(records, time.time() - t0))

    final_report = build_report(records, time.time() - t0)
    _save_outputs(final_report)
    print("\n" + format_markdown(final_report))
    print(f"Saved JSON: {REPO_ROOT / 'outputs' / 'benchmark_results.json'}")
    print(f"Saved MD:   {REPO_ROOT / 'outputs' / 'benchmark_summary.md'}")


if __name__ == "__main__":
    main()
