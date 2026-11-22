"""CLI to run a single closed-loop Planner -> MuJoCo -> Video Critic + Decisions API episode.

Usage:
  python backend/agents/run_loop.py --robot franka_emika_panda --seed 42
  python backend/agents/run_loop.py --task-type in_hand_rotation --seed 42 --target-rotation 360 --max-attempts 6
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from agents.loop import run_closed_loop
from schemas import IterationEvent, RunRequest


class StageSpinner:
    """Animated terminal spinner with live elapsed timer so users know each stage is working."""

    FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._label: str | None = None
        self._stage_start: float = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._is_tty = sys.stdout.isatty()

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _finish_current_locked(self) -> None:
        if self._label is not None:
            dt = time.time() - self._stage_start
            if self._is_tty:
                sys.stdout.write(f"\r\033[K✓ {self._label} ({dt:.1f}s)\n")
            else:
                sys.stdout.write(f"✓ {self._label} ({dt:.1f}s)\n")
            sys.stdout.flush()
            self._label = None

    def set_stage(self, label: str) -> None:
        with self._lock:
            self._finish_current_locked()
            self._label = label
            self._stage_start = time.time()
            if not self._is_tty:
                sys.stdout.write(f"… {label}...\n")
                sys.stdout.flush()

    def pause(self) -> None:
        with self._lock:
            self._finish_current_locked()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        with self._lock:
            self._finish_current_locked()

    def _run(self) -> None:
        idx = 0
        while not self._stop_event.is_set():
            with self._lock:
                if self._label is not None and self._is_tty:
                    elapsed = time.time() - self._stage_start
                    frame = self.FRAMES[idx % len(self.FRAMES)]
                    sys.stdout.write(f"\r\033[K{frame} {self._label}... ({elapsed:.1f}s)")
                    sys.stdout.flush()
                    idx += 1
            time.sleep(0.12)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run closed-loop Gemini 4 Planner-Critic pipeline.")
    parser.add_argument(
        "--robot",
        choices=["franka_emika_panda", "wonik_allegro", "wuji_hand"],
        default="franka_emika_panda",
    )
    parser.add_argument(
        "--task-type",
        choices=["pick_and_lift", "obstacle_grasp", "in_hand_rotation", "in_hand_spindle"],
        default="pick_and_lift",
    )
    parser.add_argument(
        "--object-shape",
        choices=["auto", "box", "cylinder", "sphere"],
        default="auto",
    )
    parser.add_argument("--library-object", type=str, default=None, help="Object ID from backend/library/objects.py (e.g. mug, drawer, valve, button_panel, key).")
    parser.add_argument("--library-task", type=str, default=None, help="Task ID from backend/library/tasks.py (e.g. place_on_pad, open_drawer, turn_joint, press_button, unlock_with_key).")
    parser.add_argument("--library-params-json", type=str, default="{}", help="Optional JSON dict of task params.")
    parser.add_argument("--clutter", nargs="*", default=[], help="Extra library object IDs placed in the scene.")
    parser.add_argument("--layout-seed", type=int, default=None, help="Optional layout shuffle seed for multi-object scenes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--disable", nargs="*", default=[], help="Joints to lock via fault injection.")
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument(
        "--target-rotation",
        type=float,
        default=45.0,
        help="Target cumulative yaw rotation in degrees for Allegro rotation (e.g. 45 free, 180/360 spindle).",
    )
    parser.add_argument(
        "--chunk-deg",
        type=float,
        default=None,
        help="Enable checkpointed closed-loop generation with receding-horizon chunk milestones (e.g. 90 for 360 deg).",
    )
    parser.add_argument("--mode", choices=["one_shot", "text_only", "video"], default="video")
    parser.add_argument("--planner", type=str, default="models/gemini-4-neon")
    parser.add_argument(
        "--thinking",
        choices=["minimal", "low", "medium", "high"],
        default="medium",
        help="Gemini 4 ThinkingConfig level for the Planner (minimal | low | medium | high).",
    )
    parser.add_argument("--critic", type=str, default="models/gemini-4-hydrogen")
    parser.add_argument(
        "--dr-episodes",
        type=int,
        default=0,
        help="If > 0 and run succeeds, replay the verified skill across N domain-randomized scenes.",
    )
    args = parser.parse_args()

    import json as _json

    lib_params = _json.loads(args.library_params_json) if args.library_params_json else {}
    is_custom_prompt = False
    if args.library_object and args.library_task:
        from library.catalog import render_prompt
        default_prompt = render_prompt(args.library_object, args.library_task, lib_params)
        if args.task and args.task.strip() != default_prompt.strip():
            is_custom_prompt = True
        robot_id = args.robot
    elif args.task_type in ("in_hand_rotation", "in_hand_spindle"):
        default_prompt = (
            f"Rotate the multi-colored cube in-hand by at least {args.target_rotation:.0f} degrees "
            "without dropping it."
        )
        robot_id = "wonik_allegro"
    elif args.task_type == "obstacle_grasp":
        default_prompt = (
            "Pick up the object and lift it at least 12 cm above the table while avoiding any obstacles in the workspace."
        )
        robot_id = "franka_emika_panda"
    else:
        default_prompt = "Pick up the object and lift it 15 cm above the table."
        robot_id = args.robot

    req = RunRequest(
        task_prompt=args.task or default_prompt,
        task_type=args.task_type,
        robot_id=robot_id,
        scene_seed=args.seed,
        object_shape=args.object_shape,
        library_object_id=args.library_object,
        library_task_id=args.library_task,
        library_task_params=lib_params,
        library_clutter=args.clutter,
        library_layout_seed=args.layout_seed,
        is_custom_prompt=is_custom_prompt,
        disabled_joints=args.disable,
        max_attempts=args.max_attempts,
        target_rotation_deg=args.target_rotation,
        chunk_step_deg=args.chunk_deg,
        feedback_mode=args.mode,
        planner_model=args.planner,
        thinking_level=args.thinking,
        critic_model=args.critic,
    )

    print(
        f"Starting closed loop | task={req.task_type} robot={req.robot_id} seed={req.scene_seed} "
        f"disabled={req.disabled_joints} mode={req.feedback_mode} "
        f"planner={req.planner_model} thinking={req.thinking_level} critic={req.critic_model}"
    )

    spinner = StageSpinner()
    spinner.start()

    def _print_event(ev: IterationEvent) -> None:
        spinner.pause()
        chunk_tag = f"Chunk {ev.chunk_id} | " if ev.chunk_id is not None else ""
        print(f"\n--- {chunk_tag}Attempt {ev.attempt_id} ({ev.latency_s:.2f}s) ---")
        print(f"Models:       {ev.model_used}")
        print(f"Sim Truth:    sim_success={ev.sim_success} | {ev.telemetry_summary}")
        print(
            f"Decision API: {ev.decision} "
            f"(conf={ev.decision_confidence}, score={ev.grasp_score}/3, mode={ev.failure_mode})"
        )
        if ev.decision_probabilities:
            print(f"Probs:        {ev.decision_probabilities}")
        print(f"Reason:       {ev.reason}")
        print(f"Video:        {ev.video_url}")
        if ev.trajectory_path:
            print(f"Exported:     {ev.trajectory_path}")
        print()

    try:
        summary = run_closed_loop(req, on_event=_print_event, on_status=spinner.set_stage)
    finally:
        spinner.stop()

    print("================ RUN SUMMARY ================")
    print(f"Run ID:            {summary.run_id}")
    print(f"Attempts:          {len(summary.events)}")
    print(f"Final Sim Success: {summary.final_sim_success}")
    print(f"Final Decision:    {summary.final_decision}")
    print(f"Total Latency:     {summary.total_latency_s:.2f}s")
    if summary.stitched_video_path:
        print(f"Stitched Video:    {summary.stitched_video_path}")
    if summary.exported_trajectory_path:
        print(f"Saved Trajectory:  {summary.exported_trajectory_path}")

    if args.dr_episodes > 0 and (summary.final_sim_success or summary.final_decision == "ACCEPT_TRAJECTORY") and summary.events:
        from sim.domain_rand import run_domain_randomization

        print(f"\nRunning Domain Randomization Multiplier ({args.dr_episodes} randomized episodes)...")
        dr = run_domain_randomization(
            waypoints=summary.events[-1].waypoints,
            robot_id=req.robot_id,
            task_type=req.task_type,
            base_seed=req.scene_seed,
            object_shape=req.object_shape,
            disabled_joints=req.disabled_joints,
            target_rotation_deg=req.target_rotation_deg,
            num_episodes=args.dr_episodes,
            library_object_id=req.library_object_id,
            library_task_id=req.library_task_id,
            library_task_params=req.library_task_params,
            library_clutter=req.library_clutter,
            base_layout_seed=req.library_layout_seed,
            is_custom_prompt=req.is_custom_prompt,
        )
        print(
            f"DR Result: {dr.passed_episodes}/{dr.total_episodes} passed "
            f"({dr.transfer_rate * 100:.1f}% zero-shot transfer) in {dr.elapsed_s:.2f}s | "
            f"Exported {len(dr.exported_paths)} verified .npz trajectories! ({dr.output_folder})"
        )
        if dr.sample_video_paths:
            print(f"DR Videos: {', '.join(dr.sample_video_paths)}")


if __name__ == "__main__":
    main()
