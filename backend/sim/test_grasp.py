"""Phase 2 Milestone CLI: test damped least-squares IK grasp & lift with ground-truth check.

Usage:
  python backend/sim/test_grasp.py --robot franka_emika_panda --seed 42
  python backend/sim/test_grasp.py --robot franka_emika_panda --seed 42 --disable joint4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sim.executor import default_panda_waypoints, execute_rollout
from sim.scene_builder import build_scene, summarize_kinematics


def main() -> None:
    parser = argparse.ArgumentParser(description="Test IK grasp & lift rollout in MuJoCo.")
    parser.add_argument(
        "--robot",
        choices=["franka_emika_panda", "wonik_allegro"],
        default="franka_emika_panda",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable",
        nargs="*",
        default=[],
        help="Optional joint names to lock via fault injection (e.g. joint4).",
    )
    args = parser.parse_args()

    t0 = time.time()
    bundle = build_scene(
        robot_id=args.robot,
        scene_seed=args.seed,
        disabled_joints=args.disable,
    )
    print(summarize_kinematics(bundle))

    waypoints = default_panda_waypoints()
    fault_tag = f"_fault_{'_'.join(args.disable)}" if args.disable else ""
    out_video = REPO_ROOT / "outputs" / f"rollout_{args.robot}_seed{args.seed}{fault_tag}.mp4"

    result = execute_rollout(bundle, waypoints, video_path=out_video)
    elapsed = time.time() - t0

    print("\n--- ROLLOUT RESULT ---")
    print(result.telemetry_summary)
    print(f"Elapsed: {elapsed:.2f}s | Video: {result.video_path}")


if __name__ == "__main__":
    main()
