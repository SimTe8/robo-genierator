"""Runs one planner job in its own process.

    python backend/run_worker.py outputs/jobs/<job_id>

A separate process keeps MuJoCo rendering on a main thread (required on macOS) and the API responsive.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import gl_setup  # noqa: E402,F401  (must precede mujoco)

import traceback  # noqa: E402

from planner_hub import EventLog, run_job  # noqa: E402
from schemas import PlannerBrief  # noqa: E402


def main() -> None:
    job_dir = Path(sys.argv[1]).resolve()
    try:
        run_job(job_dir)
    except Exception as exc:  # surface any crash to the UI instead of a silent stream
        (job_dir / "error.log").write_text(traceback.format_exc())
        brief = PlannerBrief.model_validate_json((job_dir / "job.json").read_text())
        EventLog(job_dir, brief.job_id)("error", f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
