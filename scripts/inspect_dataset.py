#!/usr/bin/env python3
"""Inspect, validate, and summarize exported `.npz` robot trajectory datasets.

Usage:
    python scripts/inspect_dataset.py [path_or_dir] [--check] [--export-summary out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO_ROOT / "outputs" / "trajectories"

TIME_SERIES_KEYS = (
    "observation_qpos",
    "observation_qvel",
    "action_ctrl",
    "privileged_obj_pos",
    "ee_pose",
    "gripper_state",
    "timestamps_s",
)
REQUIRED_KEYS = ("observation_qpos", "action_ctrl", "privileged_obj_pos")


def _infer_robot_from_dims(nq: int, nu: int) -> str:
    if nu == 8:
        return "franka_emika_panda"
    if nu == 16:
        return "wonik_allegro"
    if nu == 20:
        return "wuji_hand"
    return f"unknown(nq={nq},nu={nu})"


def inspect_npz(path: Path, manifest_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inspect a single `.npz` episode file and run integrity checks."""
    issues: list[str] = []
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = list(data.files)
            arrays = {k: data[k] for k in keys}
    except Exception as exc:  # noqa: BLE001
        return {
            "file": str(path),
            "batch": path.parent.name,
            "valid": False,
            "issues": [f"failed to load npz: {exc}"],
        }

    for req in REQUIRED_KEYS:
        if req not in arrays:
            issues.append(f"missing required key '{req}'")

    if issues:
        return {
            "file": str(path),
            "batch": path.parent.name,
            "valid": False,
            "issues": issues,
        }

    qpos = arrays["observation_qpos"]
    act = arrays["action_ctrl"]
    obj_pos = arrays["privileged_obj_pos"]
    qvel = arrays.get("observation_qvel")
    ee_pose = arrays.get("ee_pose")
    grip = arrays.get("gripper_state")

    t_len = int(qpos.shape[0]) if qpos.ndim >= 1 else 0
    nq = int(qpos.shape[1]) if qpos.ndim == 2 else -1
    nv = int(qvel.shape[1]) if (qvel is not None and qvel.ndim == 2) else None
    nu = int(act.shape[1]) if act.ndim == 2 else -1
    ee_dim = int(ee_pose.shape[1]) if (ee_pose is not None and ee_pose.ndim == 2) else None
    grip_dim = int(grip.shape[1]) if (grip is not None and grip.ndim == 2) else None

    if t_len < 2:
        issues.append(f"horizon T={t_len} < 2")

    for k in TIME_SERIES_KEYS:
        if k in arrays:
            arr = arrays[k]
            if arr.ndim == 0 or arr.shape[0] != t_len:
                issues.append(f"key '{k}' length {arr.shape} != T={t_len}")

    # Check finite values (no NaN / Inf) across all numeric arrays
    for k, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.number):
            if not np.all(np.isfinite(arr)):
                issues.append(f"NaN/Inf detected in '{k}'")

    # Check non-zero motion across trajectory
    if t_len >= 2 and np.all(np.isfinite(qpos)) and np.all(np.isfinite(act)) and np.all(np.isfinite(obj_pos)):
        qpos_span = float(np.ptp(qpos))
        act_span = float(np.ptp(act))
        obj_span = float(np.ptp(obj_pos))
        max_span = max(qpos_span, act_span, obj_span)
        if max_span <= 1e-6:
            issues.append(f"zero motion detected (max span={max_span:.2e})")
    else:
        qpos_span = act_span = obj_span = 0.0

    dt_val = float(arrays["dt"]) if "dt" in arrays else 0.04
    freq_hz = round(1.0 / dt_val, 2) if dt_val > 0 else 0.0

    if "robot_id" in arrays:
        robot_id = str(arrays["robot_id"].item())
    elif manifest_meta and manifest_meta.get("robot_id"):
        robot_id = str(manifest_meta["robot_id"])
    else:
        robot_id = _infer_robot_from_dims(nq, nu)

    task_label = None
    if "task_id" in arrays:
        task_label = str(arrays["task_id"].item())
    elif manifest_meta:
        task_label = manifest_meta.get("library_task_id") or manifest_meta.get("task_type")

    return {
        "file": str(path),
        "batch": path.parent.name,
        "valid": len(issues) == 0,
        "issues": issues,
        "robot_id": robot_id,
        "task": task_label or "pick_and_lift",
        "T": t_len,
        "dt": round(dt_val, 5),
        "freq_hz": freq_hz,
        "nq": nq,
        "nv": nv,
        "nu": nu,
        "ee_dim": ee_dim,
        "grip_dim": grip_dim,
        "richer_schema": all(
            k in arrays
            for k in (
                "observation_qvel",
                "ee_pose",
                "gripper_state",
                "timestamps_s",
                "dt",
                "robot_id",
                "joint_names",
                "actuator_names",
            )
        ),
        "qpos_ptp": round(qpos_span, 4),
        "obj_ptp_m": round(obj_span, 4),
        "keys": keys,
    }


def collect_npz_files(target: Path) -> list[Path]:
    if target.is_file() and target.suffix == ".npz":
        return [target]
    if target.is_dir():
        return sorted(target.rglob("*.npz"))
    return []


def format_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    hdr = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    body = ["| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)) + " |" for r in rows]
    return "\n".join([hdr, sep, *body])


LOADER_SNIPPETS = """
### Downstream Policy Learning Quickstart (PyTorch / Diffusion Policy / LeRobot)

```python
import glob, numpy as np, torch
from torch.utils.data import Dataset

class SimTrajectoryDataset(Dataset):
    \"\"\"Loads verified `.npz` rollouts for Behavior Cloning / Diffusion Policy.\"\"\"
    def __init__(self, folder: str):
        self.samples = []
        for path in sorted(glob.glob(f"{folder}/**/*.npz", recursive=True)):
            with np.load(path, allow_pickle=False) as ep:
                qpos = ep["observation_qpos"]                          # (T, nq) float32
                qvel = ep.get("observation_qvel", np.zeros_like(qpos)) # (T, nv) float32
                ee   = ep.get("ee_pose")                               # (T, 7)  [x,y,z,qw,qx,qy,qz]
                grip = ep.get("gripper_state")                         # (T, 1)  [0..1]
                act  = ep["action_ctrl"]                               # (T, nu) float32
                state_parts = [qpos, qvel] + ([ee, grip] if ee is not None and grip is not None else [])
                obs_state = np.concatenate(state_parts, axis=-1)
                for t in range(len(act)):
                    self.samples.append((obs_state[t], act[t]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        obs, act = self.samples[idx]
        # Compatible with Diffusion Policy (`obs`, `action`) & LeRobot (`observation.state`, `action`)
        return {
            "observation.state": torch.from_numpy(obs),
            "action": torch.from_numpy(act),
        }
```
""".strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect and validate exported MuJoCo `.npz` robot trajectory datasets."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_TARGET),
        help="Path to a `.npz` file or directory of trajectories (default: outputs/trajectories)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Strict integrity mode: exit non-zero if no trajectories exist or any file is corrupt/NaN/static.",
    )
    parser.add_argument(
        "--export-summary",
        metavar="OUT_JSON",
        default=None,
        help="Optional path to write a structured JSON dataset summary.",
    )
    args = parser.parse_args(argv)

    target = Path(args.path).expanduser().resolve()
    npz_files = collect_npz_files(target)

    if not npz_files:
        print(f"No `.npz` trajectory files found at: {target}")
        return 1 if args.check else 0

    # Cache manifest.json per folder if available
    manifests: dict[Path, dict[str, Any]] = {}
    for f in npz_files:
        mpath = f.parent / "manifest.json"
        if f.parent not in manifests:
            if mpath.is_file():
                try:
                    manifests[f.parent] = json.loads(mpath.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    manifests[f.parent] = {}
            else:
                manifests[f.parent] = {}

    records = [inspect_npz(f, manifests.get(f.parent)) for f in npz_files]
    valid_records = [r for r in records if r["valid"]]
    invalid_records = [r for r in records if not r["valid"]]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[rec["batch"]].append(rec)

    rows: list[list[str]] = []
    batch_summaries: list[dict[str, Any]] = []
    for batch_name, eps in sorted(grouped.items()):
        ok_eps = [e for e in eps if e["valid"]]
        robots = sorted({e.get("robot_id", "unknown") for e in ok_eps}) or ["unknown"]
        tasks = sorted({str(e.get("task", "unknown")) for e in ok_eps}) or ["unknown"]
        if ok_eps:
            t_vals = [int(e["T"]) for e in ok_eps]
            t_str = f"{min(t_vals)}..{max(t_vals)}" if min(t_vals) != max(t_vals) else f"{t_vals[0]}"
            hz_str = f"{ok_eps[0]['freq_hz']:.1f} Hz"
            nq = ok_eps[0]["nq"]
            nv = ok_eps[0]["nv"]
            nu = ok_eps[0]["nu"]
            ee_d = ok_eps[0]["ee_dim"]
            state_str = f"qpos:{nq}" + (f", qvel:{nv}" if nv is not None else "") + (f", ee:{ee_d}" if ee_d else "")
            act_str = f"ctrl:{nu}" + (", grip:1" if ok_eps[0]["grip_dim"] else "")
            status = "PASS" if len(ok_eps) == len(eps) else f"FAIL ({len(eps) - len(ok_eps)} bad)"
        else:
            t_str = "-"
            hz_str = "-"
            state_str = "-"
            act_str = "-"
            status = f"FAIL ({len(eps)} bad)"

        rows.append(
            [
                batch_name,
                f"{len(ok_eps)}/{len(eps)}",
                ",".join(robots),
                ",".join(tasks),
                t_str,
                hz_str,
                state_str,
                act_str,
                status,
            ]
        )
        batch_summaries.append(
            {
                "batch": batch_name,
                "total_files": len(eps),
                "valid_files": len(ok_eps),
                "robots": robots,
                "tasks": tasks,
                "horizon_T_min": min((int(e["T"]) for e in ok_eps), default=0),
                "horizon_T_max": max((int(e["T"]) for e in ok_eps), default=0),
                "control_hz": ok_eps[0]["freq_hz"] if ok_eps else 0.0,
                "richer_schema": all(e.get("richer_schema", False) for e in ok_eps) if ok_eps else False,
            }
        )

    headers = ["Batch", "Valid/Eps", "Robot", "Task", "Horizon T", "Freq", "State Dims", "Action Dims", "Check"]
    total_steps = sum(int(r["T"]) for r in valid_records)

    print(f"## Trajectory Dataset Summary (`{target}`)")
    print(
        f"- **Episodes**: {len(valid_records)}/{len(records)} valid across {len(grouped)} batch(es) "
        f"({total_steps} total control steps)\n"
    )
    print(format_markdown_table(headers, rows))

    if invalid_records:
        print("\n### Integrity Failures")
        for bad in invalid_records:
            print(f"- `{bad['file']}`: {'; '.join(bad['issues'])}")

    print("\n" + LOADER_SNIPPETS)

    if args.export_summary:
        out_path = Path(args.export_summary).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "target": str(target),
            "total_episodes": len(records),
            "valid_episodes": len(valid_records),
            "invalid_episodes": len(invalid_records),
            "total_timesteps": total_steps,
            "all_passed": len(invalid_records) == 0,
            "batches": batch_summaries,
            "episodes": records,
        }
        out_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        print(f"\nExported JSON summary to `{out_path}`.")

    if args.check and invalid_records:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
