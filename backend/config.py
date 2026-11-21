"""Loads config.yaml (repo root) once. Everything tunable lives there."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with open(REPO_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def task_thresholds(task_id: str) -> dict[str, float]:
    return {k: float(v) for k, v in load_config()["tasks"][task_id].items()}
