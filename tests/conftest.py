"""Shared pytest configuration: headless EGL setup, backend import path, and live-API test gating."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Must precede any mujoco import across all test modules
os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pytest  # noqa: E402


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip @pytest.mark.live tests unless RUN_LIVE_GEMINI=1 is explicitly set."""
    if os.environ.get("RUN_LIVE_GEMINI") == "1":
        return
    skip_live = pytest.mark.skip(reason="Live Gemini API tests skipped unless RUN_LIVE_GEMINI=1")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
