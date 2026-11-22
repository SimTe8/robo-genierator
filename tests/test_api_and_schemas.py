"""Tests for Pydantic contracts, FastAPI endpoints, and optional live Gemini 4 smoke test."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

import app as app_module
from schemas import PlannerSettings, RunRequest, Waypoint


def test_pydantic_schemas_defaults_and_validation() -> None:
    wp = Waypoint(phase="pre_grasp", position_offset=[0.0, 0.0, 0.12])
    assert wp.reference_frame == "target_object"
    assert wp.gripper_target == 1.0
    assert wp.duration_s == 0.6
    assert len(wp.orientation_rpy) == 3

    with pytest.raises(ValidationError):
        Waypoint(phase="pre_grasp", position_offset=[0.0, 0.0])

    with pytest.raises(ValidationError):
        Waypoint(phase="close", position_offset=[0.0, 0.0, 0.0], gripper_target=1.5)

    req = RunRequest(library_object_id="mug", library_task_id="place_on_pad", library_clutter=["can"])
    assert req.robot_id == "franka_emika_panda"
    assert req.task_type == "pick_and_lift"
    assert req.feedback_mode == "video"
    assert req.max_attempts == 4
    assert req.library_object_id == "mug"
    assert req.library_clutter == ["can"]

    settings = PlannerSettings(max_attempts=6, dr_episodes=5, thinking_level="high")
    assert settings.max_attempts == 6
    assert settings.dr_episodes == 5
    assert settings.thinking_level == "high"
    assert settings.task_type_override == "auto"
    assert settings.object_shape_override == "auto"

    with pytest.raises(ValidationError):
        PlannerSettings(max_attempts=0)


def test_fastapi_endpoints_health_catalog_scene_and_assets_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview_dir = tmp_path / "previews"
    monkeypatch.setattr(app_module, "PREVIEW_DIR", preview_dir)

    with TestClient(app_module.app) as client:
        # 1. GET /api/health
        r_health = client.get("/api/health")
        assert r_health.status_code == 200
        health_data = r_health.json()
        assert "mujoco" in health_data and len(health_data["mujoco"]) > 0
        assert "planner_mode" in health_data

        # 2. GET /api/catalog
        r_cat = client.get("/api/catalog")
        assert r_cat.status_code == 200
        cat_data = r_cat.json()
        assert len(cat_data["robots"]) == 3
        robot_ids = {r["id"] for r in cat_data["robots"]}
        assert robot_ids == {"franka_emika_panda", "wonik_allegro", "wuji_hand"}
        assert len(cat_data["objects"]) >= 24
        assert len(cat_data["tasks"]) >= 30

        # 3. POST /api/scene (renders preview PNG)
        r_scene = client.post(
            "/api/scene",
            json={
                "robot_id": "franka_emika_panda",
                "object_id": "mug",
                "task_id": "place_on_pad",
            },
        )
        assert r_scene.status_code == 200
        scene_data = r_scene.json()
        assert "scene" in scene_data
        assert "zone_center" in scene_data["scene"]
        assert scene_data["preview_url"].startswith("/outputs/previews/")
        assert len(scene_data["prompt"]) > 10

        rendered_png = preview_dir / Path(scene_data["preview_url"]).name
        assert rendered_png.is_file()
        png_bytes = rendered_png.read_bytes()
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(png_bytes) > 1000

        # Invalid robot/task combo returns 422
        r_bad = client.post(
            "/api/scene",
            json={
                "robot_id": "franka_emika_panda",
                "object_id": "cube",
                "task_id": "rotate_in_hand",
            },
        )
        assert r_bad.status_code == 422

        # 4. GET /api/assets/status
        r_assets = client.get("/api/assets/status")
        assert r_assets.status_code == 200
        assets_data = r_assets.json()
        assert isinstance(assets_data["configured"], bool)
        assert isinstance(assets_data["model"], str) and len(assets_data["model"]) > 0

        # 5. GET /api/benchmark & GET /api/datasets (Research Studio endpoints)
        r_bench = client.get("/api/benchmark")
        assert r_bench.status_code == 200
        bench_data = r_bench.json()
        assert isinstance(bench_data["available"], bool)
        assert "report" in bench_data

        r_ds = client.get("/api/datasets")
        assert r_ds.status_code == 200
        ds_data = r_ds.json()
        assert "total_npz_files" in ds_data
        assert "dr_batches" in ds_data
        assert "recent_trajectories" in ds_data


@pytest.mark.live
def test_live_gemini_closed_loop_single_attempt_smoke(tmp_path: Path) -> None:
    from agents.loop import run_closed_loop

    req = RunRequest(
        robot_id="franka_emika_panda",
        task_type="pick_and_lift",
        scene_seed=42,
        max_attempts=1,
        feedback_mode="video",
    )
    summary = run_closed_loop(req, output_dir=tmp_path)

    assert len(summary.events) == 1
    assert summary.events[0].attempt_id == 1
    assert len(summary.events[0].waypoints) >= 2
    assert summary.final_decision in ("ACCEPT_TRAJECTORY", "REFINE_GRASP", "GIVE_UP")
