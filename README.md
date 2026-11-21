# Zero-Shot Cross-Embodiment Trajectory Generator & Visual Self-Correction Loop

Built for the **Gemini 4 NYC Hackathon** (November 21, 2026) — Problem Statement Two (Element 119: media as the core input + Decisions API + real-time closed loop).

> **Pre-Event Preparation Disclosure:** Before 10:30 AM only the environment was prepared: Python and npm packages installed, the public `mujoco_menagerie` models downloaded, an unmodified Vite template created. Every source file in this repository was written during the hackathon.

## Overview

Given a natural-language manipulation task, a robot model (MJCF from `mujoco_menagerie`), a procedurally seeded object, and optional live fault injection (disabled joints or fingers), the system:
1. Summarizes the active kinematic structure from MuJoCo.
2. Uses a **Gemini 4 Planner** model to propose object-centric grasp waypoints and finger/gripper targets.
3. Executes the trajectory via damped least-squares IK in headless MuJoCo (EGL) and renders a slow-motion multi-view MP4 (side + top).
4. Uses a **Gemini 4 Critic** model + **Decisions API** watching the rollout video to accept (`ACCEPT_TRAJECTORY`), refine (`REFINE_GRASP` with structured parameter adjustments), or stop (`GIVE_UP`).
5. Verifies all results against simulator ground truth and exports verified demonstrations.

## Setup

### 1. Python Environment
```bash
conda create -n gemini4nyc python=3.11 -y
conda activate gemini4nyc
pip install -r backend/requirements.txt
```

### 2. Environment Variables
Copy `.env.example` to `.env` in the repository root and add your Gemini API key:
```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=...
```

### 3. Robot Models (one command)
Robot models are **not** committed to this repository. Download them (Franka Panda + Allegro from `mujoco_menagerie`, and the Wuji hand) with:
```bash
python scripts/setup_assets.py --check   # ~120 MB, pinned commits; --check compiles each robot in MuJoCo
```
Standard library only (runs before `pip install`), works on macOS / Linux / Windows, uses git when available (sparse checkout: only the menagerie folders we use) and otherwise downloads a GitHub zip. Re-running skips what is already there; `--force` re-downloads, `--only wuji` fetches just the Wuji hand.

### 4. Frontend
```bash
cd frontend
npm install
npm run dev
```

## Useful Commands

### Visualize Robots & Seeded Objects (MuJoCo)
```bash
# Render side+top composite PNG and MP4 previews for both robots (headless EGL)
python backend/sim/visualize.py --robot all --seed 42

# Render a single robot with a different procedural seed
python backend/sim/visualize.py --robot franka_emika_panda --seed 7

# Launch interactive MuJoCo 3D viewer (requires display / WSLg)
python backend/sim/visualize.py --robot franka_emika_panda --seed 42 --mode viewer
```

### Run Baseline IK Grasp & Lift Rollout (Phase 2 Milestone)
```bash
# Execute a healthy Panda grasp & lift with slow-motion video and simulator verification
python backend/sim/test_grasp.py --robot franka_emika_panda --seed 42

# Test live fault injection (e.g., lock joint4)
python backend/sim/test_grasp.py --robot franka_emika_panda --seed 42 --disable joint4
```

## Web Interface → Gemini 4 Planner

Pick a **robot**, an **object** and one of its **task presets**, edit the prompt, and send it to the Gemini 4 planner. The UI sets no conditions — the planner owns domain randomisation and trajectory generation.

```bash
# Terminal 1 — backend (repo root)
uvicorn app:app --app-dir backend --port 8000
# Terminal 2 — frontend
cd frontend && npm run dev     # http://localhost:5173 (proxies /api and /outputs to :8000)
```

- **Robots**: Allegro hand and Wuji hand (multi-finger, free palm on a mocap weld named `palm_mocap`) and Franka Panda (arm + parallel gripper). In-hand tasks are offered for the hands only.
- **Objects** (`backend/library/objects.py`, 24 + generated, built from MuJoCo primitives — no third-party meshes): in-hand (cube, die, egg, pen, T/L-blocks), insertion (key + lock, round/square pegs + shape sorter), tools (power screwdriver with a spring-loaded button, hammer + nail, screwdriver), physical reasoning (cup of beads + bowl, three look-alike cans of which one is full, mug, bottle, ball, can, card), articulated (button panel, drawer, valve, dial, jar lid).
- **Physical concepts** (after [PG-VLM](https://iliad.stanford.edu/pg-vlm/)): every object carries material, fragility, deformability, transparency, contents, can-contain-liquid and sealed. They are shown in the UI and sent to the planner.
- **Task presets** (`backend/library/tasks.py`, 31): e.g. insert & turn a key (the lock only turns through the key), insert a peg into the *matching* hole, hold a power screwdriver, press its button and drive a screw, press two buttons at once, open a drawer and put a block inside, pour beads into a bowl, find and lift the heaviest of three identical cans, plus grasp / in-hand / place / push tasks.
- **Multi-object scenes** (`backend/library/multi.py`): five presets spawn their own extra bodies — *stack into a tower* (2–4 copies), *stack on the block*, *put them all in the bin*, *sort by colour* (red/blue copies onto red/blue pads) and *pick it out of the clutter* (target packed among other library objects). In the **Scene objects** panel you can also add up to 5 library objects as clutter to *any* task (`clutter=[...]` in `build_task_scene`, `PlannerRequest.clutter`). Extras are placed deterministically on free table spots, named by role (`stack_1`, `bin_item_1`, `sort_1`, `base_block`, `other_1`, `clutter_1`), and listed with live positions in the brief's `default_scene.other_objects`.
- **Shuffle layout** (`library/multi.py → plan_layout`, ranges in `config.yaml → layout`): before sending, *Shuffle layout* moves the target (within ±8 cm of the workspace origin) and every other object (and turns them, except articulated objects and *turn part toward hand*, where a random yaw could already solve the task). It is driven by a seed, so the same `layout_seed` always rebuilds the same scene: the preview, `POST /api/scene` and the planner brief (`PlannerBrief.layout_seed`, `build_task_scene(..., layout_seed=…)`) all agree. Objects never overlap each other or the fixtures; if anything would start touching the robot, the same seed re-draws deterministically. *Default* returns to the fixed layout. The planner's own `object_xy` / `object_yaw_deg` still override the target's pose.
- **Simulator ground truth** (`backend/sim/check.py`): `snapshot()` before, `evaluate()` after; thresholds in `config.yaml → checks`.
- **Hand-off** (`backend/planner_hub.py`): `POST /api/planner/jobs` writes `outputs/jobs/<job_id>/job.json` (robot, prompt, object + physics, task, params, success criteria, default scene, kinematic summary) and starts the planner in a worker process; progress streams to the UI over SSE.
- **Plugging in the planner**: expose `PLANNER` in `backend/agents/planner.py` with `name` and `run(brief, job_dir, emit)`. Build scenes with your own conditions via `library.scene.build_task_scene(object_id, task_id, params, robot_id=…, object_xy=…, object_yaw_deg=…, size_scale=…, mass_scale=…, friction=…, disabled_joints=…, clutter=brief.clutter)`; execute with `sim.executor.execute_rollout`; score with `sim.check`.
- **Wuji hand model** (MIT, not committed): downloaded by `python scripts/setup_assets.py` (pinned to `wuji-technology/wuji-hand-description@bbc2f64`).
- **MuJoCo 4 / macOS notes**: joint `damping` and `stiffness` are 3-vectors in `MjSpec`; menagerie scenes compile angles in **degrees** (Wuji in radians) — `library/scene.py` handles both. No EGL on macOS: the backend picks `MUJOCO_GL=cgl` (`backend/gl_setup.py`); interactive viewers need `mjpython`.

## Side Quest: Generate Objects with Gemini

Describe an object ("a padlock with a U-shaped shackle") in the **Side quest** panel or the CLI. Gemini returns a JSON spec (MuJoCo primitives, optional spring buttons, PG-VLM physical properties, fitting task presets) — never code. The backend validates and re-centres it (one retry with the error fed back), builds it with the hand, checks it rests stably, renders a thumbnail, and on **Add to library** saves it to `backend/library/generated/<id>.json`, where it joins the object library (category *Generated*).

```bash
# GEMINI_API_KEY goes in the repo-root .env (gitignored) — never in .env.example
python backend/sidequest/forge_cli.py "a TV remote with a power button" --save
```

- Model: `config.yaml → models.asset_generator`. The key is re-read from `.env` on every call, so pasting it while the server runs works without a restart. All Gemini calls go through `backend/agents/gemini_client.py` (the only SDK import; logs to `outputs/logs/gemini_calls.jsonl`). The planner/critic should extend that file.

## Asset Attribution & Licenses

All robot MJCF models and meshes are from [`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie). Each model carries its own license in its folder:

| Model | License | License File |
| :--- | :--- | :--- |
| `franka_emika_panda` | Apache License, Version 2.0 | `assets/mujoco_menagerie/franka_emika_panda/LICENSE` |
| `wonik_allegro` | BSD 2-Clause License (Copyright (c) 2016, SimLab) | `assets/mujoco_menagerie/wonik_allegro/LICENSE` |
| `robotiq_2f85` | BSD 2-Clause License (Copyright (c) 2013, ROS-Industrial) | `assets/mujoco_menagerie/robotiq_2f85/LICENSE` |
| `universal_robots_ur5e` | BSD 3-Clause License (Copyright 2018 ROS Industrial Consortium) | `assets/mujoco_menagerie/universal_robots_ur5e/LICENSE` |
| Wuji hand (`wuji-technology/wuji-hand-description`) | MIT License (Copyright (c) 2025 Wuji Technology) | `assets/wuji-hand-description/LICENSE` |
