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

### 3. Public Robot Assets (`mujoco_menagerie`)
Robot models are **not** committed to this repository. Download them into `assets/mujoco_menagerie/`:
```bash
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git assets/mujoco_menagerie
```

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

### Run Closed-Loop Gemini 4 Planner + Video Critic + Decisions API (Phase 3)
```bash
# Healthy Panda with Carbon (Planner) + Neon (Video Critic + Decisions API)
python backend/agents/run_loop.py --robot franka_emika_panda --seed 42 \
  --planner models/gemini-4-carbon --critic models/gemini-4-neon --mode video

# Live fault injection (lock joint4) with up to 4 self-correction attempts
python backend/agents/run_loop.py --robot franka_emika_panda --seed 42 --disable joint4 \
  --planner models/gemini-4-carbon --critic models/gemini-4-neon --mode video --max-attempts 4

# Compare text-only feedback or one-shot baseline
python backend/agents/run_loop.py --robot franka_emika_panda --seed 42 --mode text_only
python backend/agents/run_loop.py --robot franka_emika_panda --seed 42 --mode one_shot

# Dexterous In-Hand Cube Rotation on Wonik Allegro (16-DOF palm-up hand)
python backend/agents/run_loop.py --task-type in_hand_rotation --seed 42 --target-rotation 60 \
  --planner models/gemini-4-carbon --critic models/gemini-4-neon --mode video --max-attempts 4

# Full 360-degree cyclic in-hand cube rotation stress test
python backend/agents/run_loop.py --task-type in_hand_rotation --seed 42 --target-rotation 360 \
  --planner models/gemini-4-carbon --critic models/gemini-4-hydrogen --mode video --max-attempts 6

# In-Hand Cube Rotation with a disabled middle finger
python backend/agents/run_loop.py --task-type in_hand_rotation --seed 42 --disable mfj1 mfj2 \
  --planner models/gemini-4-carbon --critic models/gemini-4-neon --mode video --max-attempts 4
```

## Asset Attribution & Licenses

All robot MJCF models and meshes are from [`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie). Each model carries its own license in its folder:

| Model | License | License File |
| :--- | :--- | :--- |
| `franka_emika_panda` | Apache License, Version 2.0 | `assets/mujoco_menagerie/franka_emika_panda/LICENSE` |
| `wonik_allegro` | BSD 2-Clause License (Copyright (c) 2016, SimLab) | `assets/mujoco_menagerie/wonik_allegro/LICENSE` |
| `robotiq_2f85` | BSD 2-Clause License (Copyright (c) 2013, ROS-Industrial) | `assets/mujoco_menagerie/robotiq_2f85/LICENSE` |
| `universal_robots_ur5e` | BSD 3-Clause License (Copyright 2018 ROS Industrial Consortium) | `assets/mujoco_menagerie/universal_robots_ur5e/LICENSE` |
