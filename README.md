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

## Asset Attribution & Licenses

All robot MJCF models and meshes are from [`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie). Each model carries its own license in its folder:

| Model | License | License File |
| :--- | :--- | :--- |
| `franka_emika_panda` | Apache License, Version 2.0 | `assets/mujoco_menagerie/franka_emika_panda/LICENSE` |
| `wonik_allegro` | BSD 2-Clause License (Copyright (c) 2016, SimLab) | `assets/mujoco_menagerie/wonik_allegro/LICENSE` |
| `robotiq_2f85` | BSD 2-Clause License (Copyright (c) 2013, ROS-Industrial) | `assets/mujoco_menagerie/robotiq_2f85/LICENSE` |
| `universal_robots_ur5e` | BSD 3-Clause License (Copyright 2018 ROS Industrial Consortium) | `assets/mujoco_menagerie/universal_robots_ur5e/LICENSE` |
