/**
 * Shared TypeScript contracts matching backend/schemas.py.
 */

export type RobotId = 'franka_emika_panda' | 'wonik_allegro';

export type DecisionType = 'ACCEPT_TRAJECTORY' | 'REFINE_GRASP' | 'GIVE_UP';

export type FeedbackMode = 'one_shot' | 'text_only' | 'video';

export interface Waypoint {
  phase: 'pre_grasp' | 'grasp' | 'close' | 'lift';
  position_offset: [number, number, number];
  orientation_rpy: [number, number, number];
  gripper_target: number;
  finger_targets?: Record<string, number> | null;
}

export interface RunRequest {
  task_prompt: string;
  robot_id: RobotId;
  scene_seed: number;
  disabled_joints: string[];
  max_attempts: number;
  feedback_mode: FeedbackMode;
}

export interface IterationEvent {
  attempt_id: number;
  model_used: string;
  waypoints: Waypoint[];
  video_url: string;
  telemetry_summary: string;
  sim_success: boolean;
  decision: DecisionType;
  reason: string;
  latency_s: number;
}

// ---------------------------------------------------------------------------
// Web UI -> Gemini 4 planner hand-off (mirrors the bottom of backend/schemas.py)
// ---------------------------------------------------------------------------

export interface TaskParamInfo {
  key: string;
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

export interface TaskPresetInfo {
  id: string;
  name: string;
  skill: string;
  summary: string;
  goal: string;
  fixture: string | null;
  initial_pose: string;
  params: TaskParamInfo[];
  prompt_template: string;
  criteria: string[];
  robots: string[];
}

export interface ObjectTaskInfo {
  task_id: string;
  name: string;
  params: Record<string, number>;
}

export interface ObjectInfo {
  id: string;
  name: string;
  category: string;
  description: string;
  dims_cm: number[];
  mass_kg: number;
  grasp_note: string;
  parts: string[];
  articulated: boolean;
  thumbnail_url: string;
  tasks: ObjectTaskInfo[];
  /** PG-VLM physical concepts */
  physics: {
    material: string;
    fragility: string;
    deformability: string;
    transparency: string;
    contents: string;
    can_contain_liquid: boolean;
    sealed: boolean;
  };
}

export interface RobotInfo {
  id: string;
  name: string;
  kind: string;
  description: string;
  fingers: string[];
  license: string;
}

export interface Catalog {
  robots: RobotInfo[];
  objects: ObjectInfo[];
  tasks: TaskPresetInfo[];
}

export interface PlannerRequest {
  robot_id: string;
  object_id: string;
  task_id: string;
  task_params: Record<string, number>;
  prompt: string;
}

export interface PlannerBrief {
  job_id: string;
  created_at: string;
  robot: RobotInfo;
  prompt: string;
  object: ObjectInfo;
  task: TaskPresetInfo;
  task_params: Record<string, number>;
  success_criteria: string[];
  default_scene: Record<string, unknown>;
  kinematic_summary: string;
  how_to_build_scene: string;
}

export interface CheckResult {
  success: boolean;
  checks: Record<string, boolean>;
  metrics: Record<string, number>;
}

export interface JobEvent {
  job_id: string;
  seq: number;
  kind: 'status' | 'attempt' | 'done' | 'error';
  message: string;
  iteration: IterationEvent | null;
  check: CheckResult | null;
  data: Record<string, unknown> | null;
}

/** POST /api/scene */
export interface SceneResponse {
  scene: {
    object_position: number[];
    object_yaw_deg: number;
    object_mass_kg: number;
    table_top_z: number;
    fixture: string | null;
    [key: string]: unknown;
  };
  preview_url: string;
  prompt: string;
}

export interface Health {
  mujoco: string;
  planner: string | null;
}

// ---- Side quest: Gemini asset generation -----------------------------------

export interface AssetsStatus {
  configured: boolean;
  model: string;
}

export interface AssetDraft {
  draft_id: string;
  object_id: string;
  prompt: string;
  model: string;
  attempts: number;
  warnings: string[];
  thumbnail_url: string;
  sim: { problems: string[]; settled_drift_cm: number; task_used: string };
  object: ObjectInfo;
}
