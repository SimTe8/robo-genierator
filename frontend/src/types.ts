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
