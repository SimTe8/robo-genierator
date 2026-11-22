import type { AssetDraft, AssetsStatus, Catalog, Health, JobEvent, PlannerBrief, PlannerRequest, SceneResponse } from './types';

export interface DatasetTrajectoryItem {
  filename: string;
  rel_path?: string;
  download_url: string;
  size_kb: number;
  mtime?: number;
  steps: number;
  qpos_dim: number;
  ctrl_dim: number;
  keys: string[];
  robot_id: string;
}

export interface DatasetDrBatch {
  batch_id: string;
  rel_dir?: string;
  robot_id: string;
  task: string;
  object_id?: string | null;
  npz_count: number;
  verified_episodes: number;
  attempted_episodes: number;
  success_rate: number;
  sample_video_urls: string[];
  manifest_url?: string | null;
  mtime?: number;
}

export interface DatasetExplorerResponse {
  total_npz_files: number;
  dr_batches: DatasetDrBatch[];
  recent_trajectories: DatasetTrajectoryItem[];
}

export interface BenchmarkArmSummary {
  episodes?: number;
  successes?: number;
  success_rate?: number;
  wilson_ci?: [number, number] | null;
  ci_95?: [number, number] | null;
  avg_attempts?: number;
  avg_total_latency_s?: number;
  avg_iter_latency_s?: number;
  critic_sim_agreement?: number | null;
  false_accept_rate?: number | null;
  false_reject_rate?: number | null;
  eval_attempts?: number;
  false_accepts?: number;
  false_rejects?: number;
}

export interface BenchmarkDrEntry {
  label: string;
  task_type: string;
  robot_id?: string;
  dr_transfer: {
    passed: number;
    total: number;
    transfer_rate: number;
    elapsed_s?: number;
    exported_count?: number;
  };
}

export interface BenchmarkEpisode {
  suite?: string;
  label?: string;
  run_id?: string;
  robot_id?: string;
  task_type?: string;
  feedback_mode?: string;
  planner_model?: string;
  critic_model?: string;
  duo?: string;
  attempts?: number;
  sim_success?: boolean;
  final_decision?: string;
  total_latency_s?: number;
  avg_iter_latency_s?: number;
  eval_attempts?: number;
  critic_agreements?: number;
  false_accepts?: number;
}

export interface BenchmarkReport {
  generated_at?: string;
  total_wall_time_s?: number;
  total_episodes?: number;
  decisions_model?: string;
  summary_by_arm?: Record<string, BenchmarkArmSummary>;
  summary_by_task_and_arm?: Record<string, Record<string, BenchmarkArmSummary>>;
  summary_by_duo?: Record<string, BenchmarkArmSummary>;
  critic_calibration?: Record<string, unknown>;
  calibration?: Record<string, unknown>;
  domain_randomization?: BenchmarkDrEntry[];
  where_it_breaks?: Record<string, unknown>[];
  episodes?: BenchmarkEpisode[];
}

export interface BenchmarkReportResponse {
  available: boolean;
  report: BenchmarkReport | null;
  has_v1_archive: boolean;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { detail?: unknown };
    const d = body.detail;
    throw new Error(typeof d === 'string' ? d : Array.isArray(d) ? d.map((x) => x.msg ?? x).join(' · ') : `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

const post = (url: string, body: unknown) =>
  fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });

export const api = {
  health: () => fetch('/api/health').then((r) => json<Health>(r)),
  catalog: () => fetch('/api/catalog').then((r) => json<Catalog>(r)),
  scene: (
    robot_id: string,
    object_id: string,
    task_id: string,
    task_params: Record<string, number>,
    clutter: string[] = [],
    layout_seed: number | null = null,
  ) => post('/api/scene', { robot_id, object_id, task_id, task_params, clutter, layout_seed }).then((r) => json<SceneResponse>(r)),
  sendToPlanner: (req: PlannerRequest) => post('/api/planner/jobs', req).then((r) => json<PlannerBrief>(r)),
  stopJob: (jobId: string) => post(`/api/planner/jobs/${jobId}/stop`, {}).then((r) => json<{ stopped: boolean }>(r)),
  assetsStatus: () => fetch('/api/assets/status').then((r) => json<AssetsStatus>(r)),
  generateAsset: (prompt: string) => post('/api/assets/generate', { prompt }).then((r) => json<AssetDraft>(r)),
  saveAsset: (draftId: string) => post(`/api/assets/drafts/${draftId}/save`, {}).then((r) => json<{ object_id: string }>(r)),
  benchmark: () => fetch('/api/benchmark').then((r) => json<BenchmarkReportResponse>(r)),
  datasets: () => fetch('/api/datasets').then((r) => json<DatasetExplorerResponse>(r)),
  /** Subscribe to a planner job's events. Returns an unsubscribe function. */
  streamJob(jobId: string, onEvent: (e: JobEvent) => void, onClose: () => void): () => void {
    const es = new EventSource(`/api/planner/jobs/${jobId}/events`);
    es.addEventListener('job', (msg) => {
      const ev = JSON.parse((msg as MessageEvent).data) as JobEvent;
      onEvent(ev);
      if (ev.kind === 'done' || ev.kind === 'error') {
        es.close();
        onClose();
      }
    });
    es.onerror = () => {
      es.close();
      onClose();
    };
    return () => es.close();
  },
};
