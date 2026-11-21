import type { AssetDraft, AssetsStatus, Catalog, Health, JobEvent, PlannerBrief, PlannerRequest, SceneResponse } from './types';

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
  scene: (robot_id: string, object_id: string, task_id: string, task_params: Record<string, number>) =>
    post('/api/scene', { robot_id, object_id, task_id, task_params }).then((r) => json<SceneResponse>(r)),
  sendToPlanner: (req: PlannerRequest) => post('/api/planner/jobs', req).then((r) => json<PlannerBrief>(r)),
  assetsStatus: () => fetch('/api/assets/status').then((r) => json<AssetsStatus>(r)),
  generateAsset: (prompt: string) => post('/api/assets/generate', { prompt }).then((r) => json<AssetDraft>(r)),
  saveAsset: (draftId: string) => post(`/api/assets/drafts/${draftId}/save`, {}).then((r) => json<{ object_id: string }>(r)),
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
