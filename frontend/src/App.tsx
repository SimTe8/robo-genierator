import { Bot, Hand } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from './api';
import { AssetForge } from './components/AssetForge';
import { ObjectLibrary } from './components/ObjectLibrary';
import { PlannerPanel } from './components/PlannerPanel';
import { ScenePreview } from './components/ScenePreview';
import { TaskPicker } from './components/TaskPicker';
import { OptionCard, Panel, Pill } from './components/ui';
import type { Catalog, Health, JobEvent, ObjectInfo, PlannerBrief, SceneResponse, TaskPresetInfo } from './types';

export default function App() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);

  const [robotId, setRobotId] = useState('wonik_allegro');
  const [objectId, setObjectId] = useState('key');
  const [taskId, setTaskId] = useState('unlock_with_key');
  const [params, setParams] = useState<Record<string, number>>({});
  const [promptDraft, setPromptDraft] = useState<string | null>(null); // null = use the template prompt

  const [scene, setScene] = useState<SceneResponse | null>(null);
  const [sceneLoading, setSceneLoading] = useState(false);
  const [sceneError, setSceneError] = useState<string | null>(null);

  const [brief, setBrief] = useState<PlannerBrief | null>(null);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [sending, setSending] = useState(false);
  const [live, setLive] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);
  const unsubscribe = useRef<(() => void) | null>(null);

  const reloadCatalog = (selectId?: string) =>
    api.catalog().then((c) => {
      setCatalog(c);
      const o = selectId ? c.objects.find((x) => x.id === selectId) : undefined;
      if (o) {
        setObjectId(o.id);
        setTaskId(o.tasks[0].task_id);
        setParams({});
        setPromptDraft(null);
      }
    });

  useEffect(() => {
    Promise.all([api.catalog(), api.health()])
      .then(([c, h]) => {
        setCatalog(c);
        setHealth(h);
      })
      .catch(() => setBootError('Backend not reachable. Start it with: uvicorn app:app --app-dir backend --port 8000'));
  }, []);

  const tasksById = useMemo(
    () => Object.fromEntries((catalog?.tasks ?? []).map((t) => [t.id, t])) as Record<string, TaskPresetInfo>,
    [catalog],
  );
  const object = catalog?.objects.find((o) => o.id === objectId) ?? null;
  const robot = catalog?.robots.find((r) => r.id === robotId) ?? null;
  const allowed = (id: string, rid = robotId) => tasksById[id]?.robots.includes(rid) ?? true;

  // Re-render the default scene + template prompt whenever object / task / params change.
  const sceneKey = JSON.stringify([robotId, objectId, taskId, params]);
  useEffect(() => {
    if (!catalog) return;
    setSceneLoading(true);
    const t = setTimeout(() => {
      api
        .scene(robotId, objectId, taskId, params)
        .then((s) => {
          setScene(s);
          setSceneError(null);
        })
        .catch((e: Error) => setSceneError(e.message))
        .finally(() => setSceneLoading(false));
    }, 200);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [catalog, sceneKey]);

  const pickObject = (o: ObjectInfo) => {
    setObjectId(o.id);
    const keep = o.tasks.some((t) => t.task_id === taskId) && allowed(taskId);
    if (!keep) setTaskId((o.tasks.find((t) => allowed(t.task_id)) ?? o.tasks[0]).task_id);
    setParams({});
    setPromptDraft(null);
  };

  const pickRobot = (rid: string) => {
    setRobotId(rid);
    if (object && !allowed(taskId, rid)) {
      setTaskId((object.tasks.find((t) => allowed(t.task_id, rid)) ?? object.tasks[0]).task_id);
      setParams({});
      setPromptDraft(null);
    }
  };

  const pickTask = (id: string) => {
    setTaskId(id);
    setParams({});
    setPromptDraft(null);
  };

  const prompt = promptDraft ?? scene?.prompt ?? '';

  const send = async () => {
    unsubscribe.current?.();
    setSending(true);
    setSendError(null);
    setEvents([]);
    setBrief(null);
    try {
      const b = await api.sendToPlanner({ robot_id: robotId, object_id: objectId, task_id: taskId, task_params: params, prompt });
      setBrief(b);
      setLive(true);
      unsubscribe.current = api.streamJob(b.job_id, (ev) => setEvents((prev) => [...prev, ev]), () => setLive(false));
      document.getElementById('planner')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
      setSendError((e as Error).message);
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="mx-auto max-w-[1440px] px-4 pb-16 pt-5 sm:px-6">
      <header className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">
            Robo Genierator
            <span className="ml-2 align-middle text-xs font-normal text-faint">dexterous trajectory generator</span>
          </h1>
          <p className="mt-1 text-sm text-muted">Choose a robot, an object and a task. The Gemini 4 planner generates and verifies the trajectories.</p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <Pill tone={health ? 'ok' : 'muted'}>MuJoCo {health?.mujoco ?? '…'}</Pill>
          <Pill tone={health?.planner ? 'ok' : 'warn'}>Planner: {health?.planner ?? 'not connected'}</Pill>
        </div>
      </header>

      {bootError && <p className="rounded-xl border border-bad/30 bg-bad/10 p-4 text-sm text-bad">{bootError}</p>}

      {catalog && object && robot && (
        <>
          <div className="grid gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
            <div className="flex min-w-0 flex-col gap-4">
              <Panel title="Robot" right={<span className="text-[11px] text-faint">{robot.license}</span>}>
                <div className="grid gap-2 sm:grid-cols-3">
                  {catalog.robots.map((r) => (
                    <OptionCard
                      key={r.id}
                      selected={r.id === robotId}
                      onClick={() => pickRobot(r.id)}
                      icon={r.id === 'franka_emika_panda' ? <Bot className="size-5" /> : <Hand className="size-5" />}
                      title={r.name}
                      subtitle={r.kind}
                    />
                  ))}
                </div>
                <p className="mt-2 text-xs leading-snug text-muted">{robot.description}</p>
              </Panel>
              <ObjectLibrary objects={catalog.objects} selectedId={objectId} onSelect={pickObject} />
              <AssetForge onSaved={(id) => reloadCatalog(id)} />
            </div>
            <div className="flex min-w-0 flex-col gap-4">
              <ScenePreview scene={scene} loading={sceneLoading} error={sceneError} />
              <TaskPicker
                robotId={robotId}
                robotName={robot.name}
                object={object}
                tasks={tasksById}
                taskId={taskId}
                params={params}
                prompt={prompt}
                promptEdited={promptDraft !== null}
                canSend={!!scene && !sceneLoading && prompt.trim().length > 0}
                sending={sending || live}
                onPickTask={pickTask}
                onParam={(k, v) => {
                  setParams((p) => ({ ...p, [k]: v }));
                  setPromptDraft(null);
                }}
                onPrompt={setPromptDraft}
                onResetPrompt={() => setPromptDraft(null)}
                onSend={send}
              />
            </div>
          </div>
          <div id="planner" className="mt-4 scroll-mt-4">
            <PlannerPanel brief={brief} events={events} live={live} error={sendError} />
          </div>
        </>
      )}
    </div>
  );
}
