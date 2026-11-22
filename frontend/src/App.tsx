import { Bot, Hand } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from './api';
import { AssetForge } from './components/AssetForge';
import { GeminiSparkle, HackathonBadge, HackathonFooter, HackathonModal } from './components/GeminiBrand';
import { ObjectLibrary } from './components/ObjectLibrary';
import { PlannerControls } from './components/PlannerControls';
import { PlannerPanel } from './components/PlannerPanel';
import { ResearchDashboard } from './components/ResearchDashboard';
import { ScenePreview } from './components/ScenePreview';
import { SceneObjects } from './components/SceneObjects';
import { TaskPicker } from './components/TaskPicker';
import { OptionCard, Panel, Pill } from './components/ui';
import { DEFAULT_PLANNER_SETTINGS } from './types';
import type {
  Catalog,
  Health,
  JobEvent,
  ObjectInfo,
  PlannerBrief,
  PlannerSettings,
  SceneResponse,
  TaskPresetInfo,
} from './types';

export default function App() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);
  const [hackathonOpen, setHackathonOpen] = useState(false);

  const [robotId, setRobotId] = useState('franka_emika_panda');
  const [objectId, setObjectId] = useState('mug');
  const [taskId, setTaskId] = useState('place_on_pad');
  const [params, setParams] = useState<Record<string, number>>({});
  const [clutter, setClutter] = useState<string[]>([]); // extra library objects on the table
  const [layoutSeed, setLayoutSeed] = useState<number | null>(null); // null = default layout
  const [promptDraft, setPromptDraft] = useState<string | null>(null); // null = use the template prompt
  const [settings, setSettings] = useState<PlannerSettings>(DEFAULT_PLANNER_SETTINGS);

  const [scene, setScene] = useState<SceneResponse | null>(null);
  const [sceneLoading, setSceneLoading] = useState(false);
  const [sceneError, setSceneError] = useState<string | null>(null);

  const [brief, setBrief] = useState<PlannerBrief | null>(null);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [sending, setSending] = useState(false);
  const [live, setLive] = useState(false);
  const [terminalOpen, setTerminalOpen] = useState(false);
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
      .catch(() =>
        setBootError('Backend not reachable. Start it with: uvicorn app:app --app-dir backend --port 8000'),
      );
  }, []);

  const tasksById = useMemo(
    () => Object.fromEntries((catalog?.tasks ?? []).map((t) => [t.id, t])) as Record<string, TaskPresetInfo>,
    [catalog],
  );
  const object = catalog?.objects.find((o) => o.id === objectId) ?? null;
  const robot = catalog?.robots.find((r) => r.id === robotId) ?? null;
  const activeTask = tasksById[taskId] ?? null;
  const allowed = (id: string, rid = robotId) => tasksById[id]?.robots.includes(rid) ?? true;

  // Re-render the default scene + template prompt whenever object / task / params change.
  const sceneKey = JSON.stringify([robotId, objectId, taskId, params, clutter, layoutSeed]);
  useEffect(() => {
    if (!catalog) return;
    setSceneLoading(true);
    const t = setTimeout(() => {
      api
        .scene(robotId, objectId, taskId, params, clutter, layoutSeed)
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
    setSettings((s) => ({ ...s, disabled_joints: [] }));
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

  const applyPreset = (preset: 'overhang' | 'allegro_spinner' | 'mug_pad' | 'stack_block') => {
    setClutter([]);
    setLayoutSeed(null);
    if (preset === 'overhang') {
      setRobotId('franka_emika_panda');
      setObjectId('cube');
      setTaskId('pick_lift');
      setParams({ lift_cm: 12 });
      setPromptDraft('Pick up the object and lift it at least 12 cm above the table while avoiding the red overhang obstacle.');
      setSettings({
        ...DEFAULT_PLANNER_SETTINGS,
        task_type_override: 'obstacle_grasp',
        scene_seed: 42,
        max_attempts: 4,
        dr_episodes: 20,
        planner_model: 'models/gemini-4-neon',
        critic_model: 'models/gemini-4-hydrogen',
      });
    } else if (preset === 'allegro_spinner') {
      setRobotId('wonik_allegro');
      setObjectId('cube');
      setTaskId('rotate_in_hand');
      setParams({ angle_deg: 90 });
      setPromptDraft('Rotate the multi-colored cube on the spindle in-hand by at least 90 degrees using coordinated finger gaits.');
      setSettings({
        ...DEFAULT_PLANNER_SETTINGS,
        task_type_override: 'in_hand_spindle',
        scene_seed: 302,
        max_attempts: 3,
        dr_episodes: 20,
        planner_model: 'models/gemini-4-neon',
        critic_model: 'models/gemini-4-hydrogen',
      });
    } else if (preset === 'stack_block') {
      setRobotId('franka_emika_panda');
      setObjectId('cube');
      setTaskId('stack_on_block');
      setParams({});
      setPromptDraft(null);
      setSettings({
        ...DEFAULT_PLANNER_SETTINGS,
        task_type_override: 'auto',
        scene_seed: 42,
        max_attempts: 4,
        dr_episodes: 20,
        planner_model: 'models/gemini-4-neon',
        critic_model: 'models/gemini-4-hydrogen',
      });
    } else {
      setRobotId('franka_emika_panda');
      setObjectId('mug');
      setTaskId('place_on_pad');
      setParams({});
      setPromptDraft(null);
      setSettings({
        ...DEFAULT_PLANNER_SETTINGS,
        task_type_override: 'auto',
        scene_seed: 42,
        max_attempts: 4,
        dr_episodes: 20,
        planner_model: 'models/gemini-4-neon',
        critic_model: 'models/gemini-4-hydrogen',
      });
    }
  };

  const prompt = promptDraft ?? scene?.prompt ?? '';

  const send = async () => {
    unsubscribe.current?.();
    setSending(true);
    setSendError(null);
    setEvents([]);
    setBrief(null);
    setTerminalOpen(true); // Automatically open the live terminal when generation begins
    try {
      const b = await api.sendToPlanner({
        robot_id: robotId,
        object_id: objectId,
        task_id: taskId,
        task_params: params,
        prompt,
        settings,
        clutter,
        layout_seed: layoutSeed,
      });
      setBrief(b);
      setLive(true);
      unsubscribe.current = api.streamJob(
        b.job_id,
        (ev) => setEvents((prev) => [...prev, ev]),
        () => setLive(false),
      );
      setTimeout(() => {
        document.getElementById('planner')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 60);
    } catch (e) {
      setSendError((e as Error).message);
    } finally {
      setSending(false);
    }
  };

  const stopRun = async () => {
    if (!brief) return;
    try {
      await api.stopJob(brief.job_id);
    } catch {
      // ignore if already finished
    }
  };

  return (
    <div className="min-h-screen">
      {/* Sticky Gemini Studio top bar */}
      <header className="sticky top-0 z-30 border-b border-line/70 bg-ink/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1480px] flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="grid size-9 place-items-center rounded-2xl border border-gemini-blue/35 bg-gradient-to-br from-gemini-blue/20 via-panel to-gemini-violet/20 shadow-[0_0_24px_-6px_rgba(66,133,244,0.75)]">
              <GeminiSparkle className="size-5" spinning={live || sending} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-display text-lg font-bold tracking-tight">
                  <span className="text-fg">Robo </span>
                  <span className="gemini-text">Genierator</span>
                </h1>
                <span className="hidden rounded-full border border-gemini-blue/30 bg-gemini-blue/10 px-2 py-0.5 text-[10px] font-medium text-gemini-sky sm:inline">
                  Gemini 4 Studio
                </span>
              </div>
              <p className="text-xs text-muted">
                Zero-shot cross-embodiment trajectory synthesis &amp; visual self-correction
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Pill tone={health ? 'ok' : 'muted'}>MuJoCo {health?.mujoco ?? '…'}</Pill>
            <Pill tone={health?.planner ? 'ok' : 'warn'} pulse={live}>
              Planner: {health?.planner ?? 'standby'}
            </Pill>
            <HackathonBadge onOpen={() => setHackathonOpen(true)} />
          </div>
        </div>
        {/* Signature Gemini 4 NYC electric-blue horizon line */}
        <div className="gemini-horizon h-[1.5px] w-full" />
      </header>

      <main className="mx-auto max-w-[1480px] px-4 pb-16 pt-5 sm:px-6">
        {/* Live selection breadcrumb strip + 1-click Jury Demo Presets */}
        {catalog && object && robot && (
          <div className="mb-4 flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-line/70 bg-panel/60 px-4 py-2 text-xs backdrop-blur-sm">
            <div className="flex flex-wrap items-center gap-1.5 text-muted">
              <span className="text-[11px] font-semibold text-gemini-sky">Benchmark Presets:</span>
              <button
                type="button"
                onClick={() => applyPreset('overhang')}
                className="rounded-full border border-gemini-blue/35 bg-gemini-blue/10 px-2.5 py-0.5 text-[11px] font-medium text-fg transition hover:border-gemini-sky/60 hover:bg-gemini-blue/20"
              >
                1. Overhang Grasp + 20× DR
              </button>
              <button
                type="button"
                onClick={() => applyPreset('allegro_spinner')}
                className="rounded-full border border-gemini-violet/35 bg-gemini-violet/10 px-2.5 py-0.5 text-[11px] font-medium text-fg transition hover:border-gemini-sky/60 hover:bg-gemini-violet/20"
              >
                2. Allegro 16-DOF Spinner + 20× DR
              </button>
              <button
                type="button"
                onClick={() => applyPreset('mug_pad')}
                className="rounded-full border border-ok/35 bg-ok/10 px-2.5 py-0.5 text-[11px] font-medium text-fg transition hover:border-ok/60 hover:bg-ok/20"
              >
                3. Mug Place on Pad + 20× DR
              </button>
              <button
                type="button"
                onClick={() => applyPreset('stack_block')}
                className="rounded-full border border-warn/35 bg-warn/10 px-2.5 py-0.5 text-[11px] font-medium text-fg transition hover:border-warn/60 hover:bg-warn/20"
              >
                4. Stack Block + 2-Object DR
              </button>
              <span className="ml-2 hidden text-[11px] text-faint xl:inline">
                ({robot.name} · {object.name} · {activeTask?.name ?? taskId} · DR={settings.dr_episodes}×)
              </span>
            </div>
            {(brief || live) && (
              <a
                href="#planner"
                className="inline-flex items-center gap-1.5 rounded-full border border-gemini-blue/35 bg-gemini-blue/12 px-3 py-0.5 text-[11px] font-medium text-gemini-sky transition hover:bg-gemini-blue/20"
              >
                <GeminiSparkle className="size-3" spinning={live} />
                {live ? 'Planner running · jump to terminal' : 'View latest planner run'}
              </a>
            )}
          </div>
        )}

        {bootError && (
          <div className="rounded-3xl border border-bad/30 bg-bad/10 p-5 text-sm text-bad">
            <p className="font-display font-semibold">Backend connection required</p>
            <p className="mt-1 font-mono text-xs text-bad/90">{bootError}</p>
          </div>
        )}

        {catalog && object && robot && (
          <>
            <div className="grid gap-5 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
              <div className="flex min-w-0 flex-col gap-5">
                <Panel
                  title="Embodiment"
                  step="1"
                  right={
                    <span className="rounded-full border border-line bg-panel-2/80 px-2.5 py-0.5 text-[10px] text-faint">
                      {robot.license}
                    </span>
                  }
                >
                  <div className="grid gap-2.5 sm:grid-cols-3">
                    {catalog.robots.map((r) => (
                      <OptionCard
                        key={r.id}
                        selected={r.id === robotId}
                        onClick={() => pickRobot(r.id)}
                        icon={r.id === 'franka_emika_panda' ? <Bot className="size-4" /> : <Hand className="size-4" />}
                        title={r.name}
                        subtitle={r.kind}
                      />
                    ))}
                  </div>
                  <p className="mt-3 text-xs leading-relaxed text-muted">{robot.description}</p>
                </Panel>

                <ObjectLibrary objects={catalog.objects} selectedId={objectId} onSelect={pickObject} />
                <AssetForge onSaved={(id) => reloadCatalog(id)} />
              </div>

              <div className="flex min-w-0 flex-col gap-5">
                <ScenePreview scene={scene} loading={sceneLoading} error={sceneError} robotName={robot.name} />
                <SceneObjects
                  objects={catalog.objects}
                  target={object}
                  sceneObjects={scene?.scene.other_objects ?? []}
                  clutter={clutter}
                  onChange={(ids) => {
                    setClutter(ids);
                    setPromptDraft(null);
                  }}
                  layoutSeed={layoutSeed}
                  layoutNote={scene?.scene.layout_note}
                  busy={sceneLoading}
                  onShuffle={() => setLayoutSeed(1 + Math.floor(Math.random() * 99_999))}
                  onResetLayout={() => setLayoutSeed(null)}
                />
                <PlannerControls robotId={robotId} settings={settings} onChange={setSettings} />
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

            <div id="planner" className="mt-5 scroll-mt-20">
              <PlannerPanel
                brief={brief}
                events={events}
                live={live}
                error={sendError}
                terminalOpen={terminalOpen}
                onToggleTerminal={() => setTerminalOpen((v) => !v)}
                onStop={stopRun}
              />
            </div>

            <div className="mt-5">
              <ResearchDashboard refreshTrigger={events.length} />
            </div>
          </>
        )}

        <HackathonFooter onOpenModal={() => setHackathonOpen(true)} />
      </main>

      <HackathonModal open={hackathonOpen} onClose={() => setHackathonOpen(false)} />
    </div>
  );
}
