import { Brain, ChevronDown, Dices, RotateCcw, ShieldAlert, Sliders } from 'lucide-react';
import { useState } from 'react';
import { cn } from '../lib/cn';
import { DEFAULT_PLANNER_SETTINGS } from '../types';
import type { FeedbackMode, ObjectShapeOverride, PlannerSettings, TaskTypeOverride, ThinkingLevel } from '../types';
import { Panel, Segmented } from './ui';

const GEMINI_MODELS = [
  { value: 'models/gemini-4-neon', label: 'Gemini 4 Neon' },
  { value: 'models/gemini-4-carbon', label: 'Gemini 4 Carbon' },
  { value: 'models/gemini-4-hydrogen', label: 'Gemini 4 Hydrogen' },
  { value: 'models/gemini-4-argon', label: 'Gemini 4 Argon' },
];

const FEEDBACK_OPTIONS: { value: FeedbackMode; label: string }[] = [
  { value: 'video', label: 'Video Critic' },
  { value: 'text_only', label: 'Text Only' },
  { value: 'one_shot', label: 'One-Shot' },
];

const THINKING_OPTIONS: { value: ThinkingLevel; label: string }[] = [
  { value: 'minimal', label: 'Minimal' },
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
];

const TASK_TYPE_OPTIONS: { value: TaskTypeOverride; label: string }[] = [
  { value: 'auto', label: 'Auto (from task)' },
  { value: 'pick_and_lift', label: 'Pick & Lift' },
  { value: 'obstacle_grasp', label: 'Obstacle Grasp' },
  { value: 'in_hand_rotation', label: 'In-Hand Rotation' },
  { value: 'in_hand_spindle', label: 'In-Hand Spindle' },
];

const SHAPE_OPTIONS: { value: ObjectShapeOverride; label: string }[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'box', label: 'Box' },
  { value: 'cylinder', label: 'Cylinder' },
  { value: 'sphere', label: 'Sphere' },
];

const LOOP_PRESETS = [1, 2, 4, 6, 8];
const CHUNK_PRESETS = [45, 60, 90, 120, 180];

const FAULT_PRESETS: Record<string, { label: string; joints: string[] }[]> = {
  franka_emika_panda: [
    { label: 'Lock Elbow (joint4)', joints: ['joint4'] },
    { label: 'Lock Wrist (joint6)', joints: ['joint6'] },
    { label: 'Lock Shoulder (joint2)', joints: ['joint2'] },
  ],
  wonik_allegro: [
    { label: 'Disable Middle Finger', joints: ['mfj1', 'mfj2'] },
    { label: 'Disable Ring Finger', joints: ['rfj1', 'rfj2'] },
    { label: 'Disable Index Base', joints: ['ffj1'] },
  ],
  wuji_hand: [
    { label: 'Disable Middle Finger', joints: ['right_finger3_joint1', 'right_finger3_joint3'] },
    { label: 'Disable Ring & Little', joints: ['right_finger4_joint1', 'right_finger5_joint1'] },
    { label: 'Lock Thumb Opposition', joints: ['right_finger1_joint2'] },
  ],
};

const ROBOT_JOINTS: Record<string, string[]> = {
  franka_emika_panda: ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'],
  wonik_allegro: ['ffj0', 'ffj1', 'ffj2', 'mfj0', 'mfj1', 'mfj2', 'rfj0', 'rfj1', 'rfj2', 'thj0', 'thj1', 'thj2'],
  wuji_hand: [
    'right_finger1_joint1',
    'right_finger1_joint2',
    'right_finger1_joint3',
    'right_finger1_joint4',
    'right_finger2_joint1',
    'right_finger2_joint3',
    'right_finger3_joint1',
    'right_finger3_joint3',
    'right_finger4_joint1',
    'right_finger4_joint3',
    'right_finger5_joint1',
    'right_finger5_joint3',
  ],
};

export function PlannerControls({
  robotId,
  settings,
  onChange,
}: {
  robotId: string;
  settings: PlannerSettings;
  onChange: (next: PlannerSettings) => void;
}) {
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const patch = (partial: Partial<PlannerSettings>) => onChange({ ...settings, ...partial });

  const toggleJoint = (j: string) => {
    const has = settings.disabled_joints.includes(j);
    patch({
      disabled_joints: has ? settings.disabled_joints.filter((x) => x !== j) : [...settings.disabled_joints, j],
    });
  };

  const advancedCount =
    (settings.disabled_joints.length > 0 ? 1 : 0) +
    (settings.chunk_step_deg !== null ? 1 : 0) +
    (settings.dr_episodes > 0 ? 1 : 0) +
    (settings.task_type_override !== 'auto' ? 1 : 0) +
    (settings.object_shape_override !== 'auto' ? 1 : 0) +
    (settings.scene_seed !== 42 ? 1 : 0);

  const isModified =
    JSON.stringify(settings) !== JSON.stringify(DEFAULT_PLANNER_SETTINGS);

  const presets = FAULT_PRESETS[robotId] ?? FAULT_PRESETS.wonik_allegro;
  const joints = ROBOT_JOINTS[robotId] ?? ROBOT_JOINTS.wonik_allegro;

  return (
    <Panel
      title="Planner & critic controls"
      icon={<Sliders className="size-4" />}
      right={
        <div className="flex items-center gap-2">
          {isModified && (
            <button
              type="button"
              onClick={() => onChange(DEFAULT_PLANNER_SETTINGS)}
              className="inline-flex items-center gap-1 rounded-full border border-line px-2.5 py-0.5 text-[11px] text-muted transition hover:border-gemini-sky/40 hover:text-gemini-sky"
            >
              <RotateCcw className="size-3" /> Reset
            </button>
          )}
          <span className="rounded-full border border-gemini-blue/30 bg-gemini-blue/10 px-2.5 py-0.5 font-mono text-[11px] text-gemini-sky">
            {settings.feedback_mode === 'one_shot' ? '1 loop' : `≤ ${settings.max_attempts} loops`}
          </span>
        </div>
      }
    >
      {/* Core controls grid */}
      <div className="grid gap-4 sm:grid-cols-2">
        {/* Max loops */}
        <div className="rounded-2xl border border-line/80 bg-panel-2/55 p-3.5">
          <div className="flex items-center justify-between gap-2">
            <label htmlFor="max-loops" className="text-xs font-medium text-fg">
              Max refinement loops
            </label>
            <span className="rounded-md bg-panel-3 px-2 py-0.5 font-mono text-xs font-semibold text-gemini-sky">
              {settings.feedback_mode === 'one_shot' ? '1 (one-shot)' : settings.max_attempts}
            </span>
          </div>
          <input
            id="max-loops"
            type="range"
            min={1}
            max={12}
            step={1}
            disabled={settings.feedback_mode === 'one_shot'}
            value={settings.max_attempts}
            onChange={(e) => patch({ max_attempts: Number(e.target.value) })}
            className="mt-2.5 w-full disabled:opacity-40"
          />
          <div className="mt-2.5 flex flex-wrap gap-1.5">
            {LOOP_PRESETS.map((n) => (
              <button
                key={n}
                type="button"
                disabled={settings.feedback_mode === 'one_shot'}
                onClick={() => patch({ max_attempts: n })}
                className={cn(
                  'rounded-full border px-2.5 py-0.5 font-mono text-[11px] transition disabled:opacity-40',
                  settings.max_attempts === n && settings.feedback_mode !== 'one_shot'
                    ? 'border-gemini-blue/65 bg-gemini-blue/20 text-gemini-sky'
                    : 'border-line bg-panel/70 text-muted hover:border-gemini-sky/35 hover:text-fg',
                )}
              >
                {n}
              </button>
            ))}
          </div>
        </div>

        {/* Feedback mode */}
        <div className="flex flex-col justify-between rounded-2xl border border-line/80 bg-panel-2/55 p-3.5">
          <div>
            <span className="block text-xs font-medium text-fg">Feedback &amp; critic mode</span>
            <p className="mt-0.5 text-[11px] leading-snug text-muted">
              {settings.feedback_mode === 'video'
                ? 'Critic watches the MuJoCo MP4 rollout + Decisions API.'
                : settings.feedback_mode === 'text_only'
                  ? 'Telemetry-only ablation (no video frames).'
                  : 'Single-shot baseline with no critic refinement.'}
            </p>
          </div>
          <div className="mt-2.5">
            <Segmented
              value={settings.feedback_mode}
              options={FEEDBACK_OPTIONS}
              onChange={(v) => patch({ feedback_mode: v })}
            />
          </div>
        </div>
      </div>

      {/* Model selectors */}
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <label className="block rounded-2xl border border-line/80 bg-panel-2/45 px-3.5 py-2.5">
          <span className="block text-[10px] font-medium uppercase tracking-wider text-faint">Planner model</span>
          <select
            value={settings.planner_model}
            onChange={(e) => patch({ planner_model: e.target.value })}
            className="mt-1 w-full bg-transparent text-xs font-medium text-fg outline-none"
          >
            {GEMINI_MODELS.map((m) => (
              <option key={m.value} value={m.value} className="bg-panel text-fg">
                {m.label} ({m.value.replace('models/', '')})
              </option>
            ))}
          </select>
        </label>

        <label className="block rounded-2xl border border-line/80 bg-panel-2/45 px-3.5 py-2.5">
          <span className="block text-[10px] font-medium uppercase tracking-wider text-faint">
            Video critic + Decisions API model
          </span>
          <select
            value={settings.critic_model}
            disabled={settings.feedback_mode === 'one_shot'}
            onChange={(e) => patch({ critic_model: e.target.value })}
            className="mt-1 w-full bg-transparent text-xs font-medium text-fg outline-none disabled:opacity-45"
          >
            {GEMINI_MODELS.map((m) => (
              <option key={m.value} value={m.value} className="bg-panel text-fg">
                {m.label} ({m.value.replace('models/', '')})
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* Thinking budget selector */}
      <div className="mt-3 flex flex-col justify-between gap-2.5 rounded-2xl border border-line/80 bg-panel-2/45 p-3.5 sm:flex-row sm:items-center">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <Brain className="size-3.5 text-gemini-sky" />
            <span className="text-xs font-medium text-fg">Planner thinking budget</span>
            <span className="rounded bg-panel-3 px-1.5 py-px font-mono text-[10px] text-gemini-sky">
              {settings.thinking_level}
            </span>
          </div>
          <p className="mt-0.5 text-[11px] leading-snug text-muted">
            Controls Gemini 4 reasoning budget to balance multi-step planning depth vs. latency
          </p>
        </div>
        <div className="shrink-0">
          <Segmented
            value={settings.thinking_level}
            options={THINKING_OPTIONS}
            onChange={(v) => patch({ thinking_level: v })}
          />
        </div>
      </div>

      {/* Advanced & Stress-Test Controls toggle */}
      <div className="mt-3">
        <button
          type="button"
          onClick={() => setAdvancedOpen((v) => !v)}
          className="flex w-full items-center justify-between rounded-2xl border border-line/80 bg-panel-2/35 px-3.5 py-2 text-xs font-medium text-muted transition hover:border-gemini-sky/35 hover:text-fg"
        >
          <span className="inline-flex items-center gap-2">
            <ShieldAlert className="size-3.5 text-gemini-sky" />
            Fault injection, seed, chunked 360° &amp; domain randomization
            {advancedCount > 0 && (
              <span className="rounded-full bg-gemini-blue/20 px-2 py-px font-mono text-[10px] text-gemini-sky">
                {advancedCount} active
              </span>
            )}
          </span>
          <ChevronDown className={cn('size-4 text-faint transition', advancedOpen && 'rotate-180')} />
        </button>

        {advancedOpen && (
          <div className="mt-2.5 space-y-3.5 rounded-2xl border border-line/80 bg-panel-2/40 p-3.5">
            {/* Pipeline override + shape + seed */}
            <div className="grid gap-3 sm:grid-cols-3">
              <label className="block">
                <span className="block text-[10px] font-medium uppercase tracking-wider text-faint">Pipeline task</span>
                <select
                  value={settings.task_type_override}
                  onChange={(e) => patch({ task_type_override: e.target.value as TaskTypeOverride })}
                  className="mt-1 w-full rounded-xl border border-line bg-ink/70 px-2.5 py-1.5 text-xs text-fg outline-none focus:border-gemini-blue"
                >
                  {TASK_TYPE_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value} className="bg-panel text-fg">
                      {o.label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="block">
                <span className="block text-[10px] font-medium uppercase tracking-wider text-faint">Object shape</span>
                <select
                  value={settings.object_shape_override}
                  onChange={(e) => patch({ object_shape_override: e.target.value as ObjectShapeOverride })}
                  className="mt-1 w-full rounded-xl border border-line bg-ink/70 px-2.5 py-1.5 text-xs text-fg outline-none focus:border-gemini-blue"
                >
                  {SHAPE_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value} className="bg-panel text-fg">
                      {o.label}
                    </option>
                  ))}
                </select>
              </label>

              <div>
                <span className="block text-[10px] font-medium uppercase tracking-wider text-faint">Scene seed</span>
                <div className="mt-1 flex gap-1.5">
                  <input
                    type="number"
                    value={settings.scene_seed}
                    onChange={(e) => patch({ scene_seed: Number(e.target.value) || 0 })}
                    className="w-full min-w-0 rounded-xl border border-line bg-ink/70 px-2.5 py-1.5 font-mono text-xs text-fg outline-none focus:border-gemini-blue"
                  />
                  <button
                    type="button"
                    title="Randomize seed"
                    onClick={() => patch({ scene_seed: Math.floor(Math.random() * 999) + 1 })}
                    className="grid size-8 shrink-0 place-items-center rounded-xl border border-line bg-panel text-muted transition hover:border-gemini-sky/40 hover:text-gemini-sky"
                  >
                    <Dices className="size-3.5" />
                  </button>
                </div>
              </div>
            </div>

            {/* Live fault injection */}
            <div>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-[10px] font-medium uppercase tracking-wider text-faint">
                  Live fault injection (lock joints)
                </span>
                {settings.disabled_joints.length > 0 && (
                  <button
                    type="button"
                    onClick={() => patch({ disabled_joints: [] })}
                    className="text-[11px] text-warn hover:underline"
                  >
                    Clear faults ({settings.disabled_joints.length})
                  </button>
                )}
              </div>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {presets.map((p) => {
                  const active =
                    p.joints.every((j) => settings.disabled_joints.includes(j)) &&
                    settings.disabled_joints.length === p.joints.length;
                  return (
                    <button
                      key={p.label}
                      type="button"
                      onClick={() => patch({ disabled_joints: active ? [] : p.joints })}
                      className={cn(
                        'rounded-full border px-2.5 py-0.5 text-[11px] transition',
                        active
                          ? 'border-warn/60 bg-warn/15 text-warn'
                          : 'border-line bg-panel/70 text-muted hover:border-warn/40 hover:text-fg',
                      )}
                    >
                      {p.label}
                    </button>
                  );
                })}
              </div>
              <div className="mt-2 flex flex-wrap gap-1">
                {joints.map((j) => {
                  const locked = settings.disabled_joints.includes(j);
                  return (
                    <button
                      key={j}
                      type="button"
                      onClick={() => toggleJoint(j)}
                      className={cn(
                        'rounded-lg border px-2 py-0.5 font-mono text-[10px] transition',
                        locked
                          ? 'border-bad/60 bg-bad/15 text-bad'
                          : 'border-line/70 bg-ink/50 text-faint hover:border-faint hover:text-muted',
                      )}
                    >
                      {j}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Checkpointed chunking + Domain Randomization multiplier */}
            <div className="grid gap-3 border-t border-line/60 pt-3 sm:grid-cols-2">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-fg">Receding-horizon chunks</span>
                  <button
                    type="button"
                    onClick={() => patch({ chunk_step_deg: settings.chunk_step_deg === null ? 90 : null })}
                    className={cn(
                      'rounded-full border px-2.5 py-0.5 text-[10px] font-medium transition',
                      settings.chunk_step_deg !== null
                        ? 'border-gemini-blue/60 bg-gemini-blue/20 text-gemini-sky'
                        : 'border-line bg-panel text-muted',
                    )}
                  >
                    {settings.chunk_step_deg !== null ? `On (${settings.chunk_step_deg}°)` : 'Off'}
                  </button>
                </div>
                <p className="mt-0.5 text-[11px] text-faint">Checkpointed milestones for long in-hand rotation (e.g. 360°).</p>
                {settings.chunk_step_deg !== null && (
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {CHUNK_PRESETS.map((deg) => (
                      <button
                        key={deg}
                        type="button"
                        onClick={() => patch({ chunk_step_deg: deg })}
                        className={cn(
                          'rounded-full border px-2.5 py-0.5 font-mono text-[10px] transition',
                          settings.chunk_step_deg === deg
                            ? 'border-gemini-blue/65 bg-gemini-blue/20 text-gemini-sky'
                            : 'border-line bg-panel text-muted hover:text-fg',
                        )}
                      >
                        {deg}°
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div>
                <div className="flex items-center justify-between">
                  <label htmlFor="dr-episodes" className="text-xs font-medium text-fg">
                    Domain randomization replay
                  </label>
                  <span className="font-mono text-xs text-gemini-sky">
                    {settings.dr_episodes === 0 ? 'Off' : `${settings.dr_episodes} eps`}
                  </span>
                </div>
                <p className="mt-0.5 text-[11px] text-faint">Zero-shot transfer test across randomized scenes after success.</p>
                <input
                  id="dr-episodes"
                  type="range"
                  min={0}
                  max={20}
                  step={1}
                  value={settings.dr_episodes}
                  onChange={(e) => patch({ dr_episodes: Number(e.target.value) })}
                  className="mt-2 w-full"
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </Panel>
  );
}
