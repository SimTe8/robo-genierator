import { ArrowUp, CircleCheck, Hand, LoaderCircle, RotateCcw, SlidersHorizontal } from 'lucide-react';
import type { ObjectInfo, TaskPresetInfo } from '../types';
import { GeminiSparkle } from './GeminiBrand';
import { OptionCard, Panel } from './ui';

const skillLabel: Record<string, string> = {
  grasp: 'Grasp',
  'in-hand': 'In-hand',
  place: 'Place',
  'non-prehensile': 'Push / tip',
  articulated: 'Articulated',
  tool: 'Tool use',
  insertion: 'Insertion',
  reasoning: 'Physical reasoning',
};

function PhysicsChips({ physics }: { physics: ObjectInfo['physics'] }) {
  const chips = [
    physics.material,
    physics.fragility !== 'low' && `${physics.fragility} fragility`,
    physics.deformability !== 'rigid' && `${physics.deformability} deformable`,
    physics.transparency !== 'opaque' && physics.transparency,
    physics.contents !== 'none' && `contains ${physics.contents}`,
    physics.can_contain_liquid && 'holds liquid',
    physics.sealed && 'sealed',
  ].filter(Boolean) as string[];
  return (
    <div className="mt-2.5 flex flex-wrap gap-1.5" aria-label="Physical properties">
      {chips.map((c) => (
        <span
          key={c}
          className="rounded-full border border-line/90 bg-ink/55 px-2.5 py-0.5 text-[10px] font-medium text-muted"
        >
          {c}
        </span>
      ))}
    </div>
  );
}

export function TaskPicker(props: {
  robotId: string;
  robotName: string;
  object: ObjectInfo;
  tasks: Record<string, TaskPresetInfo>;
  taskId: string;
  params: Record<string, number>;
  prompt: string;
  promptEdited: boolean;
  canSend: boolean;
  sending: boolean;
  onPickTask: (taskId: string) => void;
  onParam: (key: string, value: number) => void;
  onPrompt: (text: string) => void;
  onResetPrompt: () => void;
  onSend: () => void;
}) {
  const { object } = props;
  const task = props.tasks[props.taskId];
  const objectTask = object.tasks.find((t) => t.task_id === props.taskId);
  const sliders = task ? task.params.filter((p) => p.key !== 'axis') : [];

  return (
    <Panel
      title={`Task & Gemini 4 prompt · ${object.name}`}
      step="3"
      right={
        <span className="font-mono text-[11px] text-faint">
          {object.dims_cm.join('×')} cm · {Math.round(object.mass_kg * 1000)} g
        </span>
      }
    >
      {/* Object physical profile */}
      <div className="mb-4 rounded-2xl border border-line/80 bg-panel-2/55 p-3.5 text-xs leading-relaxed">
        <p className="text-fg/95">{object.description}</p>
        <p className="mt-1.5 flex items-start gap-2 text-muted">
          <Hand className="mt-0.5 size-3.5 shrink-0 text-gemini-sky" />
          <span>{object.grasp_note}</span>
        </p>
        {object.parts.length > 0 && (
          <p className="mt-1.5 font-mono text-[10px] text-faint">Parts: {object.parts.join(' · ')}</p>
        )}
        <PhysicsChips physics={object.physics} />
      </div>

      {/* Task presets */}
      <div className="grid gap-2.5 sm:grid-cols-2 xl:grid-cols-3">
        {object.tasks.map((t) => {
          const info = props.tasks[t.task_id];
          const allowed = !info || info.robots.includes(props.robotId);
          return (
            <OptionCard
              key={t.task_id}
              selected={props.taskId === t.task_id}
              disabled={!allowed}
              onClick={() => props.onPickTask(t.task_id)}
              title={t.name}
              subtitle={allowed ? info?.summary : `Requires a multi-finger hand (not ${props.robotName})`}
              note={info ? skillLabel[info.skill] ?? info.skill : undefined}
            />
          );
        })}
      </div>

      {/* Parameters + Simulator criteria */}
      {task && (
        <div className="mt-4 space-y-4">
          <div className="grid gap-4 rounded-2xl border border-line/80 bg-panel-2/45 p-4 lg:grid-cols-2">
            {/* Left: sliders (or task goal if no numeric sliders) */}
            <div className="flex flex-col gap-3">
              {sliders.length > 0 ? (
                <>
                  <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-faint">
                    <SlidersHorizontal className="size-3 text-gemini-sky" />
                    Task parameters
                  </div>
                  {sliders.map((p) => {
                    const v = props.params[p.key] ?? objectTask?.params[p.key] ?? p.default;
                    return (
                      <label key={p.key} className="block">
                        <span className="flex items-baseline justify-between text-xs text-muted">
                          <span>{p.label}</span>
                          <span className="rounded-md bg-panel-3/90 px-2 py-0.5 font-mono text-xs font-medium text-gemini-sky">
                            {v}
                            {p.unit === '°' ? '°' : ` ${p.unit}`}
                          </span>
                        </span>
                        <input
                          type="range"
                          min={p.min}
                          max={p.max}
                          step={p.step}
                          value={v}
                          onChange={(e) => props.onParam(p.key, Number(e.target.value))}
                          className="mt-2 w-full"
                        />
                      </label>
                    );
                  })}
                </>
              ) : (
                <div>
                  <h3 className="text-[11px] font-medium uppercase tracking-wider text-faint">Task objective</h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-muted">{task.goal}</p>
                </div>
              )}
            </div>

            {/* Right: simulator success checks */}
            <div>
              <h3 className="mb-2 text-[11px] font-medium uppercase tracking-wider text-faint">
                Simulator verification checks
              </h3>
              <ul className="space-y-1.5">
                {task.criteria.map((c) => (
                  <li key={c} className="flex items-start gap-2 text-xs leading-snug text-fg/90">
                    <CircleCheck className="mt-0.5 size-3.5 shrink-0 text-gemini-sky" />
                    <span>{c}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Gemini-style prompt composer */}
          <div className="gemini-composer rounded-3xl border border-line/90 p-3.5 sm:p-4">
            <div className="mb-2 flex items-center justify-between gap-2">
              <label htmlFor="prompt" className="flex items-center gap-2 text-xs font-medium text-gemini-sky">
                <GeminiSparkle className="size-3.5" spinning={props.sending} />
                Prompt for Gemini 4 Planner
              </label>
              {props.promptEdited && (
                <button
                  type="button"
                  onClick={props.onResetPrompt}
                  className="inline-flex items-center gap-1 rounded-full border border-line px-2.5 py-0.5 text-[11px] text-muted transition hover:border-gemini-sky/40 hover:text-gemini-sky"
                >
                  <RotateCcw className="size-3" /> Reset template
                </button>
              )}
            </div>

            <textarea
              id="prompt"
              rows={4}
              value={props.prompt}
              onChange={(e) => props.onPrompt(e.target.value)}
              onKeyDown={(e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && props.canSend && !props.sending) {
                  e.preventDefault();
                  props.onSend();
                }
              }}
              placeholder="Describe the manipulation goal for the Gemini 4 planner…"
              className="w-full resize-none bg-transparent text-sm leading-relaxed text-fg placeholder:text-faint outline-none"
            />

            <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-line/70 pt-3">
              <p className="text-[11px] leading-snug text-faint">
                Gemini 4 plans trajectories, executes in MuJoCo, and self-corrects via Video Critic + Decisions API.
              </p>
              <div className="flex items-center gap-2.5">
                <span className="hidden font-mono text-[10px] text-faint sm:inline">⌘ / Ctrl + ↵</span>
                <button
                  type="button"
                  onClick={props.onSend}
                  disabled={!props.canSend || props.sending}
                  className="gemini-btn inline-flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-semibold text-white transition disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {props.sending ? (
                    <>
                      <LoaderCircle className="size-4 animate-spin" />
                      Planning…
                    </>
                  ) : (
                    <>
                      <span>Send to Gemini 4</span>
                      <span className="grid size-5 place-items-center rounded-full bg-white/20">
                        <ArrowUp className="size-3.5 stroke-[2.5]" />
                      </span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </Panel>
  );
}
