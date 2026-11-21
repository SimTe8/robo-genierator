import { ArrowRight, CircleCheck, Hand, LoaderCircle, RotateCcw } from 'lucide-react';
import type { ObjectInfo, TaskPresetInfo } from '../types';
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
    physics.can_contain_liquid && 'can hold liquid',
    physics.sealed && 'sealed',
  ].filter(Boolean) as string[];
  return (
    <div className="mt-2 flex flex-wrap gap-1" aria-label="Physical properties">
      {chips.map((c) => (
        <span key={c} className="rounded-md border border-line bg-ink/40 px-1.5 py-0.5 text-[10px] text-muted">
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

  return (
    <Panel title={`Task for the ${object.name.toLowerCase()}`} step="2">
      <div className="mb-3 rounded-xl border border-line bg-panel-2/50 p-3 text-xs leading-relaxed">
        <p className="text-fg/90">{object.description}</p>
        <p className="mt-1 flex gap-1.5 text-muted">
          <Hand className="mt-0.5 size-3.5 shrink-0 text-accent" />
          {object.grasp_note}
        </p>
        <p className="mt-1.5 font-mono text-[10px] text-faint">
          {object.dims_cm.join(' × ')} cm · {Math.round(object.mass_kg * 1000)} g
          {object.parts.length > 0 && ` · parts: ${object.parts.join(', ')}`}
        </p>
        <PhysicsChips physics={object.physics} />
      </div>

      <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
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
              subtitle={allowed ? info?.summary : `Needs a multi-finger hand — not the ${props.robotName}`}
              note={info ? skillLabel[info.skill] ?? info.skill : undefined}
            />
          );
        })}
      </div>

      {task && (
        <div className="mt-4 grid gap-4 rounded-xl border border-line bg-panel-2/50 p-4 lg:grid-cols-2">
          <div className="flex flex-col gap-3">
            {task.params
              .filter((p) => p.key !== 'axis')
              .map((p) => {
                const v = props.params[p.key] ?? objectTask?.params[p.key] ?? p.default;
                return (
                  <label key={p.key} className="block">
                    <span className="flex items-baseline justify-between text-xs text-muted">
                      {p.label}
                      <span className="font-mono text-sm text-fg">
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
                      className="mt-1 w-full"
                    />
                  </label>
                );
              })}
            <div>
              <h3 className="mb-1.5 text-xs text-muted">Simulator success check</h3>
              <ul className="space-y-1">
                {task.criteria.map((c) => (
                  <li key={c} className="flex gap-2 text-xs leading-snug text-fg/90">
                    <CircleCheck className="mt-px size-3.5 shrink-0 text-accent" />
                    {c}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex items-baseline justify-between">
              <label htmlFor="prompt" className="text-xs text-muted">
                Prompt for the Gemini 4 planner
              </label>
              {props.promptEdited && (
                <button type="button" onClick={props.onResetPrompt} className="inline-flex items-center gap-1 text-[11px] text-accent hover:underline">
                  <RotateCcw className="size-3" /> Reset
                </button>
              )}
            </div>
            <textarea
              id="prompt"
              rows={5}
              value={props.prompt}
              onChange={(e) => props.onPrompt(e.target.value)}
              className="w-full flex-1 resize-none rounded-lg border border-line bg-ink/60 p-2.5 text-sm leading-relaxed text-fg outline-none focus:border-accent"
            />
            <p className="text-[11px] leading-snug text-faint">
              The planner chooses conditions and domain randomisation, generates trajectories, and checks each one in simulation.
            </p>
          </div>
        </div>
      )}

      <div className="mt-4 flex justify-end">
        <button
          type="button"
          onClick={props.onSend}
          disabled={!props.canSend || props.sending}
          className="inline-flex items-center gap-2 rounded-xl bg-accent px-4 py-2.5 text-sm font-semibold text-accent-ink transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {props.sending ? <LoaderCircle className="size-4 animate-spin" /> : <ArrowRight className="size-4" />}
          Send to Gemini 4 planner
        </button>
      </div>
    </Panel>
  );
}
