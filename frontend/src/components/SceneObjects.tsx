import { Layers, Plus, RotateCcw, Shuffle, X } from 'lucide-react';
import { cn } from '../lib/cn';
import type { ObjectInfo, SceneObject } from '../types';
import { Panel } from './ui';

export const MAX_CLUTTER = 5;

const ROLE_LABEL: Record<SceneObject['role'], string> = {
  stack: 'to stack',
  bin_item: 'goes in the bin',
  sort: 'to sort',
  base: 'stack on this',
  distractor: 'surrounding clutter',
  clutter: 'added by you',
};

const COLOUR_DOT: Record<string, string> = { red: 'bg-[#e6332e]', blue: 'bg-[#2e6bf2]' };

/** Everything on the table besides the robot: the target, task-spawned objects, and user-added clutter. */
export function SceneObjects({
  objects,
  target,
  sceneObjects,
  clutter,
  onChange,
  layoutSeed,
  layoutNote,
  busy,
  onShuffle,
  onResetLayout,
}: {
  objects: ObjectInfo[];
  target: ObjectInfo;
  sceneObjects: SceneObject[];
  clutter: string[];
  onChange: (ids: string[]) => void;
  /** null = default layout */
  layoutSeed: number | null;
  layoutNote?: string;
  busy: boolean;
  onShuffle: () => void;
  onResetLayout: () => void;
}) {
  const byId = Object.fromEntries(objects.map((o) => [o.id, o]));
  const fromTask = sceneObjects.filter((o) => o.role !== 'clutter');
  const candidates = objects.filter((o) => o.can_be_extra);
  const full = clutter.length >= MAX_CLUTTER;
  const total = 1 + fromTask.length + clutter.length;

  return (
    <Panel
      title="Scene objects & layout"
      icon={<Layers className="size-4" />}
      right={
        <span className="rounded-full border border-line bg-panel-2/80 px-2.5 py-0.5 font-mono text-[11px] text-muted">
          {total} on table
        </span>
      }
    >
      <div className="mb-3.5 flex flex-wrap items-center gap-2 rounded-2xl border border-line bg-panel-2/60 p-2 pl-3">
        <div className="min-w-0 flex-1">
          <p className="text-xs font-medium text-fg">
            {layoutSeed === null ? 'Default layout' : 'Shuffled layout'}
            {layoutSeed !== null && (
              <span className="ml-2 rounded-full border border-gemini-blue/35 bg-gemini-blue/12 px-2 py-0.5 font-mono text-[10px] text-gemini-sky">
                seed {layoutSeed}
              </span>
            )}
          </p>
          <p className="mt-0.5 text-[11px] leading-snug text-faint">
            {layoutNote ??
              (layoutSeed === null
                ? 'Shuffle to move every object (and turn it, where that is safe) before sending.'
                : 'The planner gets this exact arrangement: same seed, same scene.')}
          </p>
        </div>
        <div className="flex shrink-0 gap-1.5">
          {layoutSeed !== null && (
            <button
              type="button"
              onClick={onResetLayout}
              disabled={busy}
              className="inline-flex items-center gap-1.5 rounded-full border border-line px-3 py-1.5 text-xs text-muted transition hover:bg-panel-3 hover:text-fg disabled:opacity-50"
            >
              <RotateCcw className="size-3.5" />
              Default
            </button>
          )}
          <button
            type="button"
            onClick={onShuffle}
            disabled={busy}
            className="inline-flex items-center gap-1.5 rounded-full border border-gemini-blue/45 bg-gemini-blue/15 px-3 py-1.5 text-xs font-medium text-gemini-sky transition hover:bg-gemini-blue/25 disabled:opacity-50"
          >
            <Shuffle className={cn('size-3.5', busy && 'animate-pulse')} />
            {layoutSeed === null ? 'Shuffle layout' : 'Shuffle again'}
          </button>
        </div>
      </div>

      <ul className="flex flex-wrap gap-1.5" aria-label="Objects in the scene">
        <Chip thumb={target.thumbnail_url} name={target.name} note="target" tone="target" />
        {fromTask.map((o) => (
          <Chip
            key={o.body}
            thumb={byId[o.object_id]?.thumbnail_url}
            name={o.name}
            note={ROLE_LABEL[o.role]}
            dot={o.colour ? COLOUR_DOT[o.colour] : undefined}
          />
        ))}
        {clutter.map((id, i) => (
          <Chip
            key={`${id}-${i}`}
            thumb={byId[id]?.thumbnail_url}
            name={byId[id]?.name ?? id}
            note={ROLE_LABEL.clutter}
            onRemove={() => onChange(clutter.filter((_, k) => k !== i))}
          />
        ))}
      </ul>

      <div className="mt-3.5 flex items-center justify-between gap-2">
        <p className="text-[11px] font-medium uppercase tracking-wider text-faint">Add objects to the table</p>
        <span className="text-[11px] text-faint">
          {clutter.length}/{MAX_CLUTTER}
          {clutter.length > 0 && (
            <button type="button" onClick={() => onChange([])} className="ml-2 text-gemini-sky hover:underline">
              clear
            </button>
          )}
        </span>
      </div>
      <div className="mt-2 flex gap-1.5 overflow-x-auto pb-1.5" role="list">
        {candidates.map((o) => (
          <button
            key={o.id}
            type="button"
            role="listitem"
            disabled={full}
            onClick={() => onChange([...clutter, o.id])}
            title={full ? `At most ${MAX_CLUTTER} extra objects` : `Add a ${o.name.toLowerCase()} to the table`}
            className={cn(
              'group relative w-[68px] shrink-0 overflow-hidden rounded-xl border border-line bg-panel-2/70 text-left transition',
              full ? 'cursor-not-allowed opacity-40' : 'hover:border-gemini-blue/50 hover:bg-panel-3',
            )}
          >
            <img src={o.thumbnail_url} alt="" className="aspect-square w-full object-cover" loading="lazy" />
            <span className="absolute right-1 top-1 grid size-4 place-items-center rounded-full bg-ink/70 text-gemini-sky opacity-0 transition group-hover:opacity-100">
              <Plus className="size-3" />
            </span>
            <span className="block truncate px-1.5 py-1 text-[10px] text-muted">{o.name}</span>
          </button>
        ))}
      </div>
      <p className="mt-1.5 text-[11px] leading-relaxed text-faint">
        Objects never overlap each other, the fixtures, or the robot. The planner gets every object's position, size and
        yaw; the multi-object tasks (stack, sort, clear into the bin, pick from clutter) add their own.
      </p>
    </Panel>
  );
}

function Chip({
  thumb,
  name,
  note,
  tone,
  dot,
  onRemove,
}: {
  thumb?: string;
  name: string;
  note: string;
  tone?: 'target';
  dot?: string;
  onRemove?: () => void;
}) {
  return (
    <li
      className={cn(
        'flex min-w-0 items-center gap-2 rounded-full border py-0.5 pl-0.5 pr-2.5 text-xs',
        tone === 'target' ? 'border-gemini-blue/45 bg-gemini-blue/12' : 'border-line bg-panel-2/70',
      )}
    >
      {thumb ? <img src={thumb} alt="" className="size-6 rounded-full object-cover" /> : <span className="size-6" />}
      {dot && <span className={cn('size-2 rounded-full', dot)} aria-hidden />}
      <span className="truncate font-medium text-fg">{name}</span>
      <span className="truncate text-faint">{note}</span>
      {onRemove && (
        <button
          type="button"
          onClick={onRemove}
          aria-label={`Remove ${name}`}
          className="-mr-1 rounded-full p-0.5 text-faint transition hover:bg-panel-3 hover:text-fg"
        >
          <X className="size-3" />
        </button>
      )}
    </li>
  );
}
