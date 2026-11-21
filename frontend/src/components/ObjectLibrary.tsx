import { Check, Search, X } from 'lucide-react';
import { useMemo, useState } from 'react';
import { cn } from '../lib/cn';
import type { ObjectInfo } from '../types';
import { Panel } from './ui';

export function ObjectLibrary({
  objects,
  selectedId,
  onSelect,
}: {
  objects: ObjectInfo[];
  selectedId: string;
  onSelect: (o: ObjectInfo) => void;
}) {
  const categories = useMemo(() => ['All', ...new Set(objects.map((o) => o.category))], [objects]);
  const [cat, setCat] = useState('All');
  const [query, setQuery] = useState('');

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return objects.filter((o) => {
      if (cat !== 'All' && o.category !== cat) return false;
      if (!q) return true;
      return (
        o.name.toLowerCase().includes(q) ||
        o.category.toLowerCase().includes(q) ||
        o.physics.material.toLowerCase().includes(q) ||
        o.description.toLowerCase().includes(q)
      );
    });
  }, [objects, cat, query]);

  return (
    <Panel
      title="Object library"
      step="2"
      right={
        <span className="rounded-full border border-line bg-panel-2/80 px-2.5 py-0.5 font-mono text-[11px] text-muted">
          {shown.length} / {objects.length}
        </span>
      }
    >
      {/* Search bar */}
      <div className="gemini-composer mb-3 flex items-center gap-2 rounded-full border border-line px-3.5 py-1.5">
        <Search className="size-3.5 shrink-0 text-faint" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search objects by name or material…"
          aria-label="Search objects"
          className="w-full bg-transparent text-xs text-fg placeholder:text-faint outline-none"
        />
        {query && (
          <button
            type="button"
            onClick={() => setQuery('')}
            aria-label="Clear search"
            className="rounded-full p-0.5 text-faint transition hover:bg-panel-3 hover:text-fg"
          >
            <X className="size-3.5" />
          </button>
        )}
      </div>

      {/* Category pills */}
      <div className="mb-3 flex flex-wrap gap-1.5" role="tablist" aria-label="Object categories">
        {categories.map((c) => {
          const active = cat === c;
          const count = c === 'All' ? objects.length : objects.filter((o) => o.category === c).length;
          return (
            <button
              key={c}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => setCat(c)}
              className={cn(
                'rounded-full border px-3 py-1 text-xs font-medium transition',
                active
                  ? 'border-gemini-blue/60 bg-gemini-blue/15 text-gemini-sky shadow-[0_0_16px_-6px_rgba(66,133,244,0.8)]'
                  : 'border-line/80 bg-panel-2/50 text-muted hover:border-gemini-sky/30 hover:text-fg',
              )}
            >
              {c}
              <span className={cn('ml-1.5 font-mono text-[10px]', active ? 'text-gemini-sky/80' : 'text-faint')}>
                {count}
              </span>
            </button>
          );
        })}
      </div>

      {/* Thumbnail grid */}
      {shown.length === 0 ? (
        <div className="grid place-items-center rounded-2xl border border-dashed border-line py-8 text-center text-xs text-muted">
          No objects match “{query}”.
          <button
            type="button"
            onClick={() => {
              setQuery('');
              setCat('All');
            }}
            className="mt-2 text-gemini-sky hover:underline"
          >
            Reset filter
          </button>
        </div>
      ) : (
        <div className="gemini-scroll grid max-h-[440px] grid-cols-3 gap-2.5 overflow-y-auto pr-1 sm:grid-cols-4 lg:grid-cols-3 xl:grid-cols-4">
          {shown.map((o) => {
            const on = o.id === selectedId;
            return (
              <button
                key={o.id}
                type="button"
                onClick={() => onSelect(o)}
                aria-pressed={on}
                className={cn(
                  'group relative overflow-hidden rounded-2xl border text-left transition duration-150',
                  on
                    ? 'border-gemini-blue/75 shadow-[0_0_0_1px_rgba(91,155,255,0.35),0_8px_24px_-10px_rgba(66,133,244,0.7)]'
                    : 'border-line/90 hover:border-gemini-sky/40',
                )}
              >
                <div className="relative aspect-square overflow-hidden bg-panel-2">
                  <img
                    src={o.thumbnail_url}
                    alt={o.name}
                    loading="lazy"
                    className="h-full w-full object-cover transition duration-300 group-hover:scale-105"
                  />
                  {o.articulated && (
                    <span className="absolute left-1.5 top-1.5 rounded-full border border-warn/30 bg-ink/85 px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-wide text-warn backdrop-blur-sm">
                      hinged
                    </span>
                  )}
                  {on && (
                    <span className="absolute right-1.5 top-1.5 grid size-5 place-items-center rounded-full bg-gemini-blue text-white shadow">
                      <Check className="size-3 stroke-[2.8]" />
                    </span>
                  )}
                </div>
                <div className={cn('px-2.5 py-2 transition', on ? 'bg-gemini-blue/15' : 'bg-panel-2/70 group-hover:bg-panel-2')}>
                  <div className="truncate font-display text-xs font-semibold text-fg">{o.name}</div>
                  <div className="mt-0.5 truncate font-mono text-[10px] text-faint">
                    {o.tasks.length} {o.tasks.length === 1 ? 'task' : 'tasks'} · {o.dims_cm.map((d) => Math.round(d)).join('×')} cm
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </Panel>
  );
}
