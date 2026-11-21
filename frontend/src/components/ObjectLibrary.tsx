import { useState } from 'react';
import { cn } from '../lib/cn';
import type { ObjectInfo } from '../types';
import { Panel } from './ui';

export function ObjectLibrary({ objects, selectedId, onSelect }: {
  objects: ObjectInfo[];
  selectedId: string;
  onSelect: (o: ObjectInfo) => void;
}) {
  const categories = ['All', ...new Set(objects.map((o) => o.category))];
  const [cat, setCat] = useState('All');
  const shown = cat === 'All' ? objects : objects.filter((o) => o.category === cat);

  return (
    <Panel title="Object" step="1" right={<span className="text-[11px] text-faint">{objects.length} objects</span>}>
      <div className="mb-3 flex flex-wrap gap-1.5" role="tablist" aria-label="Object categories">
        {categories.map((c) => (
          <button
            key={c}
            type="button"
            role="tab"
            aria-selected={cat === c}
            onClick={() => setCat(c)}
            className={cn(
              'rounded-full border px-3 py-1 text-xs transition',
              cat === c ? 'border-accent/60 bg-accent/10 text-accent' : 'border-line text-muted hover:border-faint hover:text-fg',
            )}
          >
            {c}
            <span className="ml-1 text-faint">{c === 'All' ? objects.length : objects.filter((o) => o.category === c).length}</span>
          </button>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-2 sm:grid-cols-4 lg:grid-cols-3 xl:grid-cols-4">
        {shown.map((o) => {
          const on = o.id === selectedId;
          return (
            <button
              key={o.id}
              type="button"
              onClick={() => onSelect(o)}
              aria-pressed={on}
              className={cn(
                'group overflow-hidden rounded-xl border text-left transition',
                on ? 'border-accent/70 ring-1 ring-accent/30' : 'border-line hover:border-faint',
              )}
            >
              <div className="relative aspect-square bg-panel-2">
                <img src={o.thumbnail_url} alt="" loading="lazy" className="h-full w-full object-cover" />
                {o.articulated && (
                  <span className="absolute left-1.5 top-1.5 rounded bg-ink/80 px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-wide text-warn">
                    hinged
                  </span>
                )}
              </div>
              <div className={cn('px-2 py-1.5', on ? 'bg-accent/10' : 'bg-panel-2/60')}>
                <div className="truncate text-xs font-medium text-fg">{o.name}</div>
                <div className="truncate font-mono text-[10px] text-faint">
                  {o.tasks.length} tasks · {o.dims_cm.map((d) => Math.round(d)).join('×')} cm
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </Panel>
  );
}
