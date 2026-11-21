import { Eye, LoaderCircle, Maximize2, X } from 'lucide-react';
import { useEffect, useState } from 'react';
import type { SceneResponse } from '../types';
import { Panel } from './ui';

const fmt = (v: number[]) => `[${v.map((x) => x.toFixed(2)).join(', ')}]`;
const fixtureName: Record<string, string> = {
  target_pad: 'Green target pad',
  bin: 'Target bin',
  ghost_table: 'Target outline',
  ghost_air: 'Target orientation',
  nail_board: 'Nail board',
  sort_pads: 'Red & blue pads',
  lock: 'Lock',
  shape_board: 'Shape board',
  screw_board: 'Screw board',
  cargo: 'Block to place',
  bowl: 'Bowl',
};

export function ScenePreview({
  scene,
  loading,
  error,
  robotName,
}: {
  scene: SceneResponse | null;
  loading: boolean;
  error: string | null;
  robotName?: string;
}) {
  const s = scene?.scene;
  const [zoomed, setZoomed] = useState(false);

  useEffect(() => {
    if (!zoomed) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setZoomed(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [zoomed]);

  return (
    <>
      <Panel
        title="MuJoCo scene viewport"
        icon={<Eye className="size-4" />}
        right={
          <div className="flex items-center gap-2">
            {loading ? (
              <span className="inline-flex items-center gap-1.5 text-[11px] text-gemini-sky">
                <LoaderCircle className="size-3.5 animate-spin" /> Rendering…
              </span>
            ) : (
              <span className="text-[11px] text-faint">Side &amp; overhead cameras</span>
            )}
            {scene && (
              <button
                type="button"
                onClick={() => setZoomed(true)}
                title="Expand scene preview"
                className="grid size-7 place-items-center rounded-full border border-line bg-panel-2/80 text-muted transition hover:border-gemini-sky/40 hover:text-fg"
              >
                <Maximize2 className="size-3.5" />
              </button>
            )}
          </div>
        }
      >
        <div
          onClick={() => scene && setZoomed(true)}
          className={`group relative aspect-[8/3] overflow-hidden rounded-2xl border border-line bg-[#06080d] ${scene ? 'cursor-zoom-in' : ''} ${loading ? 'gemini-shimmer' : ''}`}
        >
          {scene && (
            <>
              <img
                src={scene.preview_url}
                alt={`MuJoCo scene with ${robotName ?? 'robot'}${s?.fixture ? ` and ${fixtureName[s.fixture] ?? s.fixture}` : ''}`}
                className={`h-full w-full object-cover transition duration-300 ${loading ? 'scale-[1.01] opacity-45' : 'group-hover:scale-[1.015]'}`}
              />
              {/* Subtle camera HUD labels */}
              <div className="pointer-events-none absolute inset-x-2.5 top-2.5 flex justify-between">
                <span className="rounded-full border border-white/10 bg-ink/70 px-2 py-0.5 font-mono text-[10px] tracking-wide text-fg/80 backdrop-blur-sm">
                  CAM 01 · SIDE
                </span>
                <span className="rounded-full border border-white/10 bg-ink/70 px-2 py-0.5 font-mono text-[10px] tracking-wide text-fg/80 backdrop-blur-sm">
                  CAM 02 · OVERHEAD
                </span>
              </div>
            </>
          )}
          {!scene && !error && (
            <div className="absolute inset-0 grid place-items-center text-xs text-muted">
              <span className="inline-flex items-center gap-2">
                <LoaderCircle className="size-4 animate-spin text-gemini-sky" />
                Rendering scene in MuJoCo…
              </span>
            </div>
          )}
          {error && <div className="absolute inset-0 grid place-items-center p-4 text-center text-xs text-bad">{error}</div>}
        </div>

        {s && (
          <dl className="mt-3 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
            <div className="rounded-xl border border-line/80 bg-panel-2/60 px-3 py-2">
              <dt className="text-[10px] uppercase tracking-wider text-faint">Object pos (m)</dt>
              <dd className="mt-0.5 font-mono text-xs text-fg">{fmt(s.object_position)}</dd>
            </div>
            <div className="rounded-xl border border-line/80 bg-panel-2/60 px-3 py-2">
              <dt className="text-[10px] uppercase tracking-wider text-faint">Yaw · mass</dt>
              <dd className="mt-0.5 font-mono text-xs text-fg">
                {s.object_yaw_deg.toFixed(0)}° · {Math.round(s.object_mass_kg * 1000)} g
              </dd>
            </div>
            <div className="rounded-xl border border-line/80 bg-panel-2/60 px-3 py-2">
              <dt className="text-[10px] uppercase tracking-wider text-faint">Table top z</dt>
              <dd className="mt-0.5 font-mono text-xs text-fg">{s.table_top_z.toFixed(3)} m</dd>
            </div>
            <div className="rounded-xl border border-line/80 bg-panel-2/60 px-3 py-2">
              <dt className="text-[10px] uppercase tracking-wider text-faint">Fixture</dt>
              <dd className="mt-0.5 truncate text-xs font-medium text-fg">
                {s.fixture ? fixtureName[s.fixture] ?? s.fixture : 'None'}
              </dd>
            </div>
          </dl>
        )}
      </Panel>

      {/* Fullscreen lightbox */}
      {zoomed && scene && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Expanded MuJoCo scene preview"
          onClick={() => setZoomed(false)}
          className="fixed inset-0 z-50 grid place-items-center bg-black/85 p-4 backdrop-blur-md"
        >
          <div onClick={(e) => e.stopPropagation()} className="relative w-full max-w-6xl">
            <button
              type="button"
              onClick={() => setZoomed(false)}
              aria-label="Close preview"
              className="absolute -top-11 right-0 inline-flex items-center gap-1.5 rounded-full border border-white/15 bg-panel/90 px-3 py-1.5 text-xs text-fg transition hover:bg-panel-2"
            >
              <X className="size-3.5" /> Close
            </button>
            <img
              src={scene.preview_url}
              alt="Expanded MuJoCo scene"
              className="w-full rounded-2xl border border-gemini-blue/35 shadow-[0_24px_80px_-12px_rgba(0,0,0,0.9)]"
            />
          </div>
        </div>
      )}
    </>
  );
}
