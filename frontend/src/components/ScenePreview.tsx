import { LoaderCircle } from 'lucide-react';
import type { SceneResponse } from '../types';
import { Panel } from './ui';

const fmt = (v: number[]) => `[${v.map((x) => x.toFixed(2)).join(', ')}]`;
const fixtureName: Record<string, string> = {
  target_pad: 'green target pad',
  bin: 'bin',
  ghost_table: 'green target outline',
  ghost_air: 'green target orientation',
  nail_board: 'nail board',
};

export function ScenePreview({ scene, loading, error }: { scene: SceneResponse | null; loading: boolean; error: string | null }) {
  const s = scene?.scene;
  return (
    <Panel
      title="Scene"
      right={loading ? <LoaderCircle className="size-4 animate-spin text-muted" /> : <span className="text-[11px] text-faint">default conditions · side · overhead</span>}
    >
      <div className="relative aspect-[8/3] overflow-hidden rounded-xl border border-line bg-panel-2">
        {scene && (
          <img
            src={scene.preview_url}
            alt={`MuJoCo scene with the Allegro hand${s?.fixture ? ` and the ${fixtureName[s.fixture] ?? s.fixture}` : ''}`}
            className={`h-full w-full object-cover transition-opacity ${loading ? 'opacity-50' : ''}`}
          />
        )}
        {!scene && !error && <div className="absolute inset-0 grid place-items-center text-xs text-muted">Rendering in MuJoCo…</div>}
        {error && <div className="absolute inset-0 grid place-items-center p-4 text-center text-xs text-bad">{error}</div>}
      </div>
      {s && (
        <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs sm:grid-cols-4">
          <div>
            <dt className="text-faint">Object position (m)</dt>
            <dd className="font-mono text-fg">{fmt(s.object_position)}</dd>
          </div>
          <div>
            <dt className="text-faint">Yaw · mass</dt>
            <dd className="font-mono text-fg">
              {s.object_yaw_deg.toFixed(0)}° · {Math.round(s.object_mass_kg * 1000)} g
            </dd>
          </div>
          <div>
            <dt className="text-faint">Table top z</dt>
            <dd className="font-mono text-fg">{s.table_top_z.toFixed(3)}</dd>
          </div>
          <div>
            <dt className="text-faint">Fixture</dt>
            <dd className="text-fg">{s.fixture ? fixtureName[s.fixture] ?? s.fixture : '—'}</dd>
          </div>
        </dl>
      )}
    </Panel>
  );
}
