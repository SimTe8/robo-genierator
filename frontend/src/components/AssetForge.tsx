import { Check, KeyRound, LoaderCircle, Sparkles, X } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api } from '../api';
import type { AssetDraft, AssetsStatus } from '../types';
import { GeminiSparkle } from './GeminiBrand';
import { Panel, Pill } from './ui';

const EXAMPLES = [
  'a TV remote with a power button',
  'a padlock with a U-shaped shackle',
  'a wrench',
  'a spray bottle with a trigger',
];

/** Side quest: describe an object, Gemini designs it from primitives, we sim-test it, you add it to the library. */
export function AssetForge({ onSaved }: { onSaved: (objectId: string) => void }) {
  const [status, setStatus] = useState<AssetsStatus | null>(null);
  const [prompt, setPrompt] = useState('');
  const [busy, setBusy] = useState<'generate' | 'save' | null>(null);
  const [draft, setDraft] = useState<AssetDraft | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .assetsStatus()
      .then(setStatus)
      .catch(() => setStatus(null));
  }, []);

  const generate = async () => {
    setBusy('generate');
    setError(null);
    setDraft(null);
    try {
      setDraft(await api.generateAsset(prompt));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const save = async () => {
    if (!draft) return;
    setBusy('save');
    try {
      const { object_id } = await api.saveAsset(draft.draft_id);
      setDraft(null);
      setPrompt('');
      onSaved(object_id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <Panel
      title="Generate custom object"
      icon={<GeminiSparkle className="size-4" spinning={busy === 'generate'} />}
      right={
        status && (
          <Pill tone={status.configured ? 'run' : 'warn'}>
            {status.configured ? status.model : 'no API key'}
          </Pill>
        )
      }
      className="border-gemini-blue/25 bg-gradient-to-b from-panel/95 via-panel/90 to-[#11172a]/90"
    >
      {status && !status.configured && (
        <p className="mb-3 flex gap-2 rounded-2xl border border-warn/30 bg-warn/8 p-3 text-xs leading-snug text-warn">
          <KeyRound className="mt-0.5 size-3.5 shrink-0" />
          <span>
            Add <code className="font-mono">GEMINI_API_KEY=…</code> in the repo-root <code className="font-mono">.env</code> (not
            <code className="font-mono"> .env.example</code>), then restart the backend.
          </span>
        </p>
      )}
      <p className="mb-3 text-xs leading-relaxed text-muted">
        Describe any real-world object. Gemini synthesizes it from physics primitives and verifies stability in MuJoCo before adding it to your library.
      </p>

      <div className="gemini-composer flex items-center gap-2 rounded-full border border-line p-1.5 pl-4">
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && prompt.trim().length >= 3 && !busy && status?.configured && generate()}
          placeholder="Ask Gemini to build an object…"
          aria-label="Object description"
          className="min-w-0 flex-1 bg-transparent text-sm text-fg placeholder:text-faint outline-none"
        />
        <button
          type="button"
          onClick={generate}
          disabled={!status?.configured || prompt.trim().length < 3 || !!busy}
          className="gemini-btn inline-flex items-center gap-1.5 rounded-full px-4 py-2 text-xs font-semibold text-white transition disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy === 'generate' ? <LoaderCircle className="size-3.5 animate-spin" /> : <Sparkles className="size-3.5" />}
          Generate
        </button>
      </div>

      <div className="mt-2.5 flex flex-wrap gap-1.5">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            onClick={() => setPrompt(ex)}
            className="rounded-full border border-line/80 bg-panel-2/60 px-2.5 py-1 text-[11px] text-muted transition hover:border-gemini-blue/40 hover:bg-panel-2 hover:text-gemini-sky"
          >
            {ex}
          </button>
        ))}
      </div>

      {busy === 'generate' && (
        <div className="mt-3 flex items-center gap-2.5 rounded-2xl border border-gemini-blue/25 bg-gemini-blue/10 px-3.5 py-2.5 text-xs text-gemini-sky">
          <GeminiSparkle className="size-4" spinning />
          <span>Gemini is designing primitives and running MuJoCo stability checks…</span>
        </div>
      )}

      {error && <p className="mt-3 rounded-2xl border border-bad/30 bg-bad/10 p-3 text-xs text-bad">{error}</p>}

      {draft && (
        <div className="mt-3.5 flex gap-3.5 rounded-2xl border border-gemini-blue/35 bg-panel-2/75 p-3.5">
          <img
            src={draft.thumbnail_url}
            alt={`Generated ${draft.object.name}`}
            className="size-24 shrink-0 rounded-xl border border-line object-cover"
          />
          <div className="min-w-0 flex-1 text-xs">
            <p className="font-display text-sm font-semibold text-fg">{draft.object.name}</p>
            <p className="mt-0.5 text-muted">{draft.object.description}</p>
            <p className="mt-1.5 font-mono text-[10px] text-faint">
              {draft.object.dims_cm.join(' × ')} cm · {Math.round(draft.object.mass_kg * 1000)} g · {draft.object.physics.material}
              {draft.object.parts.length > 0 && ` · parts: ${draft.object.parts.join(', ')}`}
            </p>
            <p className="mt-1 text-muted">Tasks: {draft.object.tasks.map((t) => t.name).join(', ')}</p>
            {draft.warnings.length > 0 && <p className="mt-1 text-warn">{draft.warnings.join(' · ')}</p>}
            <p className="mt-1 text-[11px] text-faint">
              {draft.model} · {draft.attempts === 1 ? 'first try' : `${draft.attempts} tries`} · rests stably ({draft.sim.settled_drift_cm} cm drift)
            </p>
            <div className="mt-2.5 flex gap-2">
              <button
                type="button"
                onClick={save}
                disabled={!!busy}
                className="gemini-btn inline-flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-xs font-semibold text-white disabled:opacity-40"
              >
                {busy === 'save' ? <LoaderCircle className="size-3.5 animate-spin" /> : <Check className="size-3.5" />}
                Add to library
              </button>
              <button
                type="button"
                onClick={() => setDraft(null)}
                className="inline-flex items-center gap-1 rounded-full border border-line px-3 py-1.5 text-xs text-muted transition hover:border-faint hover:text-fg"
              >
                <X className="size-3.5" /> Discard
              </button>
            </div>
          </div>
        </div>
      )}
    </Panel>
  );
}
