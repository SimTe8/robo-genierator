import { Check, KeyRound, LoaderCircle, Sparkles, X } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api } from '../api';
import type { AssetDraft, AssetsStatus } from '../types';
import { Panel, Pill } from './ui';

const EXAMPLES = ['a TV remote with a power button', 'a padlock with a U-shaped shackle', 'a wrench', 'a spray bottle with a trigger'];

/** Side quest: describe an object, Gemini designs it from primitives, we sim-test it, you add it to the library. */
export function AssetForge({ onSaved }: { onSaved: (objectId: string) => void }) {
  const [status, setStatus] = useState<AssetsStatus | null>(null);
  const [prompt, setPrompt] = useState('');
  const [busy, setBusy] = useState<'generate' | 'save' | null>(null);
  const [draft, setDraft] = useState<AssetDraft | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.assetsStatus().then(setStatus).catch(() => setStatus(null));
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
      title="Side quest · generate an object"
      right={status && <Pill tone={status.configured ? 'ok' : 'warn'}>{status.configured ? status.model : 'no API key'}</Pill>}
    >
      {status && !status.configured && (
        <p className="mb-3 flex gap-2 rounded-lg border border-warn/30 bg-warn/5 p-2.5 text-xs leading-snug text-warn">
          <KeyRound className="mt-0.5 size-3.5 shrink-0" />
          <span>
            Put <code className="font-mono">GEMINI_API_KEY=…</code> in the repo-root <code className="font-mono">.env</code> (not
            <code className="font-mono"> .env.example</code>, which is public), then restart the backend.
          </span>
        </p>
      )}
      <p className="mb-2 text-xs text-muted">Describe an object. Gemini designs it from primitives; it is checked in MuJoCo before you add it.</p>
      <div className="flex gap-2">
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && prompt.trim().length >= 3 && !busy && generate()}
          placeholder="e.g. a padlock with a U-shaped shackle"
          aria-label="Object description"
          className="min-w-0 flex-1 rounded-lg border border-line bg-ink/60 px-2.5 py-2 text-sm text-fg outline-none focus:border-accent"
        />
        <button
          type="button"
          onClick={generate}
          disabled={!status?.configured || prompt.trim().length < 3 || !!busy}
          className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-3 py-2 text-sm font-semibold text-accent-ink disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy === 'generate' ? <LoaderCircle className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
          Generate
        </button>
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {EXAMPLES.map((ex) => (
          <button key={ex} type="button" onClick={() => setPrompt(ex)} className="rounded-md border border-line px-2 py-0.5 text-[11px] text-muted hover:border-faint hover:text-fg">
            {ex}
          </button>
        ))}
      </div>
      {busy === 'generate' && <p className="mt-3 text-xs text-muted">Asking Gemini, then building and simulating the object…</p>}
      {error && <p className="mt-3 rounded-lg border border-bad/30 bg-bad/10 p-2.5 text-xs text-bad">{error}</p>}

      {draft && (
        <div className="mt-3 flex gap-3 rounded-xl border border-line bg-panel-2/60 p-3">
          <img src={draft.thumbnail_url} alt={`Generated ${draft.object.name}`} className="size-24 shrink-0 rounded-lg border border-line object-cover" />
          <div className="min-w-0 flex-1 text-xs">
            <p className="text-sm font-medium text-fg">{draft.object.name}</p>
            <p className="mt-0.5 text-muted">{draft.object.description}</p>
            <p className="mt-1 font-mono text-[10px] text-faint">
              {draft.object.dims_cm.join(' × ')} cm · {Math.round(draft.object.mass_kg * 1000)} g · {draft.object.physics.material}
              {draft.object.parts.length > 0 && ` · parts: ${draft.object.parts.join(', ')}`}
            </p>
            <p className="mt-1 text-muted">Tasks: {draft.object.tasks.map((t) => t.name).join(', ')}</p>
            {draft.warnings.length > 0 && <p className="mt-1 text-warn">{draft.warnings.join(' · ')}</p>}
            <p className="mt-1 text-faint">
              {draft.model} · {draft.attempts === 1 ? 'first try' : `${draft.attempts} tries`} · rests stably ({draft.sim.settled_drift_cm} cm drift)
            </p>
            <div className="mt-2 flex gap-2">
              <button type="button" onClick={save} disabled={!!busy} className="inline-flex items-center gap-1 rounded-lg bg-accent px-2.5 py-1 font-semibold text-accent-ink disabled:opacity-40">
                {busy === 'save' ? <LoaderCircle className="size-3.5 animate-spin" /> : <Check className="size-3.5" />} Add to library
              </button>
              <button type="button" onClick={() => setDraft(null)} className="inline-flex items-center gap-1 rounded-lg border border-line px-2.5 py-1 text-muted hover:text-fg">
                <X className="size-3.5" /> Discard
              </button>
            </div>
          </div>
        </div>
      )}
    </Panel>
  );
}
