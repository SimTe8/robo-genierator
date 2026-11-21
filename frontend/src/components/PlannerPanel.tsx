import { ChevronDown, CircleAlert, CircleCheck, CircleX, Send } from 'lucide-react';
import { cn } from '../lib/cn';
import type { JobEvent, PlannerBrief } from '../types';
import { Panel, Pill } from './ui';

const kindTone = { status: 'run', attempt: 'accent', done: 'ok', error: 'bad' } as const;

export function PlannerPanel({ brief, events, live, error }: {
  brief: PlannerBrief | null;
  events: JobEvent[];
  live: boolean;
  error: string | null;
}) {
  if (!brief && !error) {
    return (
      <Panel title="Gemini 4 planner">
        <div className="grid place-items-center rounded-xl border border-dashed border-line py-8 text-center">
          <Send className="mb-2 size-5 text-faint" />
          <p className="text-sm text-muted">Pick an object and a task, then send it to the planner.</p>
          <p className="mt-1 text-xs text-faint">The brief it receives and everything it reports back show up here.</p>
        </div>
      </Panel>
    );
  }

  const final = events.find((e) => e.kind === 'done' || e.kind === 'error');
  const attempts = events.filter((e) => e.kind === 'attempt');

  return (
    <Panel
      title="Gemini 4 planner"
      right={live ? <Pill tone="run" pulse>live</Pill> : final ? <Pill tone={final.kind === 'error' ? 'bad' : 'ok'}>{final.kind}</Pill> : null}
    >
      {error && <p className="mb-3 rounded-lg border border-bad/30 bg-bad/10 p-3 text-xs text-bad">{error}</p>}

      {brief && (
        <div className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1 text-xs">
          <span className="font-mono text-faint">{brief.job_id}</span>
          <span className="text-muted">
            {brief.object.name} · {brief.task.name}
          </span>
          <span className="w-full text-sm text-fg">“{brief.prompt}”</span>
        </div>
      )}

      {final && (
        <div
          className={cn(
            'mb-3 flex items-start gap-2 rounded-xl border p-3 text-sm',
            final.kind === 'error' ? 'border-bad/30 bg-bad/5 text-bad' : 'border-ok/30 bg-ok/5 text-ok',
          )}
        >
          {final.kind === 'error' ? <CircleAlert className="mt-0.5 size-4 shrink-0" /> : <CircleCheck className="mt-0.5 size-4 shrink-0" />}
          <span>
            {final.message}
            {typeof final.data?.job_json === 'string' && (
              <a href={final.data.job_json} target="_blank" rel="noreferrer" className="ml-2 underline">
                job.json
              </a>
            )}
          </span>
        </div>
      )}

      {attempts.length > 0 && (
        <div className="mb-3 grid gap-3 md:grid-cols-2">
          {attempts.map((a) => (
            <article key={a.seq} className="overflow-hidden rounded-xl border border-line bg-panel-2/50">
              {a.iteration?.video_url && (
                <video src={a.iteration.video_url} controls autoPlay muted loop playsInline className="aspect-[8/3] w-full bg-ink object-cover" />
              )}
              <div className="p-3">
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-xs font-medium">{a.message}</span>
                  {a.check && <Pill tone={a.check.success ? 'ok' : 'bad'}>sim {a.check.success ? 'success' : 'fail'}</Pill>}
                  {a.iteration && <Pill tone="muted">{a.iteration.decision.replace('_', ' ').toLowerCase()}</Pill>}
                </div>
                {a.check && (
                  <ul className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1">
                    {Object.entries(a.check.checks).map(([k, ok]) => (
                      <li key={k} className="flex items-center gap-1.5 text-[11px] text-muted">
                        {ok ? <CircleCheck className="size-3.5 text-ok" /> : <CircleX className="size-3.5 text-bad" />}
                        {k.replaceAll('_', ' ')}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </article>
          ))}
        </div>
      )}

      <div className="grid gap-3 lg:grid-cols-2">
        {brief && (
          <details className="group min-w-0 rounded-xl border border-line bg-panel-2/50">
            <summary className="flex cursor-pointer list-none items-center justify-between px-3 py-2 text-xs font-medium text-muted">
              Brief sent to the planner (job.json)
              <ChevronDown className="size-4 transition group-open:rotate-180" />
            </summary>
            <pre className="max-h-80 overflow-auto border-t border-line p-3 font-mono text-[11px] leading-relaxed text-muted">
              {JSON.stringify({ ...brief, kinematic_summary: brief.kinematic_summary.split('\n') }, null, 2)}
            </pre>
          </details>
        )}
        <details className="group min-w-0 rounded-xl border border-line bg-panel-2/50" open>
          <summary className="flex cursor-pointer list-none items-center justify-between px-3 py-2 text-xs font-medium text-muted">
            Planner log ({events.length})
            <ChevronDown className="size-4 transition group-open:rotate-180" />
          </summary>
          <ol className="max-h-80 overflow-auto border-t border-line p-2 font-mono text-[11px]">
            {events.map((e) => (
              <li key={e.seq} className="flex gap-2 px-1 py-0.5">
                <span className="w-5 shrink-0 text-right text-faint">{e.seq}</span>
                <span className={cn('w-14 shrink-0', `text-${kindTone[e.kind]}`)}>{e.kind}</span>
                <span className="text-muted">{e.message}</span>
              </li>
            ))}
          </ol>
        </details>
      </div>
    </Panel>
  );
}
