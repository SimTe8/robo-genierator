import { ChevronDown, CircleAlert, CircleCheck, CircleX, Clock, Cpu, Film, Sparkles, Terminal } from 'lucide-react';
import { cn } from '../lib/cn';
import type { DecisionType, JobEvent, PlannerBrief } from '../types';
import { GeminiSparkle } from './GeminiBrand';
import { LiveTerminal } from './LiveTerminal';
import { Panel, Pill } from './ui';

const kindColor: Record<JobEvent['kind'], string> = {
  status: 'text-gemini-sky',
  terminal: 'text-faint',
  attempt: 'text-[#d4b8ff]',
  done: 'text-ok',
  error: 'text-bad',
};

const decisionTone: Record<DecisionType, 'ok' | 'run' | 'bad'> = {
  ACCEPT_TRAJECTORY: 'ok',
  REFINE_GRASP: 'run',
  GIVE_UP: 'bad',
};

export function PlannerPanel({
  brief,
  events,
  live,
  error,
  terminalOpen,
  onToggleTerminal,
  onStop,
}: {
  brief: PlannerBrief | null;
  events: JobEvent[];
  live: boolean;
  error: string | null;
  terminalOpen: boolean;
  onToggleTerminal: () => void;
  onStop?: () => void;
}) {
  if (!brief && !error) {
    return (
      <Panel
        title="Gemini 4 closed-loop execution"
        icon={<GeminiSparkle className="size-4" />}
        right={<span className="text-[11px] text-faint">Planner · MuJoCo · Video Critic</span>}
      >
        <div className="grid place-items-center rounded-2xl border border-dashed border-line/90 bg-panel-2/30 px-4 py-10 text-center">
          <div className="mb-3 grid size-11 place-items-center rounded-2xl border border-gemini-blue/30 bg-gemini-blue/10">
            <GeminiSparkle className="size-6" />
          </div>
          <p className="font-display text-sm font-semibold text-fg">Ready for closed-loop trajectory synthesis</p>
          <p className="mt-1 max-w-md text-xs leading-relaxed text-muted">
            Select a robot, object, and task above, tune loop controls, then send your prompt. A live terminal opens
            automatically alongside rollout videos and Decisions API verdicts.
          </p>
          <div className="mt-4 flex flex-wrap justify-center gap-2 text-[11px] text-muted">
            <span className="rounded-full border border-line bg-panel px-3 py-1">1. Plan Waypoints</span>
            <span className="rounded-full border border-line bg-panel px-3 py-1">2. Simulate in MuJoCo</span>
            <span className="rounded-full border border-line bg-panel px-3 py-1">3. Video Critic + Decisions API</span>
          </div>
        </div>
      </Panel>
    );
  }

  const final = events.find((e) => e.kind === 'done' || e.kind === 'error');
  const attempts = events.filter((e) => e.kind === 'attempt');
  const terminalLines = events
    .filter((e) => e.kind === 'terminal')
    .map((e) => ({ seq: e.seq, text: e.message }));
  const nonTerminalEvents = events.filter((e) => e.kind !== 'terminal');
  const latestStatus = [...events].reverse().find((e) => e.kind === 'status');
  const drResultLine =
    (typeof final?.data?.dr_result === 'string' && final.data.dr_result) ||
    [...terminalLines].reverse().find((l) => l.text.trim().startsWith('DR Result:'))?.text.trim().replace(/^DR Result:\s*/, '') ||
    null;
  const drVideoUrls: string[] = Array.isArray(final?.data?.dr_video_urls)
    ? (final.data.dr_video_urls as string[]).filter((u) => typeof u === 'string' && u.length > 0)
    : [];
  const stitchedVideoUrl =
    typeof final?.data?.stitched_video_url === 'string' && final.data.stitched_video_url
      ? final.data.stitched_video_url
      : null;
  const exitCode = typeof final?.data?.exit_code === 'number' ? final.data.exit_code : null;

  return (
    <Panel
      title="Gemini 4 closed-loop execution"
      icon={<GeminiSparkle className="size-4" spinning={live} />}
      right={
        <div className="flex items-center gap-2">
          {brief && (
            <button
              type="button"
              onClick={onToggleTerminal}
              className={cn(
                'inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-[11px] font-medium transition',
                terminalOpen
                  ? 'border-gemini-blue/50 bg-gemini-blue/15 text-gemini-sky'
                  : 'border-line bg-panel-2 text-muted hover:border-gemini-sky/40 hover:text-fg',
              )}
            >
              <Terminal className="size-3" />
              {terminalOpen ? 'Hide terminal' : `Open terminal (${terminalLines.length})`}
            </button>
          )}
          {live ? (
            <Pill tone="run" pulse>
              live stream
            </Pill>
          ) : final ? (
            <Pill tone={final.kind === 'error' ? 'bad' : 'ok'}>
              {final.kind === 'error' ? 'stopped' : 'completed'}
            </Pill>
          ) : null}
        </div>
      }
      className={cn(live && 'border-gemini-blue/40 shadow-[0_0_40px_-18px_rgba(66,133,244,0.65)]')}
    >
      {error && (
        <p className="mb-3.5 rounded-2xl border border-bad/30 bg-bad/10 p-3.5 text-xs text-bad">{error}</p>
      )}

      {/* Brief prompt + active settings header */}
      {brief && (
        <div className="mb-4 rounded-2xl border border-line/80 bg-panel-2/55 p-3.5">
          <div className="flex flex-wrap items-center gap-2 text-[11px]">
            <span className="rounded-full border border-gemini-blue/30 bg-gemini-blue/12 px-2.5 py-0.5 font-mono text-gemini-sky">
              {brief.job_id}
            </span>
            <span className="text-muted">
              {brief.robot.name} · {brief.object.name} · {brief.task.name}
            </span>
            {brief.settings && (
              <span className="ml-auto font-mono text-[10px] text-faint">
                mode={brief.settings.feedback_mode} · max={brief.settings.max_attempts} · seed={brief.settings.scene_seed}
                {brief.settings.disabled_joints.length > 0 && ` · faults=${brief.settings.disabled_joints.join(',')}`}
              </span>
            )}
          </div>
          <p className="mt-2 text-sm leading-relaxed text-fg">“{brief.prompt}”</p>
        </div>
      )}

      {/* Live thinking indicator */}
      {live && (
        <div className="mb-4 flex items-center gap-3 rounded-2xl border border-gemini-blue/35 bg-gradient-to-r from-gemini-blue/14 via-gemini-violet/10 to-transparent px-4 py-3 text-xs text-gemini-sky">
          <GeminiSparkle className="size-4" spinning />
          <span className="font-medium">{latestStatus?.message ?? 'Gemini 4 is planning and simulating…'}</span>
        </div>
      )}

      {/* Rollout Videos (shown prominently above terminal as each attempt completes) */}
      {stitchedVideoUrl && (
        <div className="mb-4 overflow-hidden rounded-2xl border border-gemini-blue/45 bg-panel-2/70">
          <div className="flex items-center justify-between border-b border-line/70 px-4 py-2.5 text-xs">
            <span className="inline-flex items-center gap-2 font-display font-semibold text-gemini-sky">
              <Sparkles className="size-3.5" /> Stitched Checkpointed Closed-Loop Rollout
            </span>
            <span className="font-mono text-[10px] text-faint">Continuous verified trajectory</span>
          </div>
          <video
            src={stitchedVideoUrl}
            controls
            autoPlay
            muted
            loop
            playsInline
            className="aspect-[8/3] w-full bg-black object-cover"
          />
        </div>
      )}

      {/* Auto-opened interactive Live Terminal with Braille turning-dots spinner */}
      {brief && terminalOpen && (
        <div className="mb-4">
          <LiveTerminal
            key={brief.job_id}
            jobId={brief.job_id}
            lines={terminalLines}
            live={live}
            activeStage={latestStatus?.message ?? null}
            exitCode={exitCode}
            onStop={onStop}
            onClose={onToggleTerminal}
          />
        </div>
      )}

      {/* Final verdict banner */}
      {final && (
        <div
          className={cn(
            'mb-4 flex items-start gap-2.5 rounded-2xl border p-3.5 text-sm',
            final.kind === 'error'
              ? 'border-bad/35 bg-bad/8 text-bad'
              : 'border-ok/35 bg-ok/8 text-ok',
          )}
        >
          {final.kind === 'error' ? (
            <CircleAlert className="mt-0.5 size-4 shrink-0" />
          ) : (
            <CircleCheck className="mt-0.5 size-4 shrink-0" />
          )}
          <span className="leading-relaxed">
            {final.message}
            {typeof final.data?.job_json === 'string' && (
              <a
                href={final.data.job_json}
                target="_blank"
                rel="noreferrer"
                className="ml-2 font-mono text-xs underline underline-offset-2"
              >
                job.json
              </a>
            )}
          </span>
        </div>
      )}

      {/* Domain Randomization Dataset Multiplication Showcase (Badge + Sample Randomized Videos) */}
      {drResultLine && (
        <div className="mb-4 overflow-hidden rounded-2xl border border-ok/45 bg-gradient-to-br from-ok/10 via-gemini-blue/8 to-panel-2/70">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-ok/25 px-4 py-3">
            <div className="flex items-center gap-2">
              <Sparkles className="size-4 text-ok" />
              <span className="font-display text-sm font-semibold text-fg">
                Randomized Expert Dataset Generated
              </span>
            </div>
            <span className="rounded-full border border-ok/40 bg-ok/15 px-3 py-0.5 font-mono text-xs font-semibold text-ok">
              {drResultLine}
            </span>
          </div>
          {drVideoUrls.length > 0 && (
            <div className="grid gap-3 p-3.5 sm:grid-cols-3">
              {drVideoUrls.map((url, i) => (
                <div key={url} className="overflow-hidden rounded-xl border border-line/80 bg-black">
                  <div className="relative">
                    <video
                      src={url}
                      controls
                      autoPlay
                      muted
                      loop
                      playsInline
                      className="aspect-[8/3] w-full bg-black object-cover"
                    />
                    <span className="pointer-events-none absolute left-2 top-2 rounded-full border border-ok/40 bg-ink/80 px-2 py-0.5 font-mono text-[10px] font-medium text-ok backdrop-blur-sm">
                      Randomized Rollout #{i + 1}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Attempt cards with rollout video + critic verdict + probabilities + simulator checks */}
      {attempts.length > 0 && (
        <div className="mb-4 grid gap-4 md:grid-cols-2">
          {attempts.map((a, idx) => (
            <article
              key={a.seq}
              className="overflow-hidden rounded-2xl border border-line/90 bg-panel-2/60 transition hover:border-gemini-blue/40"
            >
              {a.iteration?.video_url ? (
                <div className="relative bg-black">
                  <video
                    src={a.iteration.video_url}
                    controls
                    autoPlay
                    muted
                    loop
                    playsInline
                    className="aspect-[8/3] w-full bg-ink object-cover"
                  />
                  <span className="pointer-events-none absolute left-2.5 top-2.5 inline-flex items-center gap-1 rounded-full border border-white/10 bg-ink/75 px-2.5 py-0.5 font-mono text-[10px] text-fg/90 backdrop-blur-sm">
                    <Film className="size-3 text-gemini-sky" />
                    {a.iteration.chunk_id != null ? `Chunk ${a.iteration.chunk_id} · ` : ''}
                    Attempt #{a.iteration.attempt_id ?? idx + 1}
                  </span>
                </div>
              ) : null}

              <div className="p-3.5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-display text-xs font-semibold text-fg">{a.message}</span>
                  <div className="flex flex-wrap items-center gap-1.5">
                    {a.check && (
                      <Pill tone={a.check.success ? 'ok' : 'bad'}>
                        Sim {a.check.success ? 'Pass' : 'Fail'}
                      </Pill>
                    )}
                    {a.iteration && (
                      <Pill tone={decisionTone[a.iteration.decision] ?? 'muted'}>
                        {a.iteration.decision.replaceAll('_', ' ')}
                      </Pill>
                    )}
                  </div>
                </div>

                {a.iteration && (
                  <div className="mt-2 flex flex-wrap items-center gap-2.5 font-mono text-[10px] text-faint">
                    {a.iteration.model_used && (
                      <span className="inline-flex items-center gap-1">
                        <Cpu className="size-3 text-gemini-sky" />
                        {a.iteration.model_used}
                      </span>
                    )}
                    {typeof a.iteration.latency_s === 'number' && (
                      <span className="inline-flex items-center gap-1">
                        <Clock className="size-3 text-faint" />
                        {a.iteration.latency_s.toFixed(1)}s
                      </span>
                    )}
                    {typeof a.iteration.grasp_score === 'number' && (
                      <span className="rounded bg-panel-3 px-1.5 py-px text-fg">
                        score {a.iteration.grasp_score}/3
                      </span>
                    )}
                    {typeof a.iteration.decision_confidence === 'number' && (
                      <span className="rounded bg-gemini-blue/15 px-1.5 py-px text-gemini-sky">
                        conf {(a.iteration.decision_confidence * 100).toFixed(0)}%
                      </span>
                    )}
                    {a.iteration.failure_mode && a.iteration.failure_mode !== 'NONE_SUCCESS' && (
                      <span className="rounded bg-warn/15 px-1.5 py-px text-warn">
                        {a.iteration.failure_mode}
                      </span>
                    )}
                  </div>
                )}

                {a.iteration?.decision_probabilities && (
                  <div className="mt-2.5 space-y-1 rounded-xl border border-line/60 bg-ink/40 p-2.5">
                    {Object.entries(a.iteration.decision_probabilities).map(([label, prob]) => (
                      <div key={label} className="flex items-center gap-2 font-mono text-[10px]">
                        <span className="w-32 truncate text-muted">{label}</span>
                        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-panel-3">
                          <div
                            className={cn(
                              'h-full rounded-full',
                              label === 'ACCEPT_TRAJECTORY'
                                ? 'bg-ok'
                                : label === 'GIVE_UP'
                                  ? 'bg-bad'
                                  : 'bg-gemini-blue',
                            )}
                            style={{ width: `${Math.max(2, Math.min(100, prob * 100))}%` }}
                          />
                        </div>
                        <span className="w-9 text-right text-fg">{(prob * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                )}

                {a.iteration?.reason && (
                  <p className="mt-2.5 rounded-xl border border-line/70 bg-ink/45 px-3 py-2 text-xs leading-relaxed text-muted">
                    {a.iteration.reason}
                  </p>
                )}

                {a.iteration?.telemetry_summary && (
                  <p className="mt-2 font-mono text-[10px] leading-relaxed text-faint">
                    {a.iteration.telemetry_summary}
                  </p>
                )}

                {a.check && Object.keys(a.check.checks).length > 0 && (
                  <ul className="mt-2.5 grid grid-cols-2 gap-x-3 gap-y-1.5 border-t border-line/60 pt-2.5">
                    {Object.entries(a.check.checks).map(([k, ok]) => (
                      <li key={k} className="flex items-center gap-1.5 text-[11px] text-muted">
                        {ok ? (
                          <CircleCheck className="size-3.5 shrink-0 text-ok" />
                        ) : (
                          <CircleX className="size-3.5 shrink-0 text-bad" />
                        )}
                        <span className="truncate">{k.replaceAll('_', ' ')}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </article>
          ))}
        </div>
      )}

      {/* Gemini-style collapsible Brief & Structured Event Log drawers */}
      <div className="grid gap-3 lg:grid-cols-2">
        {brief && (
          <details className="group min-w-0 rounded-2xl border border-line/80 bg-panel-2/45">
            <summary className="flex cursor-pointer list-none items-center justify-between px-4 py-2.5 text-xs font-medium text-muted transition hover:text-fg">
              <span className="inline-flex items-center gap-2">
                <GeminiSparkle className="size-3.5" />
                Compiled planner brief (job.json)
              </span>
              <ChevronDown className="size-4 text-faint transition group-open:rotate-180" />
            </summary>
            <pre className="gemini-scroll max-h-80 overflow-auto border-t border-line/70 p-3.5 font-mono text-[11px] leading-relaxed text-muted">
              {JSON.stringify({ ...brief, kinematic_summary: brief.kinematic_summary.split('\n') }, null, 2)}
            </pre>
          </details>
        )}
        <details className="group min-w-0 rounded-2xl border border-line/80 bg-panel-2/45">
          <summary className="flex cursor-pointer list-none items-center justify-between px-4 py-2.5 text-xs font-medium text-muted transition hover:text-fg">
            <span className="inline-flex items-center gap-2">
              <Terminal className="size-3.5 text-gemini-sky" />
              Structured events ({nonTerminalEvents.length})
            </span>
            <ChevronDown className="size-4 text-faint transition group-open:rotate-180" />
          </summary>
          <ol className="gemini-scroll max-h-80 overflow-auto border-t border-line/70 p-2.5 font-mono text-[11px]">
            {nonTerminalEvents.map((e) => (
              <li key={e.seq} className="flex gap-2.5 rounded-lg px-2 py-1 hover:bg-panel-3/40">
                <span className="w-5 shrink-0 text-right text-faint">{e.seq}</span>
                <span className={cn('w-14 shrink-0 font-medium', kindColor[e.kind])}>{e.kind}</span>
                <span className="text-muted">{e.message}</span>
              </li>
            ))}
          </ol>
        </details>
      </div>
    </Panel>
  );
}
