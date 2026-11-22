import {
  BarChart3,
  Check,
  Copy,
  Database,
  Download,
  ExternalLink,
  FileCode2,
  Film,
  RefreshCw,
  ShieldCheck,
  Terminal,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from '../api';
import type {
  BenchmarkArmSummary,
  BenchmarkEpisode,
  BenchmarkReportResponse,
  DatasetExplorerResponse,
} from '../api';
import { cn } from '../lib/cn';
import { GeminiSparkle } from './GeminiBrand';
import { Panel, Pill, Segmented } from './ui';

type DashboardTab = 'datasets' | 'benchmark';

const TAB_OPTIONS: { value: DashboardTab; label: string }[] = [
  { value: 'datasets', label: 'Dataset & .npz Explorer' },
  { value: 'benchmark', label: 'Evaluation Benchmark' },
];

const ROBOT_LABELS: Record<string, string> = {
  franka_emika_panda: 'Panda 7-DoF',
  wonik_allegro: 'Allegro 16-DoF',
  wuji_hand: 'Wuji 20-DoF',
};

const ARM_LABELS: Record<string, { title: string; subtitle: string }> = {
  one_shot: {
    title: 'one_shot',
    subtitle: 'Single-shot Gemini 4 planner baseline (no critic loop)',
  },
  text_only: {
    title: 'text_only',
    subtitle: 'MuJoCo scalar telemetry feedback only (no rollout video)',
  },
  video: {
    title: 'video',
    subtitle: 'Full Multimodal Video Critic + Decisions API closed loop',
  },
};

const PYTHON_LOADER_SNIPPET = `import glob
import numpy as np

# Load physics-verified 25 Hz demonstrations for BC / Diffusion Policy / VLA fine-tuning
dataset = []
for path in sorted(glob.glob("outputs/trajectories/**/*.npz", recursive=True)):
    with np.load(path) as demo:
        dataset.append({
            "observation_qpos": demo["observation_qpos"],  # (T, D_q) joint positions [rad]
            "observation_qvel": demo["observation_qvel"] if "observation_qvel" in demo else None,
            "ee_pose": demo["ee_pose"] if "ee_pose" in demo else None,
            "action_ctrl": demo["action_ctrl"],            # (T, D_u) 25 Hz actuator targets
            "privileged_obj_pos": demo["privileged_obj_pos"] if "privileged_obj_pos" in demo else None,
        })

print(f"Loaded {len(dataset)} verified .npz rollouts (T0={len(dataset[0]['action_ctrl'])} steps @ 25 Hz)")`;

const INSPECT_CLI_CMD = 'python scripts/inspect_dataset.py --check';
const BENCHMARK_CLI_CMD = 'python backend/agents/benchmark.py --quick --workers 4';

function fmtPct(rate: number | null | undefined): string {
  if (typeof rate !== 'number' || Number.isNaN(rate)) return '—';
  const pct = rate <= 1.0001 ? rate * 100 : rate;
  return `${pct.toFixed(1)}%`;
}

function pctWidth(rate: number | null | undefined): number {
  if (typeof rate !== 'number' || Number.isNaN(rate)) return 0;
  const pct = rate <= 1.0001 ? rate * 100 : rate;
  return Math.max(0, Math.min(100, pct));
}

function formatRobotBadge(robotId: string): string {
  return ROBOT_LABELS[robotId] ?? robotId;
}

function aggregateDuosFromEpisodes(episodes: BenchmarkEpisode[]): Record<string, BenchmarkArmSummary> {
  const groups: Record<
    string,
    { episodes: number; successes: number; attempts: number; totalLatency: number; iterLatency: number }
  > = {};

  for (const ep of episodes) {
    const planner = (ep.planner_model ?? 'unknown').replace('models/', '');
    const critic =
      ep.feedback_mode === 'one_shot'
        ? 'none (one-shot)'
        : (ep.critic_model ?? 'unknown').replace('models/', '');
    const key = ep.duo && ep.duo.trim() ? ep.duo : `${planner} → ${critic}`;
    if (!groups[key]) {
      groups[key] = { episodes: 0, successes: 0, attempts: 0, totalLatency: 0, iterLatency: 0 };
    }
    const g = groups[key];
    g.episodes += 1;
    if (ep.sim_success) g.successes += 1;
    g.attempts += ep.attempts ?? 1;
    g.totalLatency += ep.total_latency_s ?? 0;
    g.iterLatency += ep.avg_iter_latency_s ?? ep.total_latency_s ?? 0;
  }

  const out: Record<string, BenchmarkArmSummary> = {};
  for (const [key, g] of Object.entries(groups)) {
    out[key] = {
      episodes: g.episodes,
      successes: g.successes,
      success_rate: g.episodes > 0 ? g.successes / g.episodes : 0,
      avg_attempts: g.episodes > 0 ? g.attempts / g.episodes : 0,
      avg_total_latency_s: g.episodes > 0 ? g.totalLatency / g.episodes : 0,
      avg_iter_latency_s: g.episodes > 0 ? g.iterLatency / g.episodes : 0,
    };
  }
  return out;
}

function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);

  const onCopy = () => {
    navigator.clipboard?.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    });
  };

  return (
    <button
      type="button"
      onClick={onCopy}
      className="inline-flex items-center gap-1.5 rounded-full border border-line bg-panel px-2.5 py-1 text-[11px] font-medium text-muted transition hover:border-gemini-sky/50 hover:text-gemini-sky"
    >
      {copied ? <Check className="size-3 text-ok" /> : <Copy className="size-3" />}
      {copied ? 'Copied' : label}
    </button>
  );
}

export function ResearchDashboard({ refreshTrigger = 0 }: { refreshTrigger?: number }) {
  const [tab, setTab] = useState<DashboardTab>('datasets');
  const [datasets, setDatasets] = useState<DatasetExplorerResponse | null>(null);
  const [benchmark, setBenchmark] = useState<BenchmarkReportResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [showAllBatches, setShowAllBatches] = useState(false);

  const fetchAll = useCallback(async () => {
    setLoading(true);
    try {
      const [d, b] = await Promise.all([api.datasets(), api.benchmark()]);
      setDatasets(d);
      setBenchmark(b);
    } catch {
      // Keep previous snapshot if backend is momentarily busy
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const delay = refreshTrigger === 0 ? 0 : 450;
    const t = setTimeout(() => {
      void fetchAll();
    }, delay);
    return () => clearTimeout(t);
  }, [fetchAll, refreshTrigger]);

  const totalDrEpisodes = useMemo(
    () => (datasets?.dr_batches ?? []).reduce((acc, b) => acc + (b.verified_episodes || b.npz_count || 0), 0),
    [datasets],
  );

  const report = benchmark?.report ?? null;

  const calibrationStats = useMemo(() => {
    if (!report) return null;
    const cal = (report.critic_calibration ?? report.calibration) as Record<string, unknown> | undefined;
    if (cal && typeof cal.agreement === 'number') {
      return {
        agreement: cal.agreement as number,
        falseAcceptRate: typeof cal.false_accept_rate === 'number' ? (cal.false_accept_rate as number) : 0,
        evalAttempts: typeof cal.eval_attempts === 'number' ? (cal.eval_attempts as number) : 0,
        falseAccepts: typeof cal.false_accepts === 'number' ? (cal.false_accepts as number) : 0,
      };
    }

    const byArm = report.summary_by_arm ?? {};
    let totalEval = 0;
    let weightedAgreement = 0;
    let totalFalseAccepts = 0;

    for (const [arm, s] of Object.entries(byArm)) {
      if (arm === 'one_shot') continue;
      const n = s.eval_attempts ?? 0;
      if (n > 0 && typeof s.critic_sim_agreement === 'number') {
        totalEval += n;
        weightedAgreement += s.critic_sim_agreement * n;
        totalFalseAccepts += s.false_accepts ?? Math.round((s.false_accept_rate ?? 0) * n);
      }
    }

    if (totalEval === 0 && Array.isArray(report.episodes)) {
      for (const ep of report.episodes) {
        const n = ep.eval_attempts ?? 0;
        if (n > 0) {
          totalEval += n;
          weightedAgreement += ep.critic_agreements ?? 0;
          totalFalseAccepts += ep.false_accepts ?? 0;
        }
      }
    }

    return {
      agreement: totalEval > 0 ? weightedAgreement / totalEval : null,
      falseAcceptRate: totalEval > 0 ? totalFalseAccepts / totalEval : null,
      evalAttempts: totalEval,
      falseAccepts: totalFalseAccepts,
    };
  }, [report]);

  const duoEntries = useMemo(() => {
    if (!report) return [];
    const direct = report.summary_by_duo ?? {};
    if (Object.keys(direct).length > 0) {
      return Object.entries(direct);
    }
    if (Array.isArray(report.episodes) && report.episodes.length > 0) {
      return Object.entries(aggregateDuosFromEpisodes(report.episodes));
    }
    return [];
  }, [report]);

  const visibleBatches = showAllBatches
    ? (datasets?.dr_batches ?? [])
    : (datasets?.dr_batches ?? []).slice(0, 6);

  return (
    <Panel
      title="Research Dataset & Evaluation Studio"
      icon={<Database className="size-4" />}
      right={
        <div className="flex flex-wrap items-center gap-2">
          <Segmented value={tab} options={TAB_OPTIONS} onChange={setTab} />
          <button
            type="button"
            onClick={() => void fetchAll()}
            disabled={loading}
            className="inline-flex items-center gap-1.5 rounded-full border border-line bg-panel-2 px-3 py-1 text-[11px] font-medium text-muted transition hover:border-gemini-sky/45 hover:text-fg disabled:opacity-50"
          >
            <RefreshCw className={cn('size-3', loading && 'animate-spin text-gemini-sky')} />
            Refresh
          </button>
        </div>
      }
    >
      {tab === 'datasets' ? (
        <div className="space-y-5">
          {/* Summary metric cards */}
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-2xl border border-ok/35 bg-gradient-to-br from-ok/12 via-panel-2/70 to-panel p-4">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-ok">
                Total Verified .npz Demos
              </span>
              <div className="mt-1.5 flex items-baseline gap-2">
                <span className="font-display text-2xl font-bold text-fg">
                  {datasets?.total_npz_files ?? '—'}
                </span>
                <span className="font-mono text-[11px] text-ok">sim-verified</span>
              </div>
              <p className="mt-1 text-[11px] text-muted">
                MuJoCo collision &amp; goal-checked expert rollouts
              </p>
            </div>

            <div className="rounded-2xl border border-gemini-blue/35 bg-gradient-to-br from-gemini-blue/12 via-panel-2/70 to-panel p-4">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-gemini-sky">
                Domain Randomization Batches
              </span>
              <div className="mt-1.5 flex items-baseline gap-2">
                <span className="font-display text-2xl font-bold text-fg">
                  {datasets?.dr_batches.length ?? '—'}
                </span>
                <span className="font-mono text-[11px] text-gemini-sky">
                  ({totalDrEpisodes} DR eps)
                </span>
              </div>
              <p className="mt-1 text-[11px] text-muted">
                Object pose, mass, friction &amp; 2-object frame perturbations
              </p>
            </div>

            <div className="rounded-2xl border border-line/80 bg-panel-2/55 p-4">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-faint">
                Embodied Robots
              </span>
              <div className="mt-2 flex flex-wrap gap-1.5">
                <Pill tone="ok">Panda 7-DoF</Pill>
                <Pill tone="run">Allegro 16-DoF</Pill>
                <Pill tone="muted">Wuji 20-DoF</Pill>
              </div>
              <p className="mt-2 text-[11px] text-muted">
                Arm + parallel jaw &amp; anthropomorphic multi-finger hands
              </p>
            </div>

            <div className="rounded-2xl border border-line/80 bg-panel-2/55 p-4">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-faint">
                Control Rate &amp; Schema
              </span>
              <div className="mt-1.5 flex items-baseline gap-2">
                <span className="font-display text-2xl font-bold text-fg">25 Hz</span>
                <span className="font-mono text-[11px] text-faint">dt = 0.04s</span>
              </div>
              <p className="mt-1 text-[11px] text-muted">
                Dense <code className="font-mono text-fg">qpos / qvel / ee_pose / ctrl</code> tensors
              </p>
            </div>
          </div>

          {/* Recent .npz trajectories list */}
          <div className="rounded-2xl border border-line/80 bg-panel-2/40">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line/70 px-4 py-3">
              <div className="flex items-center gap-2">
                <FileCode2 className="size-4 text-gemini-sky" />
                <h3 className="font-display text-xs font-semibold text-fg">
                  Recent Verified <code className="font-mono text-gemini-sky">.npz</code> Trajectories
                </h3>
              </div>
              <span className="font-mono text-[11px] text-faint">
                Showing latest {datasets?.recent_trajectories.length ?? 0} of {datasets?.total_npz_files ?? 0} files
              </span>
            </div>

            {datasets && datasets.recent_trajectories.length > 0 ? (
              <div className="divide-y divide-line/60">
                {datasets.recent_trajectories.map((traj) => (
                  <div
                    key={traj.download_url}
                    className="flex flex-wrap items-center justify-between gap-3 px-4 py-2.5 text-xs transition hover:bg-panel-3/35"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="truncate font-mono text-xs font-medium text-fg">
                          {traj.rel_path ?? traj.filename}
                        </span>
                        <span className="rounded-full border border-gemini-blue/30 bg-gemini-blue/10 px-2 py-0.5 font-mono text-[10px] text-gemini-sky">
                          {formatRobotBadge(traj.robot_id)}
                        </span>
                        <span className="rounded bg-panel-3 px-1.5 py-0.5 font-mono text-[10px] text-fg">
                          T={traj.steps} ({(traj.steps * 0.04).toFixed(2)}s)
                        </span>
                        <span className="font-mono text-[10px] text-muted">
                          qpos={traj.qpos_dim}D · ctrl={traj.ctrl_dim}D · {traj.size_kb.toFixed(1)} KB
                        </span>
                      </div>
                      {traj.keys.length > 0 && (
                        <div className="mt-1 flex flex-wrap gap-1">
                          {traj.keys.map((k) => (
                            <span
                              key={k}
                              className="rounded border border-line/70 bg-ink/55 px-1.5 py-px font-mono text-[10px] text-faint"
                            >
                              {k}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>

                    <a
                      href={traj.download_url}
                      download={traj.filename}
                      className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-ok/40 bg-ok/12 px-3 py-1 font-mono text-[11px] font-medium text-ok transition hover:border-ok/70 hover:bg-ok/20"
                    >
                      <Download className="size-3" />
                      .npz
                    </a>
                  </div>
                ))}
              </div>
            ) : (
              <div className="px-4 py-8 text-center text-xs text-muted">
                No <code className="font-mono text-fg">.npz</code> trajectories exported yet. Launch any Benchmark
                Preset above to synthesize and verify demonstrations!
              </div>
            )}
          </div>

          {/* Domain Randomization Batches */}
          {datasets && datasets.dr_batches.length > 0 && (
            <div>
              <div className="mb-2.5 flex items-center justify-between">
                <h3 className="font-display text-xs font-semibold text-fg">
                  Domain Randomization Batches ({datasets.dr_batches.length})
                </h3>
                {datasets.dr_batches.length > 6 && (
                  <button
                    type="button"
                    onClick={() => setShowAllBatches((v) => !v)}
                    className="text-[11px] font-medium text-gemini-sky hover:underline"
                  >
                    {showAllBatches ? 'Show latest 6' : `Show all ${datasets.dr_batches.length} batches`}
                  </button>
                )}
              </div>

              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {visibleBatches.map((batch) => (
                  <div
                    key={batch.batch_id}
                    className="flex flex-col justify-between rounded-2xl border border-line/85 bg-panel-2/50 p-3.5"
                  >
                    <div>
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate font-mono text-xs font-semibold text-fg">
                          {batch.batch_id}
                        </span>
                        <Pill tone={batch.success_rate >= 0.7 ? 'ok' : 'warn'}>
                          {batch.verified_episodes}/{batch.attempted_episodes} ({fmtPct(batch.success_rate)})
                        </Pill>
                      </div>
                      <p className="mt-1 text-[11px] text-muted">
                        {formatRobotBadge(batch.robot_id)} · <span className="font-mono">{batch.task}</span>
                        {batch.object_id ? ` · ${batch.object_id}` : ''}
                      </p>
                      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-panel-3">
                        <div
                          className="h-full rounded-full bg-ok"
                          style={{ width: `${pctWidth(batch.success_rate)}%` }}
                        />
                      </div>
                    </div>

                    <div className="mt-3 flex flex-wrap items-center gap-1.5 pt-1">
                      {batch.sample_video_urls.map((url, idx) => (
                        <a
                          key={url}
                          href={url}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-1 rounded-full border border-line bg-panel px-2 py-0.5 font-mono text-[10px] text-gemini-sky transition hover:border-gemini-sky/50"
                        >
                          <Film className="size-2.5" /> sample #{idx + 1}
                        </a>
                      ))}
                      {batch.manifest_url && (
                        <a
                          href={batch.manifest_url}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-1 rounded-full border border-line bg-panel px-2 py-0.5 font-mono text-[10px] text-muted hover:text-fg"
                        >
                          manifest.json <ExternalLink className="size-2.5" />
                        </a>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Researcher Python loader snippet + CLI inspector */}
          <div className="rounded-2xl border border-gemini-blue/30 bg-ink/75 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <span className="font-display text-xs font-semibold text-fg">
                  Downstream Policy Training Loader (Behavior Cloning · Diffusion Policy · VLA)
                </span>
                <p className="mt-0.5 text-[11px] text-muted">
                  Load exported <code className="font-mono text-gemini-sky">.npz</code> episodes directly into PyTorch
                  or JAX replay buffers.
                </p>
              </div>
              <CopyButton text={PYTHON_LOADER_SNIPPET} label="Copy Python" />
            </div>

            <pre className="gemini-scroll mt-3 overflow-x-auto rounded-xl border border-line/70 bg-panel/70 p-3.5 font-mono text-[11px] leading-relaxed text-fg/90">
              {PYTHON_LOADER_SNIPPET}
            </pre>

            <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-line/70 bg-panel-2/60 px-3.5 py-2">
              <div className="flex items-center gap-2 font-mono text-[11px] text-muted">
                <Terminal className="size-3.5 text-ok" />
                <span>Validate dataset schema &amp; NaN/finite checks:</span>
                <code className="rounded bg-ink px-2 py-0.5 text-ok">{INSPECT_CLI_CMD}</code>
              </div>
              <CopyButton text={INSPECT_CLI_CMD} label="Copy CLI" />
            </div>
          </div>
        </div>
      ) : (
        /* TAB 2: Evaluation Benchmark (§ 6 Ablation & Calibration) */
        <div className="space-y-5">
          {benchmark?.available && report ? (
            <>
              {/* Metadata bar */}
              <div className="flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-line/80 bg-panel-2/50 px-4 py-2.5 text-xs">
                <div className="flex flex-wrap items-center gap-2">
                  <Pill tone="ok">Live Report Loaded</Pill>
                  {report.generated_at && (
                    <span className="font-mono text-[11px] text-muted">Generated: {report.generated_at}</span>
                  )}
                  {typeof report.total_episodes === 'number' && (
                    <span className="font-mono text-[11px] text-fg">
                      · {report.total_episodes} benchmark episodes
                    </span>
                  )}
                  {report.decisions_model && (
                    <span className="rounded-full border border-gemini-blue/30 bg-gemini-blue/10 px-2.5 py-0.5 font-mono text-[10px] text-gemini-sky">
                      Decisions API: {report.decisions_model}
                    </span>
                  )}
                </div>
                {benchmark.has_v1_archive && (
                  <span className="font-mono text-[10px] text-faint">
                    v1 baseline archived (benchmark_results_v1_initial.json)
                  </span>
                )}
              </div>

              {/* Critic-Simulator Agreement & False Accept Rate Highlight */}
              {calibrationStats && (
                <div className="grid gap-3 sm:grid-cols-3">
                  <div className="rounded-2xl border border-gemini-blue/35 bg-gradient-to-br from-gemini-blue/14 via-panel-2/70 to-panel p-4">
                    <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-gemini-sky">
                      <ShieldCheck className="size-3.5" />
                      Critic–Simulator Agreement
                    </div>
                    <div className="mt-1.5 font-display text-2xl font-bold text-fg">
                      {fmtPct(calibrationStats.agreement)}
                    </div>
                    <p className="mt-1 text-[11px] text-muted">
                      Alignment between Gemini 4 Decisions API &amp; ground-truth MuJoCo checks
                    </p>
                  </div>

                  <div className="rounded-2xl border border-ok/40 bg-gradient-to-br from-ok/14 via-panel-2/70 to-panel p-4">
                    <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-ok">
                      <ShieldCheck className="size-3.5" />
                      False Accept Rate (FAR)
                    </div>
                    <div className="mt-1.5 flex items-baseline gap-2">
                      <span className="font-display text-2xl font-bold text-ok">
                        {fmtPct(calibrationStats.falseAcceptRate)}
                      </span>
                      <span className="font-mono text-[11px] text-muted">
                        ({calibrationStats.falseAccepts}/{calibrationStats.evalAttempts} attempts)
                      </span>
                    </div>
                    <p className="mt-1 text-[11px] text-muted">
                      Prevents failed rollouts from corrupting exported <code className="font-mono">.npz</code> datasets
                    </p>
                  </div>

                  <div className="rounded-2xl border border-line/80 bg-panel-2/55 p-4">
                    <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-faint">
                      <BarChart3 className="size-3.5 text-gemini-sky" />
                      Evaluated Closed-Loop Attempts
                    </div>
                    <div className="mt-1.5 flex items-baseline gap-2">
                      <span className="font-display text-2xl font-bold text-fg">
                        {calibrationStats.evalAttempts}
                      </span>
                      <span className="font-mono text-[11px] text-faint">verdicts</span>
                    </div>
                    <p className="mt-1 text-[11px] text-muted">
                      Calibrated <code className="font-mono text-fg">ACCEPT_TRAJECTORY</code> vs{' '}
                      <code className="font-mono text-fg">REFINE_GRASP</code> routing
                    </p>
                  </div>
                </div>
              )}

              {/* 3-Arm Ablation Table */}
              {report.summary_by_arm && Object.keys(report.summary_by_arm).length > 0 && (
                <div className="overflow-hidden rounded-2xl border border-line/80 bg-panel-2/40">
                  <div className="border-b border-line/70 px-4 py-3">
                    <h3 className="font-display text-xs font-semibold text-fg">
                      3-Arm Feedback Ablation (<code className="font-mono text-gemini-sky">one_shot</code> vs{' '}
                      <code className="font-mono text-gemini-sky">text_only</code> vs{' '}
                      <code className="font-mono text-gemini-sky">video</code>)
                    </h3>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead className="border-b border-line/60 bg-panel/60 font-mono text-[10px] uppercase tracking-wider text-faint">
                        <tr>
                          <th className="px-4 py-2.5">Feedback Arm</th>
                          <th className="px-4 py-2.5">Success Rate</th>
                          <th className="px-4 py-2.5">Passed</th>
                          <th className="px-4 py-2.5">Avg Loops</th>
                          <th className="px-4 py-2.5">Iter / Total Latency</th>
                          <th className="px-4 py-2.5">Critic–Sim Agree</th>
                          <th className="px-4 py-2.5">False Accept</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-line/50">
                        {(['one_shot', 'text_only', 'video'] as const)
                          .filter((arm) => report.summary_by_arm?.[arm])
                          .map((arm) => {
                            const row = report.summary_by_arm![arm];
                            const meta = ARM_LABELS[arm] ?? { title: arm, subtitle: '' };
                            const ci = row.wilson_ci ?? row.ci_95;
                            return (
                              <tr key={arm} className="transition hover:bg-panel-3/30">
                                <td className="px-4 py-3">
                                  <div className="font-mono text-xs font-semibold text-fg">{meta.title}</div>
                                  <div className="text-[11px] text-muted">{meta.subtitle}</div>
                                </td>
                                <td className="px-4 py-3">
                                  <div className="flex items-center gap-2">
                                    <div className="h-1.5 w-20 overflow-hidden rounded-full bg-panel-3">
                                      <div
                                        className={cn(
                                          'h-full rounded-full',
                                          arm === 'video' ? 'bg-ok' : 'bg-gemini-blue',
                                        )}
                                        style={{ width: `${pctWidth(row.success_rate)}%` }}
                                      />
                                    </div>
                                    <span className="font-mono font-semibold text-fg">
                                      {fmtPct(row.success_rate)}
                                    </span>
                                  </div>
                                  {ci && (
                                    <span className="font-mono text-[10px] text-faint">
                                      95% CI [{fmtPct(ci[0])}, {fmtPct(ci[1])}]
                                    </span>
                                  )}
                                </td>
                                <td className="px-4 py-3 font-mono text-muted">
                                  {row.successes ?? 0}/{row.episodes ?? 0}
                                </td>
                                <td className="px-4 py-3 font-mono text-muted">
                                  {typeof row.avg_attempts === 'number' ? row.avg_attempts.toFixed(2) : '—'}
                                </td>
                                <td className="px-4 py-3 font-mono text-muted">
                                  {typeof row.avg_iter_latency_s === 'number'
                                    ? `${row.avg_iter_latency_s.toFixed(1)}s`
                                    : '—'}{' '}
                                  /{' '}
                                  {typeof row.avg_total_latency_s === 'number'
                                    ? `${row.avg_total_latency_s.toFixed(1)}s`
                                    : '—'}
                                </td>
                                <td className="px-4 py-3 font-mono text-gemini-sky">
                                  {fmtPct(row.critic_sim_agreement)}
                                </td>
                                <td className="px-4 py-3 font-mono text-ok">
                                  {fmtPct(row.false_accept_rate)}
                                </td>
                              </tr>
                            );
                          })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Model-Duo Latency / Success Table & DR Zero-Shot Transfer Bars */}
              <div className="grid gap-4 lg:grid-cols-2">
                <div className="overflow-hidden rounded-2xl border border-line/80 bg-panel-2/40">
                  <div className="border-b border-line/70 px-4 py-3">
                    <h3 className="font-display text-xs font-semibold text-fg">
                      Model-Duo Latency &amp; Task Success
                    </h3>
                  </div>
                  {duoEntries.length > 0 ? (
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead className="border-b border-line/60 bg-panel/60 font-mono text-[10px] uppercase tracking-wider text-faint">
                          <tr>
                            <th className="px-3.5 py-2">Planner → Critic</th>
                            <th className="px-3.5 py-2">Success</th>
                            <th className="px-3.5 py-2">Loops</th>
                            <th className="px-3.5 py-2">Iter Latency</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-line/50">
                          {duoEntries.map(([duoName, d]) => (
                            <tr key={duoName} className="hover:bg-panel-3/30">
                              <td className="px-3.5 py-2.5 font-mono text-[11px] text-fg">{duoName}</td>
                              <td className="px-3.5 py-2.5 font-mono text-ok">
                                {fmtPct(d.success_rate)} ({d.successes ?? 0}/{d.episodes ?? 0})
                              </td>
                              <td className="px-3.5 py-2.5 font-mono text-muted">
                                {typeof d.avg_attempts === 'number' ? d.avg_attempts.toFixed(2) : '—'}
                              </td>
                              <td className="px-3.5 py-2.5 font-mono text-muted">
                                {typeof d.avg_iter_latency_s === 'number'
                                  ? `${d.avg_iter_latency_s.toFixed(1)}s`
                                  : '—'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="p-4 text-xs text-muted">
                      Run <code className="font-mono text-fg">--suite duo</code> to populate multi-model latency
                      comparisons.
                    </p>
                  )}
                </div>

                {/* DR Zero-Shot Transfer Bars */}
                <div className="rounded-2xl border border-line/80 bg-panel-2/40 p-4">
                  <h3 className="font-display text-xs font-semibold text-fg">
                    Domain Randomization Zero-Shot Transfer
                  </h3>
                  <p className="mt-0.5 text-[11px] text-muted">
                    Object-relative trajectory transfer across 20× randomized scene seeds
                  </p>

                  {Array.isArray(report.domain_randomization) && report.domain_randomization.length > 0 ? (
                    <div className="mt-3 space-y-2.5">
                      {report.domain_randomization.map((dr) => (
                        <div key={dr.label} className="rounded-xl border border-line/65 bg-ink/45 p-2.5">
                          <div className="flex items-center justify-between gap-2 text-[11px]">
                            <span className="truncate font-mono text-fg">{dr.label}</span>
                            <span className="shrink-0 font-mono font-semibold text-ok">
                              {dr.dr_transfer.passed}/{dr.dr_transfer.total} (
                              {fmtPct(dr.dr_transfer.transfer_rate)})
                            </span>
                          </div>
                          <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-panel-3">
                            <div
                              className="h-full rounded-full bg-gradient-to-r from-gemini-blue to-ok"
                              style={{ width: `${pctWidth(dr.dr_transfer.transfer_rate)}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="mt-3 text-xs text-muted">
                      No DR transfer rows recorded in this benchmark run.
                    </p>
                  )}
                </div>
              </div>
            </>
          ) : (
            /* Empty state with CLI command card when benchmark_results.json has not been generated yet */
            <div className="rounded-2xl border border-dashed border-line/90 bg-panel-2/35 p-6 text-center">
              <div className="mx-auto mb-3 grid size-11 place-items-center rounded-2xl border border-gemini-blue/30 bg-gemini-blue/10">
                <GeminiSparkle className="size-5" />
              </div>
              <h3 className="font-display text-sm font-semibold text-fg">
                Evaluation Benchmark Report Not Generated Yet
              </h3>
              <p className="mx-auto mt-1 max-w-lg text-xs leading-relaxed text-muted">
                Run the multi-worker evaluation harness to generate <code className="font-mono text-fg">outputs/benchmark_results.json</code>{' '}
                with 3-arm feedback ablations (<code className="font-mono">one_shot</code> vs{' '}
                <code className="font-mono">text_only</code> vs <code className="font-mono">video</code>), Critic–Simulator
                calibration, Model-Duo latency tables, and 20× DR transfer metrics.
              </p>
              <div className="mx-auto mt-4 flex max-w-xl flex-wrap items-center justify-between gap-2 rounded-xl border border-line bg-ink/80 px-4 py-2.5">
                <code className="font-mono text-xs text-gemini-sky">{BENCHMARK_CLI_CMD}</code>
                <CopyButton text={BENCHMARK_CLI_CMD} label="Copy Command" />
              </div>
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}
