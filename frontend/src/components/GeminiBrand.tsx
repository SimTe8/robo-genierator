import { Calendar, ExternalLink, GitBranch, MapPin, Sparkles, X } from 'lucide-react';
import { useEffect, useId } from 'react';
import { cn } from '../lib/cn';

/** Iconic Gemini 4-point concave sparkle mark with luminous blue-to-violet gradient. */
export function GeminiSparkle({
  className,
  spinning = false,
}: {
  className?: string;
  spinning?: boolean;
}) {
  const id = useId();
  return (
    <svg
      viewBox="0 0 28 28"
      fill="none"
      aria-hidden="true"
      className={cn('shrink-0', spinning && 'gemini-sparkle-spin', className)}
    >
      <defs>
        <linearGradient id={id} x1="2" y1="2" x2="26" y2="26" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#38bdf8" />
          <stop offset="48%" stopColor="#4285f4" />
          <stop offset="82%" stopColor="#9b72cb" />
          <stop offset="100%" stopColor="#d2e3fc" />
        </linearGradient>
      </defs>
      <path
        d="M14 1.5C14.7 8.9 19.1 13.3 26.5 14C19.1 14.7 14.7 19.1 14 26.5C13.3 19.1 8.9 14.7 1.5 14C8.9 13.3 13.3 8.9 14 1.5Z"
        fill={`url(#${id})`}
      />
    </svg>
  );
}

/** Compact header badge featuring the official Gemini 4 NYC Hackathon emblem. */
export function HackathonBadge({ onOpen }: { onOpen: () => void }) {
  return (
    <button
      type="button"
      onClick={onOpen}
      title="Gemini 4 NYC Hackathon · Event details"
      className="group inline-flex items-center gap-2.5 rounded-full border border-gemini-blue/30 bg-panel/90 py-1 pl-1.5 pr-3.5 text-left transition hover:border-gemini-sky/60 hover:bg-panel-2 hover:shadow-[0_0_24px_-6px_rgba(66,133,244,0.6)]"
    >
      <span className="relative h-7 w-9 overflow-hidden rounded-full border border-gemini-blue/40 bg-black shadow-inner">
        <img
          src="/gemini-4-nyc-logo.png"
          alt="Gemini 4 NYC Hackathon"
          className="h-full w-full object-cover object-center transition duration-300 group-hover:scale-110"
        />
      </span>
      <span className="leading-tight">
        <span className="flex items-center gap-1 font-display text-[11px] font-semibold tracking-tight text-fg">
          Gemini 4 NYC
          <span className="rounded bg-gemini-blue/20 px-1 py-px font-mono text-[9px] font-medium text-gemini-sky">
            2026
          </span>
        </span>
        <span className="block text-[10px] text-muted">Hackathon · Element 119</span>
      </span>
    </button>
  );
}

/** Lightbox modal showcasing the official Gemini 4 NYC Hackathon banner and track details. */
export function HackathonModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Gemini 4 NYC Hackathon"
      onClick={onClose}
      className="fixed inset-0 z-50 grid place-items-center bg-black/75 p-4 backdrop-blur-md"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="relative w-full max-w-2xl overflow-hidden rounded-3xl border border-gemini-blue/35 bg-panel shadow-[0_24px_80px_-12px_rgba(26,115,232,0.45)]"
      >
        <button
          type="button"
          onClick={onClose}
          aria-label="Close modal"
          className="absolute right-3.5 top-3.5 z-10 grid size-8 place-items-center rounded-full border border-white/15 bg-black/60 text-fg/80 backdrop-blur transition hover:bg-black/85 hover:text-white"
        >
          <X className="size-4" />
        </button>

        <div className="relative aspect-[1200/630] w-full bg-black">
          <img
            src="/gemini-4-nyc-banner.png"
            alt="Gemini 4 NYC Hackathon — Cerebral Valley"
            className="h-full w-full object-cover"
          />
          <div className="pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-panel to-transparent" />
        </div>

        <div className="p-5 sm:p-6">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-gemini-blue/35 bg-gemini-blue/15 px-3 py-0.5 text-[11px] font-medium text-gemini-sky">
              <Sparkles className="size-3" /> Problem Statement Two · Element 119
            </span>
            <span className="inline-flex items-center gap-1 text-xs text-muted">
              <Calendar className="size-3.5 text-faint" /> Nov 21, 2026
            </span>
            <span className="inline-flex items-center gap-1 text-xs text-muted">
              <MapPin className="size-3.5 text-faint" /> New York City · Cerebral Valley
            </span>
          </div>

          <h2 className="mt-3 font-display text-lg font-semibold text-fg">
            Robo Genierator — Zero-Shot Cross-Embodiment Trajectory Studio
          </h2>
          <p className="mt-1.5 text-xs leading-relaxed text-muted">
            Built live at the Gemini 4 NYC Hackathon by <strong className="text-fg">Simon Tebeck</strong> (Cornell
            University) and <strong className="text-fg">Ujwal Konanur Vinay</strong> (UMass). Combines a{' '}
            <strong className="text-fg">Gemini 4 Planner</strong>, headless{' '}
            <strong className="text-fg">MuJoCo physics rollouts</strong>, and a{' '}
            <strong className="text-fg">Gemini 4 Video Critic + Decisions API</strong> closed loop to synthesize and
            visually self-correct dexterous manipulation trajectories across robots and objects.
          </p>

          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4 text-[11px] text-faint">
            <span className="font-mono tracking-wider text-gemini-sky/80">BUILD / EXPERIMENT / SHIP</span>
            <a
              href="https://cerebralvalley.ai/e/gemini-4-nyc/details"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-gemini-sky hover:underline"
            >
              Event details <ExternalLink className="size-3" />
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}

/** Subtle Gemini 4 NYC Hackathon & Authors footer bar. */
export function HackathonFooter({ onOpenModal }: { onOpenModal: () => void }) {
  return (
    <footer className="mt-10 border-t border-line/70 pt-5">
      <div className="flex flex-col gap-2.5 text-xs text-muted">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <GeminiSparkle className="size-4" />
            <span>
              Created by <strong className="font-medium text-fg">Simon Tebeck</strong>{' '}
              <span className="text-faint">(Cornell University)</span> &amp;{' '}
              <strong className="font-medium text-fg">Ujwal Konanur Vinay</strong>{' '}
              <span className="text-faint">(UMass)</span>
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <span className="inline-flex items-center gap-1.5 text-[11px] text-faint">
              <GitBranch className="size-3 text-gemini-sky/80" />
              Open-Source Robotics Demonstration Synthesis · MuJoCo 4 + Gemini 4
            </span>
            <span className="rounded-full border border-line bg-panel px-2.5 py-0.5 text-[10px] text-muted">
              Element 119 · Multimodal Closed Loop
            </span>
          </div>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-faint">
          <span>
            Originated at the{' '}
            <button
              type="button"
              onClick={onOpenModal}
              className="font-medium text-gemini-sky transition hover:text-fg hover:underline"
            >
              Gemini 4 NYC Hackathon
            </button>{' '}
            · Cerebral Valley · Nov 21, 2026
          </span>
          <span>Apache-2.0 License</span>
        </div>
      </div>
    </footer>
  );
}
