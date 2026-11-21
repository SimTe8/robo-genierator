import { ArrowDown, Check, Copy, Maximize2, Minimize2, Square, Terminal as TerminalIcon, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { cn } from '../lib/cn';

export interface TerminalLine {
  seq: number;
  text: string;
}

function colorizeLine(text: string): string {
  const t = text.trim();
  if (t.startsWith('$ ')) return 'text-gemini-sky font-semibold';
  if (t.startsWith('[bridge]')) return 'text-warn';
  if (t.startsWith('✓ ')) return 'text-ok';
  if (t.startsWith('… ')) return 'text-gemini-sky/90';
  if (t.startsWith('--- ') && t.endsWith('---')) return 'text-[#b58ef0] font-semibold';
  if (t.startsWith('================')) return 'text-gemini-sky font-bold';
  if (t.startsWith('Decision API:')) {
    if (t.includes('ACCEPT_TRAJECTORY')) return 'text-ok font-semibold';
    if (t.includes('GIVE_UP')) return 'text-bad font-semibold';
    return 'text-gemini-sky font-semibold';
  }
  if (t.startsWith('Sim Truth:')) {
    return t.includes('sim_success=True') ? 'text-ok' : 'text-warn';
  }
  if (t.startsWith('Final Sim Success:')) {
    return t.includes('True') ? 'text-ok font-semibold' : 'text-bad font-semibold';
  }
  if (t.startsWith('DR Result:')) return 'text-ok font-semibold';
  if (t.startsWith('Traceback') || t.includes('Error:') || t.includes('Exception:')) return 'text-bad';
  if (t.startsWith('Reason:') || t.startsWith('Video:') || t.startsWith('Stitched Video:') || t.startsWith('Exported:')) {
    return 'text-fg/95';
  }
  return 'text-muted';
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

export function LiveTerminal({
  jobId,
  lines,
  live,
  exitCode,
  onStop,
  onClose,
}: {
  jobId: string;
  lines: TerminalLine[];
  live: boolean;
  exitCode?: number | null;
  onStop?: () => void;
  onClose?: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const [pinned, setPinned] = useState(true);
  const [elapsed, setElapsed] = useState(0);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!live) return;
    const id = window.setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => window.clearInterval(id);
  }, [live]);

  useEffect(() => {
    if (!pinned || !scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [lines.length, pinned, expanded]);

  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setExpanded(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [expanded]);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 36;
    setPinned(nearBottom);
  };

  const copyAll = async () => {
    await navigator.clipboard.writeText(lines.map((l) => l.text).join('\n'));
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const terminalBox = (
    <div
      className={cn(
        'flex flex-col overflow-hidden rounded-2xl border border-gemini-blue/35 bg-[#05070c] shadow-[0_18px_50px_-20px_rgba(0,0,0,0.95)]',
        expanded ? 'h-[82vh] w-full max-w-5xl' : 'w-full',
      )}
    >
      {/* Terminal titlebar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line/80 bg-[#0d111b] px-3.5 py-2">
        <div className="flex items-center gap-2.5">
          <div className="flex items-center gap-1.5" aria-hidden="true">
            <span className="size-2.5 rounded-full bg-[#ff5f56]/85" />
            <span className="size-2.5 rounded-full bg-[#ffbd2e]/85" />
            <span className="size-2.5 rounded-full bg-[#27c93f]/85" />
          </div>
          <span className="inline-flex items-center gap-1.5 font-mono text-xs font-medium text-fg">
            <TerminalIcon className="size-3.5 text-gemini-sky" />
            run_loop.py
          </span>
          <span className="hidden font-mono text-[10px] text-faint sm:inline">{jobId}</span>
        </div>

        <div className="flex items-center gap-2">
          <span className="font-mono text-[11px] text-faint">{formatElapsed(elapsed)}</span>
          {live ? (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-gemini-blue/40 bg-gemini-blue/15 px-2.5 py-0.5 font-mono text-[10px] font-medium text-gemini-sky">
              <span className="size-1.5 rounded-full bg-gemini-sky pulse-dot" />
              RUNNING
            </span>
          ) : (
            <span
              className={cn(
                'rounded-full border px-2.5 py-0.5 font-mono text-[10px] font-medium',
                exitCode === 0
                  ? 'border-ok/35 bg-ok/10 text-ok'
                  : 'border-warn/35 bg-warn/10 text-warn',
              )}
            >
              {exitCode === 0 ? 'EXITED 0' : exitCode != null ? `EXIT ${exitCode}` : 'FINISHED'}
            </span>
          )}

          {live && onStop && (
            <button
              type="button"
              onClick={onStop}
              className="inline-flex items-center gap-1 rounded-full border border-bad/45 bg-bad/12 px-2.5 py-0.5 text-[11px] font-medium text-bad transition hover:bg-bad/25"
            >
              <Square className="size-2.5 fill-current" /> Stop
            </button>
          )}

          <button
            type="button"
            onClick={copyAll}
            title="Copy terminal output"
            className="grid size-7 place-items-center rounded-lg border border-line/80 bg-panel/70 text-muted transition hover:border-gemini-sky/40 hover:text-fg"
          >
            {copied ? <Check className="size-3.5 text-ok" /> : <Copy className="size-3.5" />}
          </button>

          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            title={expanded ? 'Exit fullscreen terminal' : 'Expand terminal'}
            className="grid size-7 place-items-center rounded-lg border border-line/80 bg-panel/70 text-muted transition hover:border-gemini-sky/40 hover:text-fg"
          >
            {expanded ? <Minimize2 className="size-3.5" /> : <Maximize2 className="size-3.5" />}
          </button>

          {onClose && !expanded && (
            <button
              type="button"
              onClick={onClose}
              title="Hide terminal"
              className="grid size-7 place-items-center rounded-lg border border-line/80 bg-panel/70 text-muted transition hover:border-gemini-sky/40 hover:text-fg"
            >
              <X className="size-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Live stdout scroll area */}
      <div className="relative flex-1">
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className={cn(
            'gemini-scroll overflow-auto p-3.5 font-mono text-[12px] leading-[1.6]',
            expanded ? 'h-full' : 'max-h-[340px] min-h-[180px]',
          )}
        >
          {lines.length === 0 ? (
            <div className="flex items-center gap-2 text-faint">
              <span className="text-gemini-sky">$</span>
              <span>Starting planner-critic worker process…</span>
            </div>
          ) : (
            lines.map((line) => (
              <div key={line.seq} className={cn('whitespace-pre-wrap break-words', colorizeLine(line.text))}>
                {line.text || '\u00A0'}
              </div>
            ))
          )}
          {live && (
            <div className="mt-1 inline-flex items-center gap-1.5 text-gemini-sky">
              <span className="inline-block h-3.5 w-2 animate-pulse bg-gemini-sky/80" />
            </div>
          )}
        </div>

        {!pinned && (
          <button
            type="button"
            onClick={() => {
              setPinned(true);
              if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
            }}
            className="absolute bottom-3 right-3 inline-flex items-center gap-1 rounded-full border border-gemini-blue/45 bg-panel/95 px-3 py-1 text-[11px] font-medium text-gemini-sky shadow-lg backdrop-blur transition hover:bg-panel-2"
          >
            <ArrowDown className="size-3" /> Latest
          </button>
        )}
      </div>
    </div>
  );

  if (expanded) {
    return (
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Live planner-critic terminal"
        onClick={() => setExpanded(false)}
        className="fixed inset-0 z-50 grid place-items-center bg-black/80 p-4 backdrop-blur-md"
      >
        <div onClick={(e) => e.stopPropagation()} className="w-full max-w-5xl">
          {terminalBox}
        </div>
      </div>
    );
  }

  return terminalBox;
}
