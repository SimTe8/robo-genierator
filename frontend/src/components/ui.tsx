import type { ReactNode } from 'react';
import { cn } from '../lib/cn';

export function Panel({ title, step, right, children, className }: {
  title: string;
  step?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn('rounded-2xl border border-line bg-panel/90 p-4 sm:p-5', className)}>
      <header className="mb-3 flex items-center justify-between gap-3">
        <h2 className="flex items-center gap-2 text-sm font-semibold tracking-wide text-fg">
          {step && (
            <span className="grid size-5 place-items-center rounded-md bg-panel-2 font-mono text-[11px] text-muted">{step}</span>
          )}
          {title}
        </h2>
        {right}
      </header>
      {children}
    </section>
  );
}

export function OptionCard({ selected, disabled, onClick, icon, title, subtitle, note }: {
  selected: boolean;
  disabled?: boolean;
  onClick: () => void;
  icon?: ReactNode;
  title: string;
  subtitle?: string;
  note?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={selected}
      className={cn(
        'group flex w-full items-start gap-3 rounded-xl border p-3 text-left transition',
        selected
          ? 'border-accent/70 bg-accent/10 ring-1 ring-accent/30'
          : 'border-line bg-panel-2/60 hover:border-faint hover:bg-panel-2',
        disabled && 'cursor-not-allowed opacity-45 hover:border-line hover:bg-panel-2/60',
      )}
    >
      {icon && (
        <span className={cn('mt-0.5 shrink-0 text-muted', selected && 'text-accent')}>{icon}</span>
      )}
      <span className="min-w-0">
        <span className="block text-sm font-medium text-fg">{title}</span>
        {subtitle && <span className="mt-0.5 block text-xs leading-snug text-muted">{subtitle}</span>}
        {note && <span className="mt-1 block text-[11px] leading-snug text-warn">{note}</span>}
      </span>
    </button>
  );
}

export function Segmented<T extends string>({ value, options, onChange }: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex rounded-lg border border-line bg-panel-2 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={cn(
            'rounded-md px-3 py-1 text-xs font-medium transition',
            value === o.value ? 'bg-accent text-accent-ink' : 'text-muted hover:text-fg',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Pill({ tone = 'muted', children, pulse }: {
  tone?: 'muted' | 'ok' | 'warn' | 'bad' | 'run' | 'accent';
  children: ReactNode;
  pulse?: boolean;
}) {
  const tones = {
    muted: 'border-line text-muted',
    ok: 'border-ok/30 text-ok',
    warn: 'border-warn/30 text-warn',
    bad: 'border-bad/30 text-bad',
    run: 'border-run/30 text-run',
    accent: 'border-accent/30 text-accent',
  };
  const dots = { muted: 'bg-faint', ok: 'bg-ok', warn: 'bg-warn', bad: 'bg-bad', run: 'bg-run', accent: 'bg-accent' };
  return (
    <span className={cn('inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-medium', tones[tone])}>
      <span className={cn('size-1.5 rounded-full', dots[tone], pulse && 'pulse-dot')} />
      {children}
    </span>
  );
}
