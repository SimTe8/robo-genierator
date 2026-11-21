import type { ReactNode } from 'react';
import { cn } from '../lib/cn';

export function Panel({
  title,
  step,
  icon,
  right,
  children,
  className,
}: {
  title: string;
  step?: string;
  icon?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={cn(
        'rounded-3xl border border-line/90 bg-panel/85 p-4 shadow-[0_12px_40px_-24px_rgba(0,0,0,0.85)] backdrop-blur-md sm:p-5',
        className,
      )}
    >
      <header className="mb-3.5 flex items-center justify-between gap-3">
        <h2 className="flex items-center gap-2.5 font-display text-[15px] font-semibold tracking-tight text-fg">
          {step && (
            <span className="grid size-6 place-items-center rounded-full border border-gemini-blue/35 bg-gemini-blue/15 font-mono text-[11px] font-medium text-gemini-sky">
              {step}
            </span>
          )}
          {icon && <span className="text-gemini-sky">{icon}</span>}
          {title}
        </h2>
        {right}
      </header>
      {children}
    </section>
  );
}

export function OptionCard({
  selected,
  disabled,
  onClick,
  icon,
  title,
  subtitle,
  note,
  badge,
}: {
  selected: boolean;
  disabled?: boolean;
  onClick: () => void;
  icon?: ReactNode;
  title: string;
  subtitle?: string;
  note?: string;
  badge?: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={selected}
      className={cn(
        'group relative flex w-full items-start gap-3 rounded-2xl border p-3 text-left transition duration-150',
        selected
          ? 'border-gemini-blue/70 bg-gradient-to-br from-gemini-blue/16 via-gemini-blue/8 to-gemini-violet/10 shadow-[0_0_0_1px_rgba(91,155,255,0.28),0_8px_24px_-12px_rgba(66,133,244,0.65)]'
          : 'border-line/90 bg-panel-2/65 hover:border-gemini-sky/35 hover:bg-panel-2',
        disabled && 'cursor-not-allowed opacity-40 hover:border-line/90 hover:bg-panel-2/65',
      )}
    >
      {icon && (
        <span
          className={cn(
            'mt-0.5 grid size-8 shrink-0 place-items-center rounded-xl border transition',
            selected
              ? 'border-gemini-blue/40 bg-gemini-blue/20 text-gemini-sky'
              : 'border-line bg-panel/80 text-muted group-hover:text-fg',
          )}
        >
          {icon}
        </span>
      )}
      <span className="min-w-0 flex-1">
        <span className="flex items-center justify-between gap-1.5">
          <span className="truncate font-display text-[13px] font-semibold text-fg">{title}</span>
          {badge}
        </span>
        {subtitle && <span className="mt-0.5 block text-xs leading-snug text-muted">{subtitle}</span>}
        {note && (
          <span className="mt-1.5 inline-flex items-center rounded-full border border-gemini-blue/25 bg-gemini-blue/10 px-2 py-0.5 text-[10px] font-medium text-gemini-sky">
            {note}
          </span>
        )}
      </span>
    </button>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex rounded-full border border-line bg-panel-2 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={cn(
            'rounded-full px-3 py-1 text-xs font-medium transition',
            value === o.value ? 'gemini-btn text-white' : 'text-muted hover:text-fg',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Pill({
  tone = 'muted',
  children,
  pulse,
  className,
}: {
  tone?: 'muted' | 'ok' | 'warn' | 'bad' | 'run' | 'accent';
  children: ReactNode;
  pulse?: boolean;
  className?: string;
}) {
  const tones = {
    muted: 'border-line bg-panel-2/70 text-muted',
    ok: 'border-ok/30 bg-ok/10 text-ok',
    warn: 'border-warn/30 bg-warn/10 text-warn',
    bad: 'border-bad/30 bg-bad/10 text-bad',
    run: 'border-gemini-blue/35 bg-gemini-blue/12 text-gemini-sky',
    accent: 'border-gemini-violet/40 bg-gemini-violet/15 text-[#d4b8ff]',
  };
  const dots = {
    muted: 'bg-faint',
    ok: 'bg-ok',
    warn: 'bg-warn',
    bad: 'bg-bad',
    run: 'bg-gemini-sky',
    accent: 'bg-gemini-violet',
  };
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-medium',
        tones[tone],
        className,
      )}
    >
      <span className={cn('size-1.5 rounded-full', dots[tone], pulse && 'pulse-dot')} />
      {children}
    </span>
  );
}
