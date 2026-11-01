import { useEffect, useId, useRef, type ButtonHTMLAttributes, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes, type TextareaHTMLAttributes } from 'react'
import { createPortal } from 'react-dom'
import { useTranslation } from 'react-i18next'

import { Icon, type IconName } from './icons'

// ---------------------------------------------------------------------------
// Button
// ---------------------------------------------------------------------------

type ButtonVariant = 'default' | 'primary' | 'ghost' | 'danger'

export function Button({
  variant = 'default',
  size,
  icon,
  loading,
  block,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant
  size?: 'sm'
  icon?: IconName
  loading?: boolean
  block?: boolean
}) {
  const classes = ['btn']
  if (variant !== 'default') classes.push(variant)
  if (size) classes.push(size)
  if (block) classes.push('block')
  return (
    <button type="button" className={classes.join(' ')} {...rest} disabled={rest.disabled || loading}>
      {loading ? <span className="spinner" aria-hidden="true" /> : icon ? <Icon name={icon} size={size === 'sm' ? 14 : 16} /> : null}
      {children}
    </button>
  )
}

// ---------------------------------------------------------------------------
// Layout primitives
// ---------------------------------------------------------------------------

export function Card({
  children,
  className = '',
  accent,
  tight,
  flush,
  as: Tag = 'section',
}: {
  children: ReactNode
  className?: string
  accent?: boolean
  tight?: boolean
  flush?: boolean
  as?: 'section' | 'div' | 'article'
}) {
  const classes = ['card']
  if (accent) classes.push('accent')
  if (tight) classes.push('tight')
  if (flush) classes.push('flush')
  if (className) classes.push(className)
  return <Tag className={classes.join(' ')}>{children}</Tag>
}

export function CardHeader({
  title,
  subtitle,
  actions,
  icon,
}: {
  title: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  icon?: IconName
}) {
  return (
    <div className="card-header">
      {icon ? <Icon name={icon} size={17} style={{ marginTop: 2, color: 'var(--text-3)' }} /> : null}
      <div style={{ minWidth: 0 }}>
        <h2>{title}</h2>
        {subtitle ? <div className="sub">{subtitle}</div> : null}
      </div>
      {actions ? <div className="actions">{actions}</div> : null}
    </div>
  )
}

export function Stat({
  label,
  value,
  hint,
  small,
}: {
  label: ReactNode
  value: ReactNode
  hint?: ReactNode
  small?: boolean
}) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className={small ? 'stat-value sm' : 'stat-value'}>{value}</span>
      {hint ? <span className="stat-hint">{hint}</span> : null}
    </div>
  )
}

export function KV({ items }: { items: Array<[ReactNode, ReactNode]> }) {
  return (
    <dl className="kv">
      {items.map(([key, value], index) => (
        <div key={index} style={{ display: 'contents' }}>
          <dt>{key}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  )
}

export function Badge({
  children,
  tone = 'plain',
  dot,
  title,
}: {
  children: ReactNode
  tone?: 'ok' | 'warn' | 'danger' | 'info' | 'accent' | 'plain'
  dot?: boolean
  title?: string
}) {
  return (
    <span className={`badge ${tone}`} title={title}>
      {dot ? <span className={`dot ${tone}`} aria-hidden="true" /> : null}
      {children}
    </span>
  )
}

export function Progress({ value, indeterminate }: { value?: number; indeterminate?: boolean }) {
  const width = Math.max(0, Math.min(100, value ?? 0))
  return (
    <div className={`progress ${indeterminate ? 'indeterminate' : ''}`} role="progressbar" aria-valuenow={indeterminate ? undefined : Math.round(width)} aria-valuemin={0} aria-valuemax={100}>
      <i style={{ width: `${width}%` }} />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Form controls
// ---------------------------------------------------------------------------

export function Field({
  label,
  hint,
  error,
  htmlFor,
  children,
}: {
  label?: ReactNode
  hint?: ReactNode
  error?: ReactNode
  htmlFor?: string
  children: ReactNode
}) {
  return (
    <div className="field">
      {label ? (
        <label className="label" htmlFor={htmlFor}>
          {label}
        </label>
      ) : null}
      {children}
      {hint ? <span className="hint">{hint}</span> : null}
      {error ? <span className="error-text">{error}</span> : null}
    </div>
  )
}

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input className="input" {...props} />
}

export function Textarea(props: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea className="textarea" {...props} />
}

export function Select({ children, ...rest }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select className="select" {...rest}>
      {children}
    </select>
  )
}

export function Toggle({
  checked,
  onChange,
  label,
  description,
  disabled,
}: {
  checked: boolean
  onChange: (next: boolean) => void
  label: ReactNode
  description?: ReactNode
  disabled?: boolean
}) {
  const id = useId()
  return (
    <div className="row between" style={{ alignItems: 'flex-start', gap: 16 }}>
      <div style={{ minWidth: 0 }}>
        <label className="label" htmlFor={id} style={{ fontSize: 12.5 }}>
          {label}
        </label>
        {description ? <div className="hint" style={{ marginTop: 2 }}>{description}</div> : null}
      </div>
      <input
        id={id}
        type="checkbox"
        role="switch"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
        style={{ width: 18, height: 18, accentColor: 'var(--accent)', marginTop: 2 }}
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

export function Tabs<T extends string>({
  tabs,
  value,
  onChange,
  label,
}: {
  tabs: Array<{ id: T; label: ReactNode; badge?: ReactNode }>
  value: T
  onChange: (id: T) => void
  label?: string
}) {
  return (
    <div className="tabs" role="tablist" aria-label={label}>
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          aria-selected={value === tab.id}
          className={`tab ${value === tab.id ? 'active' : ''}`}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
          {tab.badge ? <span style={{ marginLeft: 6 }}>{tab.badge}</span> : null}
        </button>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Empty state / alerts
// ---------------------------------------------------------------------------

export function EmptyState({
  title,
  children,
  action,
  icon = 'box',
}: {
  title: ReactNode
  children?: ReactNode
  action?: ReactNode
  icon?: IconName
}) {
  return (
    <div className="empty">
      <Icon name={icon} size={26} style={{ opacity: 0.6 }} />
      <strong>{title}</strong>
      {children ? <p>{children}</p> : null}
      {action}
    </div>
  )
}

export function Alert({
  tone = 'info',
  title,
  children,
}: {
  tone?: 'ok' | 'warn' | 'danger' | 'info' | 'plain'
  title?: ReactNode
  children?: ReactNode
}) {
  return (
    <div className={`alert ${tone}`} role={tone === 'danger' ? 'alert' : undefined}>
      <Icon name={tone === 'danger' || tone === 'warn' ? 'alert' : tone === 'ok' ? 'check' : 'shield'} size={16} style={{ marginTop: 1, flex: 'none' }} />
      <div>
        {title ? <strong>{title}</strong> : null}
        {title && children ? ' ' : null}
        {children}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Modal
// ---------------------------------------------------------------------------

export function Modal({
  open,
  onClose,
  title,
  children,
  footer,
  wide,
}: {
  open: boolean
  onClose: () => void
  title: ReactNode
  children: ReactNode
  footer?: ReactNode
  wide?: boolean
}) {
  const { t } = useTranslation()
  const ref = useRef<HTMLDivElement>(null)
  const titleId = useId()

  useEffect(() => {
    if (!open) return
    const previous = document.activeElement as HTMLElement | null
    const node = ref.current
    node?.focus()
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
      if (event.key === 'Tab' && node) {
        const focusable = node.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        )
        if (focusable.length === 0) return
        const first = focusable[0]
        const last = focusable[focusable.length - 1]
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault()
          last.focus()
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault()
          first.focus()
        }
      }
    }
    document.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = ''
      previous?.focus()
    }
  }, [open, onClose])

  if (!open) return null
  return createPortal(
    <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        ref={ref}
        tabIndex={-1}
        style={wide ? { width: 'min(900px, 100%)' } : undefined}
      >
        <div className="modal-head">
          <h2 id={titleId}>{title}</h2>
          <Button variant="ghost" size="sm" icon="close" onClick={onClose} aria-label={t('common.close')} />
        </div>
        {children}
        {footer ? <div className="modal-foot">{footer}</div> : null}
      </div>
    </div>,
    document.body,
  )
}

// ---------------------------------------------------------------------------
// Copy button
// ---------------------------------------------------------------------------

export function CopyButton({ value, label }: { value: string; label?: string }) {
  const { t } = useTranslation()
  return (
    <Button
      size="sm"
      variant="ghost"
      icon="copy"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value)
        } catch {
          /* clipboard can be blocked */
        }
      }}
    >
      {label ?? t('common.copy')}
    </Button>
  )
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="row" role="status" aria-live="polite">
      <span className="spinner" />
      {label ? <span className="small muted">{label}</span> : null}
    </span>
  )
}
