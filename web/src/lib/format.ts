export function formatBytes(bytes: number | null | undefined, digits = 1): string {
  if (bytes === null || bytes === undefined || Number.isNaN(bytes)) return '—'
  if (bytes === 0) return '0 B'
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  const exponent = Math.min(Math.floor(Math.log(Math.abs(bytes)) / Math.log(1024)), units.length - 1)
  const value = bytes / Math.pow(1024, exponent)
  return `${value.toFixed(exponent === 0 ? 0 : digits)} ${units[exponent]}`
}

export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(value)
}

export function formatPercent(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return '—'
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`
  if (seconds < 60) return `${seconds.toFixed(1)} s`
  const minutes = Math.floor(seconds / 60)
  const rest = Math.round(seconds % 60)
  if (minutes < 60) return `${minutes}m ${rest}s`
  const hours = Math.floor(minutes / 60)
  return `${hours}h ${minutes % 60}m`
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

export function timeAgo(value: string | null | undefined): string {
  if (!value) return '—'
  const date = new Date(value)
  const diff = (Date.now() - date.getTime()) / 1000
  if (Number.isNaN(diff)) return value
  if (diff < 60) return `${Math.round(diff)}s`
  if (diff < 3600) return `${Math.round(diff / 60)}m`
  if (diff < 86400) return `${Math.round(diff / 3600)}h`
  return `${Math.round(diff / 86400)}d`
}

export function jobDuration(started: string | null, finished: string | null): number | null {
  if (!started) return null
  const start = new Date(started).getTime()
  const end = finished ? new Date(finished).getTime() : Date.now()
  if (Number.isNaN(start) || Number.isNaN(end)) return null
  return (end - start) / 1000
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

export function statusTone(status: string): 'ok' | 'warn' | 'danger' | 'info' | 'plain' {
  switch (status) {
    case 'succeeded':
    case 'ready':
    case 'pass':
    case 'verified':
    case 'completed':
    case 'promoted':
    case 'confirmed':
    case 'detected':
      return 'ok'
    case 'running':
    case 'starting':
    case 'queued':
    case 'evaluated':
    case 'review':
    case 'suggested':
    case 'awaiting_approval':
      return 'info'
    case 'cancelled':
    case 'interrupted':
    case 'partial':
    case 'needs_review':
    case 'tight':
    case 'risky':
      return 'warn'
    case 'failed':
    case 'error':
    case 'too_large':
    case 'unsupported':
    case 'restricted':
    case 'excluded':
      return 'danger'
    default:
      return 'plain'
  }
}

export function fitTone(status: string): 'ok' | 'warn' | 'danger' | 'plain' {
  if (status === 'comfortable') return 'ok'
  if (status === 'tight') return 'warn'
  if (status === 'risky') return 'warn'
  if (status === 'too_large') return 'danger'
  return 'plain'
}
