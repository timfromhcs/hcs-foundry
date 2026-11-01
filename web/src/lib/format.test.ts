import { describe, expect, it } from 'vitest'

import { fitTone, formatBytes, formatDuration, formatPercent, statusTone } from './format'

describe('format helpers', () => {
  it('formats bytes with binary units', () => {
    expect(formatBytes(0)).toBe('0 B')
    expect(formatBytes(1024)).toBe('1.0 KiB')
    expect(formatBytes(1024 ** 3)).toBe('1.0 GiB')
    expect(formatBytes(null)).toBe('—')
  })

  it('formats percentages and durations', () => {
    expect(formatPercent(0.8333)).toBe('83%')
    expect(formatPercent(null)).toBe('—')
    expect(formatDuration(0.25)).toBe('250 ms')
    expect(formatDuration(65)).toBe('1m 5s')
    expect(formatDuration(3600)).toBe('1h 0m')
  })

  it('maps statuses to tones without using colour alone', () => {
    expect(statusTone('succeeded')).toBe('ok')
    expect(statusTone('failed')).toBe('danger')
    expect(statusTone('running')).toBe('info')
    expect(statusTone('cancelled')).toBe('warn')
    expect(fitTone('comfortable')).toBe('ok')
    expect(fitTone('too_large')).toBe('danger')
  })
})
