import { describe, expect, it } from 'vitest'

import { computeSafeLimits, formatBytes, type BrowserCapabilities } from './browserRuntime'

function caps(overrides: Partial<BrowserCapabilities> = {}): BrowserCapabilities {
  return {
    webgpu: { available: true, vendor: 'amd', architecture: 'rdna-2', maxBufferBytes: 1024 * 1024 * 1024, maxStorageBufferBytes: 512 * 1024 * 1024 },
    wasm: { available: true, simd: true, threads: true, sharedArrayBuffer: true },
    cores: 16,
    deviceMemoryGb: 8,
    storageQuotaBytes: 4 * 1024 * 1024 * 1024,
    storageUsageBytes: 0,
    secureContext: true,
    crossOriginIsolated: true,
    ...overrides,
  }
}

describe('computeSafeLimits', () => {
  it('prefers webgpu with q4 and a larger context', () => {
    const limits = computeSafeLimits(caps())
    expect(limits.backend).toBe('webgpu')
    expect(limits.dtype).toBe('q4')
    expect(limits.maxContext).toBe(2048)
    expect(limits.supported).toBe(true)
    expect(limits.maxModelBytes).toBeLessThanOrEqual(3 * 1024 * 1024 * 1024)
    expect(limits.reasons.some((reason) => reason.includes('memory budget'))).toBe(true)
  })

  it('falls back to wasm without webgpu and says why', () => {
    const limits = computeSafeLimits(caps({ webgpu: { available: false, reason: 'no adapter was granted' } }))
    expect(limits.backend).toBe('wasm')
    expect(limits.dtype).toBe('q8')
    expect(limits.maxContext).toBe(512)
    expect(limits.reasons.join(' ')).toContain('no adapter was granted')
  })

  it('refuses when wasm is unavailable', () => {
    const limits = computeSafeLimits(caps({ webgpu: { available: false }, wasm: { available: false, simd: false, threads: false, sharedArrayBuffer: false } }))
    expect(limits.supported).toBe(false)
    expect(limits.backend).toBe('none')
    expect(limits.blockers.join(' ')).toContain('WebAssembly is unavailable')
  })

  it('refuses an insecure context', () => {
    const limits = computeSafeLimits(caps({ secureContext: false }))
    expect(limits.supported).toBe(false)
    expect(limits.blockers.join(' ')).toContain('secure context')
  })

  it('refuses a webgpu adapter whose buffers are too small', () => {
    const limits = computeSafeLimits(
      caps({ webgpu: { available: true, maxBufferBytes: 64 * 1024 * 1024, maxStorageBufferBytes: 32 * 1024 * 1024 } }),
    )
    expect(limits.supported).toBe(false)
    expect(limits.blockers.length).toBeGreaterThanOrEqual(2)
  })

  it('keeps threads at one when the page is not cross-origin isolated', () => {
    const limits = computeSafeLimits(caps({ crossOriginIsolated: false }))
    expect(limits.threads).toBe(1)
    expect(limits.reasons.join(' ')).toContain('not cross-origin isolated')
  })

  it('scales the memory budget with deviceMemory but respects the backend ceiling', () => {
    const low = computeSafeLimits(caps({ deviceMemoryGb: 2 }))
    const high = computeSafeLimits(caps({ deviceMemoryGb: 16 }))
    expect(low.maxModelBytes).toBeGreaterThanOrEqual(384 * 1024 * 1024)
    expect(high.maxModelBytes).toBe(3 * 1024 * 1024 * 1024)
    expect(low.maxModelBytes).toBeLessThan(high.maxModelBytes)
    // a WASM-only tab gets a smaller ceiling: the measured allocation probe granted ~1.5 GB
    const wasmOnly = computeSafeLimits(caps({ deviceMemoryGb: 16, webgpu: { available: false } }))
    expect(wasmOnly.maxModelBytes).toBeLessThanOrEqual(1.25 * 1024 * 1024 * 1024)
  })

  it('refuses when the storage quota cannot hold a model', () => {
    const limits = computeSafeLimits(caps({ storageQuotaBytes: 100 * 1024 * 1024 }))
    expect(limits.supported).toBe(false)
    expect(limits.blockers.join(' ')).toContain('storage quota')
  })

  it('marks wasm without simd as slower but usable', () => {
    const limits = computeSafeLimits(
      caps({ webgpu: { available: false }, wasm: { available: true, simd: false, threads: false, sharedArrayBuffer: false } }),
    )
    expect(limits.backend).toBe('wasm')
    expect(limits.dtype).toBe('fp32')
    expect(limits.reasons.join(' ')).toContain('no WASM SIMD')
  })
})

describe('formatBytes', () => {
  it('formats megabytes and gigabytes', () => {
    expect(formatBytes(0)).toBe('0 MB')
    expect(formatBytes(512 * 1024 * 1024)).toBe('512 MB')
    expect(formatBytes(2.5 * 1024 * 1024 * 1024)).toBe('2.50 GB')
  })
})
