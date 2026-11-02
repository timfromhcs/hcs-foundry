/**
 * Browser runtime: WebGPU/WASM inference with honest capability detection.
 *
 * The browser can run small models locally — that is genuinely useful for a quick check without
 * starting the native service — but it is not parity with llama.cpp: no Vulkan offload, no GGUF,
 * memory limited by the browser, and quantization is limited to what ONNX Runtime Web supports.
 * This module computes the limits from *measured* capabilities and refuses models that do not fit
 * instead of crashing the tab.
 */

export type WebGpuInfo = {
  available: boolean
  vendor?: string
  architecture?: string
  device?: string
  description?: string
  maxBufferBytes?: number
  maxStorageBufferBytes?: number
  reason?: string
}

export type WasmInfo = {
  available: boolean
  simd: boolean
  threads: boolean
  sharedArrayBuffer: boolean
}

export type BrowserCapabilities = {
  webgpu: WebGpuInfo
  wasm: WasmInfo
  cores: number
  deviceMemoryGb?: number
  storageQuotaBytes?: number
  storageUsageBytes?: number
  secureContext: boolean
  crossOriginIsolated: boolean
}

export type SafeLimits = {
  supported: boolean
  backend: 'webgpu' | 'wasm' | 'none'
  dtype: 'q4' | 'q8' | 'fp16' | 'fp32'
  maxModelBytes: number
  maxContext: number
  maxNewTokens: number
  threads: number
  reasons: string[]
  blockers: string[]
}

const MB = 1024 * 1024
const GB = 1024 * MB

function wasmFeature(bytes: number[], validate = false): boolean {
  try {
    if (validate) return WebAssembly.validate(new Uint8Array(bytes))
    // compiling (not instantiating) is enough to know the feature exists
    new WebAssembly.Module(new Uint8Array(bytes))
    return true
  } catch {
    return false
  }
}

/** Minimal valid modules that need SIMD / threads / relaxed SIMD. */
const SIMD_PROBE = [
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, 0x03,
  0x02, 0x01, 0x00, 0x07, 0x05, 0x01, 0x01, 0x66, 0x00, 0x00, 0x0a, 0x08, 0x01, 0x06, 0x00, 0x41,
  0x00, 0xfd, 0x0f, 0x0b,
]

const THREADS_PROBE = [
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x05, 0x04, 0x01, 0x03, 0x01, 0x01,
]

export async function detectCapabilities(): Promise<BrowserCapabilities> {
  const nav = navigator as Navigator & { deviceMemory?: number; gpu?: unknown }
  let webgpu: WebGpuInfo = { available: false, reason: 'navigator.gpu is missing' }
  const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<any> } }).gpu
  if (gpu && typeof gpu.requestAdapter === 'function') {
    try {
      const adapter = await gpu.requestAdapter()
      if (!adapter) {
        webgpu = { available: false, reason: 'no adapter was granted' }
      } else {
        const info = typeof adapter.info === 'object' && adapter.info ? adapter.info : {}
        const limits = adapter.limits ?? {}
        webgpu = {
          available: true,
          vendor: info.vendor || undefined,
          architecture: info.architecture || undefined,
          device: info.device || undefined,
          description: info.description || undefined,
          maxBufferBytes: Number(limits.maxBufferSize ?? 0) || undefined,
          maxStorageBufferBytes: Number(limits.maxStorageBufferBindingSize ?? 0) || undefined,
        }
      }
    } catch (error) {
      webgpu = { available: false, reason: error instanceof Error ? error.message : String(error) }
    }
  }

  const sharedArrayBuffer = typeof SharedArrayBuffer !== 'undefined'
  const wasm: WasmInfo = {
    available: typeof WebAssembly === 'object',
    simd: wasmFeature(SIMD_PROBE, true),
    threads: sharedArrayBuffer && wasmFeature(THREADS_PROBE, true),
    sharedArrayBuffer,
  }

  let storageQuotaBytes: number | undefined
  let storageUsageBytes: number | undefined
  try {
    if (navigator.storage?.estimate) {
      const estimate = await navigator.storage.estimate()
      storageQuotaBytes = estimate.quota ?? undefined
      storageUsageBytes = estimate.usage ?? undefined
    }
  } catch {
    /* storage estimation is best effort */
  }

  return {
    webgpu,
    wasm,
    cores: Math.max(1, nav.hardwareConcurrency || 1),
    deviceMemoryGb: nav.deviceMemory,
    storageQuotaBytes,
    storageUsageBytes,
    secureContext: Boolean(window.isSecureContext),
    crossOriginIsolated: Boolean((window as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated),
  }
}

/**
 * Turn capabilities into limits. Every number is explained, because a limit without a reason is
 * indistinguishable from a bug.
 */
export function computeSafeLimits(caps: BrowserCapabilities): SafeLimits {
  const reasons: string[] = []
  const blockers: string[] = []

  if (!caps.wasm.available) blockers.push('WebAssembly is unavailable — this browser cannot run a model')
  if (!caps.secureContext) blockers.push('the page is not a secure context (https or localhost is required)')

  const memoryGb = caps.deviceMemoryGb ?? 4
  // Browsers do not expose real free RAM; deviceMemory is a coarse bucket (max 8 on Chromium).
  // A local allocation probe on this machine granted ~1.5 GB to a tab, so the WASM ceiling is
  // deliberately below that and WebGPU gets the larger budget because buffers live on the GPU.
  const ceiling = caps.webgpu.available ? 3 * GB : 1.25 * GB
  const budget = Math.min(ceiling, Math.max(384 * MB, memoryGb * GB * 0.5))
  reasons.push(
    `memory budget ${Math.round(budget / MB)} MB = half of deviceMemory(${memoryGb} GB), capped at ${Math.round(ceiling / MB)} MB (${caps.webgpu.available ? 'WebGPU' : 'WASM, measured tab ceiling ~1.5 GB'})`,
  )

  let backend: SafeLimits['backend'] = 'none'
  let dtype: SafeLimits['dtype'] = 'q8'
  let maxContext = 512
  let threads = 1

  if (caps.webgpu.available) {
    backend = 'webgpu'
    dtype = 'q4'
    maxContext = 2048
    reasons.push('WebGPU adapter granted → INT4 (q4) weights and KV cache on the GPU')
    if (caps.webgpu.maxBufferBytes) {
      reasons.push(`maxBufferSize ${Math.round(caps.webgpu.maxBufferBytes / MB)} MB`)
      if (caps.webgpu.maxBufferBytes < 256 * MB) {
        blockers.push(`WebGPU maxBufferSize is only ${Math.round(caps.webgpu.maxBufferBytes / MB)} MB`)
      }
    }
    if (caps.webgpu.maxStorageBufferBytes && caps.webgpu.maxStorageBufferBytes < 128 * MB) {
      blockers.push(
        `WebGPU maxStorageBufferBindingSize is ${Math.round(caps.webgpu.maxStorageBufferBytes / MB)} MB — too small for a transformer layer`,
      )
    }
  } else if (caps.wasm.available) {
    backend = 'wasm'
    dtype = caps.wasm.simd ? 'q8' : 'fp32'
    maxContext = 512
    reasons.push(`no WebGPU (${caps.webgpu.reason ?? 'not granted'}) → WASM fallback, smaller context`)
    if (!caps.wasm.simd) reasons.push('no WASM SIMD → expect roughly 3–5× slower')
  }

  if (caps.wasm.threads && caps.crossOriginIsolated) {
    threads = Math.max(1, Math.min(4, caps.cores - 1))
    reasons.push(`WASM threads enabled (${threads} workers, crossOriginIsolated)`)
  } else if (caps.wasm.threads && !caps.crossOriginIsolated) {
    reasons.push('WASM threads are compiled in but the page is not cross-origin isolated → single thread')
  } else {
    reasons.push('no WASM threads → single thread')
  }

  const maxNewTokens = backend === 'webgpu' ? 256 : 128
  if (caps.storageQuotaBytes && caps.storageQuotaBytes < 512 * MB) {
    blockers.push(`storage quota is ${Math.round(caps.storageQuotaBytes / MB)} MB — too small to cache a model`)
  }

  return {
    supported: backend !== 'none' && blockers.length === 0,
    backend,
    dtype,
    maxModelBytes: Math.floor(budget),
    maxContext,
    maxNewTokens,
    threads,
    reasons,
    blockers,
  }
}

export type BrowserModel = {
  id: string
  size_bytes: number
  ready: boolean
  missing: string[]
  file_count: number
}

export type LoadProgress = { stage: string; percent?: number }

export type EngineHandle = {
  modelId: string
  backend: string
  dtype: string
  generate(prompt: string, options: { maxNewTokens: number; onToken?: (text: string) => void; signal?: AbortSignal }): Promise<{ text: string; seconds: number; tokensPerSecond: number; tokens: number }>
  dispose(): Promise<void>
}

/**
 * Load a model from the local service and return a generation handle.
 * Transformers.js is imported lazily so the app bundle stays small until it is used.
 */
export async function loadBrowserModel(options: {
  model: BrowserModel
  baseUrl: string
  dtype: SafeLimits['dtype']
  onProgress?: (progress: LoadProgress) => void
}): Promise<EngineHandle> {
  const { pipeline, env } = (await import('@huggingface/transformers')) as any
  // Local-first: no remote models, no CDN. Everything comes from the loopback service.
  env.allowRemoteModels = false
  env.allowLocalModels = false

  const url = `${options.baseUrl.replace(/\/$/, '')}/api/v1/browser/models/${encodeURIComponent(options.model.id)}/`
  options.onProgress?.({ stage: 'loading' })
  const generator = await pipeline('text-generation', url, {
    dtype: options.dtype,
    device: options.dtype === 'fp32' ? 'wasm' : 'auto',
    progress_callback: (event: any) => {
      if (event?.status === 'progress' && typeof event.progress === 'number') {
        options.onProgress?.({ stage: event.file ?? 'downloading', percent: Math.round(event.progress) })
      }
    },
  })

  return {
    modelId: options.model.id,
    backend: options.dtype,
    dtype: options.dtype,
    async generate(prompt, { maxNewTokens, onToken, signal }) {
      const started = performance.now()
      let text = ''
      const streamer = onToken
        ? (token: string) => {
            if (signal?.aborted) throw new DOMException('aborted', 'AbortError')
            text += token
            onToken(token)
          }
        : undefined
      const output = await generator(prompt, {
        max_new_tokens: maxNewTokens,
        do_sample: false,
        streamer,
      })
      const seconds = (performance.now() - started) / 1000
      const full = Array.isArray(output) ? output[0]?.generated_text ?? '' : String(output ?? '')
      const finalText = text || full
      const tokens = Math.max(1, Math.round(finalText.length / 4))
      return { text: finalText, seconds, tokens, tokensPerSecond: tokens / Math.max(0.001, seconds) }
    },
    async dispose() {
      try {
        await generator.dispose?.()
      } catch {
        /* disposing is best effort */
      }
    },
  }
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 MB'
  const mb = bytes / MB
  if (mb < 1024) return `${Math.round(mb)} MB`
  return `${(mb / 1024).toFixed(2)} GB`
}
