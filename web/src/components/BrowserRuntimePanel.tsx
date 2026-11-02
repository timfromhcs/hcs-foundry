import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { api } from '../lib/api'
import {
  computeSafeLimits,
  detectCapabilities,
  formatBytes,
  loadBrowserModel,
  type BrowserCapabilities,
  type BrowserModel,
  type EngineHandle,
  type SafeLimits,
} from '../lib/browserRuntime'
import { Alert, Badge, Button, Card, CardHeader, Spinner } from './ui'

type State = {
  caps: BrowserCapabilities | null
  limits: SafeLimits | null
  models: BrowserModel[]
  hint: string
  error: string | null
}

/**
 * WebGPU/WASM runtime panel. It reports what the browser really offers, what that allows, and
 * refuses models that do not fit — no fake "loading" states, no CDN.
 */
export function BrowserRuntimePanel() {
  const { t } = useTranslation()
  const [state, setState] = useState<State>({ caps: null, limits: null, models: [], hint: '', error: null })
  const [selected, setSelected] = useState<string>('')
  const [engine, setEngine] = useState<EngineHandle | null>(null)
  const [busy, setBusy] = useState(false)
  const [stage, setStage] = useState<string>('')
  const [output, setOutput] = useState('')
  const [prompt, setPrompt] = useState('Explain in one sentence why a browser cannot run a 4B model.')
  const [metrics, setMetrics] = useState<string>('')
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const [caps, runtime] = await Promise.all([detectCapabilities(), api.browserRuntime()])
        if (cancelled) return
        setState({
          caps,
          limits: computeSafeLimits(caps),
          models: (runtime.models ?? []) as BrowserModel[],
          hint: String(runtime.hint ?? ''),
          error: null,
        })
        const firstReady = ((runtime.models ?? []) as BrowserModel[]).find((model) => model.ready)
        if (firstReady) setSelected(firstReady.id)
      } catch (error) {
        if (!cancelled) {
          setState((current) => ({ ...current, error: error instanceof Error ? error.message : String(error) }))
        }
      }
    })()
    return () => {
      cancelled = true
      void engine?.dispose()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const model = state.models.find((item) => item.id === selected) ?? null
  const tooBig = Boolean(model && state.limits && model.size_bytes > state.limits.maxModelBytes)

  async function load() {
    if (!model || !state.limits || tooBig) return
    setBusy(true)
    setStage(t('browser.loading'))
    setOutput('')
    setMetrics('')
    try {
      const handle = await loadBrowserModel({
        model,
        baseUrl: window.location.origin,
        dtype: state.limits.dtype,
        onProgress: (progress) => setStage(progress.percent ? `${progress.stage} ${progress.percent}%` : progress.stage),
      })
      setEngine(handle)
      setStage(t('browser.ready'))
    } catch (error) {
      setStage('')
      setState((current) => ({ ...current, error: error instanceof Error ? error.message : String(error) }))
    } finally {
      setBusy(false)
    }
  }

  async function run() {
    if (!engine || !state.limits) return
    const controller = new AbortController()
    abortRef.current = controller
    setBusy(true)
    setOutput('')
    try {
      const result = await engine.generate(prompt, {
        maxNewTokens: state.limits.maxNewTokens,
        signal: controller.signal,
        onToken: (token) => setOutput((current) => current + token),
      })
      setOutput(result.text)
      setMetrics(`${result.tokensPerSecond.toFixed(1)} tok/s · ${result.tokens} tokens · ${result.seconds.toFixed(2)} s`)
    } catch (error) {
      if ((error as Error)?.name !== 'AbortError') {
        setMetrics(error instanceof Error ? error.message : String(error))
      }
    } finally {
      setBusy(false)
      abortRef.current = null
    }
  }

  const { caps, limits } = state
  return (
    <Card>
      <CardHeader
        title={t('browser.title')}
        icon="cpu"
        subtitle={t('browser.subtitle')}
        actions={
          limits ? (
            <Badge tone={limits.supported ? (limits.backend === 'webgpu' ? 'ok' : 'warn') : 'danger'}>
              {limits.supported ? limits.backend.toUpperCase() : t('browser.unsupported')}
            </Badge>
          ) : null
        }
      />
      <div role="region" aria-label={t('browser.title')} aria-busy={busy}>
      {state.error ? (
        <div style={{ marginBottom: 10 }}>
          <Alert tone="warn">{state.error}</Alert>
        </div>
      ) : null}
      {!caps || !limits ? (
        <Spinner />
      ) : (
        <div className="stack" style={{ gap: 12 }}>
          <div className="pill-row">
            <Badge tone={caps.webgpu.available ? 'ok' : 'plain'}>
              WebGPU: {caps.webgpu.available ? caps.webgpu.vendor || 'yes' : 'no'}
            </Badge>
            <Badge tone={caps.wasm.simd ? 'ok' : 'warn'}>WASM SIMD: {caps.wasm.simd ? 'yes' : 'no'}</Badge>
            <Badge tone={caps.wasm.threads ? 'ok' : 'plain'}>threads: {caps.wasm.threads ? 'yes' : 'no'}</Badge>
            <Badge tone="plain">{caps.cores} cores</Badge>
            {caps.deviceMemoryGb ? <Badge tone="plain">{caps.deviceMemoryGb} GB device memory</Badge> : null}
            <Badge tone={caps.secureContext ? 'ok' : 'danger'}>{caps.secureContext ? 'secure context' : 'not secure'}</Badge>
          </div>

          {limits.blockers.length > 0 ? (
            <Alert tone="danger" title={t('browser.blocked')}>
              <ul style={{ margin: 0, paddingLeft: 18 }}>
                {limits.blockers.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </Alert>
          ) : null}

          <div className="muted small">
            {t('browser.limits', {
              model: formatBytes(limits.maxModelBytes),
              context: limits.maxContext,
              dtype: limits.dtype,
              threads: limits.threads,
            })}
          </div>
          <ul className="muted small" style={{ margin: 0, paddingLeft: 18 }}>
            {limits.reasons.map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>

          {state.models.length === 0 ? (
            <Alert tone="info" title={t('browser.no_models')}>
              {state.hint}
            </Alert>
          ) : (
            <div className="stack" style={{ gap: 8 }}>
              <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
                <select
                  value={selected}
                  onChange={(event) => setSelected(event.target.value)}
                  aria-label={t('browser.pick')}
                  style={{ flex: 1, minWidth: 220, padding: '8px 10px', borderRadius: 8 }}
                >
                  {state.models.map((item) => (
                    <option key={item.id} value={item.id} disabled={!item.ready}>
                      {item.id} · {formatBytes(item.size_bytes)} {item.ready ? '' : `(${t('browser.incomplete')}: ${item.missing.join(', ')})`}
                    </option>
                  ))}
                </select>
                <Button icon="download" onClick={() => void load()} disabled={!model?.ready || tooBig || busy || !limits.supported}>
                  {t('browser.load')}
                </Button>
              </div>
              {tooBig ? (
                <Alert tone="warn">
                  {t('browser.too_big', { size: formatBytes(model?.size_bytes ?? 0), limit: formatBytes(limits.maxModelBytes) })}
                </Alert>
              ) : null}
              {stage ? (
                <div role="status" aria-live="polite" className="tiny muted">
                  {stage}
                </div>
              ) : null}

              {engine ? (
                <div className="stack" style={{ gap: 8 }}>
                  <textarea
                    value={prompt}
                    onChange={(event) => setPrompt(event.target.value)}
                    rows={2}
                    aria-label={t('browser.prompt')}
                    style={{ width: '100%', padding: 10, borderRadius: 8, font: 'inherit' }}
                  />
                  <div className="row" style={{ gap: 8 }}>
                    <Button variant="primary" icon="play" onClick={() => void run()} disabled={busy || !prompt.trim()}>
                      {t('browser.run')}
                    </Button>
                    {busy ? (
                      <Button variant="ghost" icon="stop" onClick={() => abortRef.current?.abort()}>
                        {t('common.cancel')}
                      </Button>
                    ) : null}
                    {metrics ? (
                      <span role="status" aria-live="polite" className="tiny mono">
                        {metrics}
                      </span>
                    ) : null}
                  </div>
                  <pre
                    aria-live="polite"
                    style={{ margin: 0, padding: 10, minHeight: 60, whiteSpace: 'pre-wrap', background: 'var(--surface-2)', borderRadius: 8 }}
                  >
                    {output || t('browser.empty_output')}
                  </pre>
                </div>
              ) : null}
            </div>
          )}

          <div className="tiny muted">{t('browser.honest')}</div>
        </div>
      )}
      </div>
    </Card>
  )
}
