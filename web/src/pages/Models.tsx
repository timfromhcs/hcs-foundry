import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'

import { useApp } from '../context/AppContext'
import { api, ApiError } from '../lib/api'
import { useAsync, useJob } from '../lib/hooks'
import { fitTone, formatBytes, formatDate, formatNumber } from '../lib/format'
import type { Fit, Model } from '../lib/types'
import { Alert, Badge, Button, Card, CardHeader, EmptyState, Field, Input, KV, Progress, Select, Spinner } from '../components/ui'

const TASKS = ['chat', 'coding', 'reasoning', 'agentic', 'vision', 'embedding', 'reranking']

export function Models() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { notify, startJob, refreshInfo } = useApp()
  const models = useAsync(() => api.models(), [])
  const [query, setQuery] = useState('')
  const [task, setTask] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [scanJobId, setScanJobId] = useState<string | null>(null)
  const scanJob = useJob(scanJobId)
  const [status, setStatus] = useState<Record<string, unknown> | null>(null)

  useEffect(() => {
    api.inferenceStatus().then(setStatus).catch(() => setStatus(null))
  }, [])

  useEffect(() => {
    if (scanJob.job?.status === 'succeeded') {
      models.reload()
      notify('ok', String(scanJob.job.output?.discovered ?? '') + ' GGUF')
      setScanJobId(null)
    }
    if (scanJob.job?.status === 'failed') {
      notify('danger', scanJob.job.error ?? t('common.error'))
      setScanJobId(null)
    }
  }, [scanJob.job?.status, scanJob.job?.error, scanJob.job?.output, models.reload, notify, t])

  const filtered = useMemo(() => {
    const list = models.data?.models ?? []
    const needle = query.trim().toLowerCase()
    return list.filter((model) => {
      if (task && model.task !== task) return false
      if (!needle) return true
      return model.name.toLowerCase().includes(needle) || model.path.toLowerCase().includes(needle)
    })
  }, [models.data, query, task])

  const selected = filtered.find((model) => model.id === selectedId) ?? filtered[0] ?? null
  const loadedId = (status?.model_id as string | undefined) ?? null

  return (
    <>
      <div className="row between">
        <div>
          <p className="muted small">{t('models.subtitle')}</p>
        </div>
        <div className="row">
          <Button icon="download" onClick={() => navigate('/hub')}>
            {t('hub.pull_from_hub')}
          </Button>
          <Button
            variant="primary"
            icon="search"
            loading={scanJob.job ? ['queued', 'running'].includes(scanJob.job.status) : false}
            onClick={async () => {
              try {
                const id = await startJob(() => api.scanModels(), t('models.scanning'))
                setScanJobId(id)
              } catch (error) {
                notify('danger', error instanceof Error ? error.message : String(error))
              }
            }}
          >
            {t('models.scan')}
          </Button>
        </div>
      </div>

      {models.data && models.data.roots.length > 0 ? (
        <div className="row" style={{ gap: 8 }}>
          <span className="tiny dim">{t('models.roots')}:</span>
          {models.data.roots.map((root) => (
            <Badge key={root} tone="plain">{root}</Badge>
          ))}
        </div>
      ) : null}

      <div className="row" style={{ gap: 10 }}>
        <div className="grow" style={{ maxWidth: 380 }}>
          <Input placeholder={t('models.search_placeholder')} value={query} onChange={(event) => setQuery(event.target.value)} aria-label={t('common.search')} />
        </div>
        <Select value={task} onChange={(event) => setTask(event.target.value)} style={{ width: 180 }} aria-label={t('models.filter_task')}>
          <option value="">{t('models.filter_task')}: {t('common.none')}</option>
          {TASKS.map((item) => (
            <option key={item} value={item}>{item}</option>
          ))}
        </Select>
        <span className="tiny dim">{filtered.length} / {models.data?.models.length ?? 0}</span>
      </div>

      {models.loading ? (
        <Card><Spinner label={t('common.loading')} /></Card>
      ) : models.error ? (
        <Alert tone="danger">{models.error}</Alert>
      ) : filtered.length === 0 ? (
        <Card>
          <EmptyState title={t('models.empty')} icon="box" action={<Button icon="search" onClick={async () => {
            try {
              const id = await startJob(() => api.scanModels(), t('models.scanning'))
              setScanJobId(id)
            } catch (error) {
              notify('danger', error instanceof Error ? error.message : String(error))
            }
          }}>{t('models.scan')}</Button>}>
            {t('models.empty_hint')}
          </EmptyState>
        </Card>
      ) : (
        <div className="grid sidebar-right">
          <Card flush>
            <div className="table-wrap">
              <table className="table responsive">
                <thead>
                  <tr>
                    <th>{t('common.name')}</th>
                    <th>{t('models.quant')}</th>
                    <th className="num">{t('models.size')}</th>
                    <th className="num">{t('models.context')}</th>
                    <th>{t('models.fit')}</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((model) => {
                    const fit = model.default_fit
                    return (
                      <tr
                        key={model.id}
                        onClick={() => setSelectedId(model.id)}
                        style={{ cursor: 'pointer', background: selected?.id === model.id ? 'var(--surface-2)' : undefined }}
                      >
                        <td data-label={t('common.name')}>
                          <div className="row" style={{ gap: 6 }}>
                            <span className="truncate" style={{ maxWidth: 260, fontWeight: 560 }}>{model.name}</span>
                            {loadedId === model.id ? <Badge tone="ok" dot>{t('models.model_loaded')}</Badge> : null}
                          </div>
                          <div className="tiny dim truncate" style={{ maxWidth: 320 }}>{model.path}</div>
                        </td>
                        <td data-label={t('models.quant')}><Badge tone="accent">{model.quantization ?? '—'}</Badge></td>
                        <td data-label={t('models.size')} className="num">{formatBytes(model.size_bytes)}</td>
                        <td data-label={t('models.context')} className="num">{formatNumber(model.context_max ?? 0)}</td>
                        <td data-label={t('models.fit')}>{fit ? <Badge tone={fitTone(fit.status)}>{t(`models.fit_${fit.status}`)}</Badge> : '—'}</td>
                        <td className="num">
                          <Button size="sm" variant="ghost" onClick={(event) => { event.stopPropagation(); setSelectedId(model.id) }}>
                            {t('models.select')}
                          </Button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>

          <div className="stack">
            {selected ? (
              <ModelDetail
                model={selected}
                loaded={loadedId === selected.id}
                status={status}
                onLoaded={async () => {
                  const next = await api.inferenceStatus()
                  setStatus(next)
                  await refreshInfo()
                }}
                onUnloaded={async () => {
                  setStatus(null)
                  await refreshInfo()
                }}
                onChanged={() => models.reload()}
                onOpenPlayground={() => navigate('/playground')}
                startJob={startJob}
                notify={notify}
              />
            ) : null}
          </div>
        </div>
      )}

      <Alert tone="info">{t('models.download_note')}</Alert>
    </>
  )
}

function ModelDetail({
  model,
  loaded,
  status,
  onLoaded,
  onUnloaded,
  onChanged,
  onOpenPlayground,
  startJob,
  notify,
}: {
  model: Model
  loaded: boolean
  status: Record<string, unknown> | null
  onLoaded: () => Promise<void>
  onUnloaded: () => Promise<void>
  onChanged: () => void
  onOpenPlayground: () => void
  startJob: (request: () => Promise<{ job_id: string }>, label: string) => Promise<string>
  notify: (kind: 'ok' | 'danger' | 'warn' | 'info', message: string) => void
}) {
  const { t } = useTranslation()
  const maxContext = Math.min(model.context_max ?? 8192, 131072)
  const [context, setContext] = useState(Math.min(4096, maxContext))
  const [gpuLayers, setGpuLayers] = useState(99)
  const [threads, setThreads] = useState(0)
  const [flash, setFlash] = useState<'auto' | 'on' | 'off'>('auto')
  const [kv, setKv] = useState('f16')
  const [fit, setFit] = useState<Fit | null>(model.default_fit ?? null)
  const [busy, setBusy] = useState(false)
  const [force, setForce] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setContext(Math.min(4096, maxContext))
    setFit(model.default_fit ?? null)
    setForce(false)
    setError(null)
  }, [model.id, maxContext, model.default_fit])

  useEffect(() => {
    let cancelled = false
    const timer = window.setTimeout(async () => {
      try {
        const data = await api.modelFit(model.id, context)
        if (!cancelled) setFit(data.fit)
      } catch {
        /* fit stays at the last value */
      }
    }, 350)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [model.id, context])

  const estimated = fit ? fit.estimated_bytes_high : model.size_bytes
  const usage = fit ? Math.min(100, (fit.estimated_bytes_high / Math.max(1, fit.safe_budget_bytes)) * 100) : 0

  const load = async (forceLoad: boolean) => {
    setBusy(true)
    setError(null)
    try {
      await api.loadModel({
        model_id: model.id,
        context,
        gpu_layers: gpuLayers,
        threads,
        flash_attn: flash,
        kv_type_k: kv,
        kv_type_v: kv,
        force: forceLoad,
      })
      notify('ok', t('models.model_loaded'))
      await onLoaded()
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message)
        const detail = err.detail as { fit?: Fit } | undefined
        if (detail?.fit) setFit(detail.fit)
      } else {
        setError(err instanceof Error ? err.message : String(err))
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <Card>
        <CardHeader title={model.name} subtitle={model.path} icon="box" actions={
          <Badge tone={model.sha256 ? 'ok' : 'plain'}>{model.sha256 ? t('common.verified') : t('models.not_verified')}</Badge>
        } />
        <div className="pill-row" style={{ marginBottom: 12 }}>
          <Badge tone="accent">{model.quantization ?? '—'}</Badge>
          <Badge tone="plain">{model.architecture ?? '—'}</Badge>
          <Badge tone="plain">{model.task ?? '—'}</Badge>
          {model.meta?.has_projector ? <Badge tone="info">{t('models.vision')}</Badge> : null}
        </div>
        <KV
          items={[
            [t('models.params'), model.parameter_count ? `${(model.parameter_count / 1e9).toFixed(2)} B` : '—'],
            [t('models.context'), formatNumber(model.context_max ?? 0)],
            [t('models.size'), formatBytes(model.size_bytes)],
            [t('models.kv_per_token'), model.meta?.gguf?.kv_bytes_per_token ? formatBytes(Number(model.meta.gguf.kv_bytes_per_token)) : '—'],
            [t('models.license'), model.license ?? <span className="dim">{t('models.unknown_license')}</span>],
            [t('models.hash'), <span className="mono tiny">{model.sha256 ? `${model.sha256.slice(0, 24)}…` : '—'}</span>],
            [t('models.verified_at'), model.verified_at ? formatDate(model.verified_at) : '—'],
          ]}
        />
        <div className="row" style={{ marginTop: 12 }}>
          <Button
            size="sm"
            variant="ghost"
            icon="check"
            onClick={async () => {
              try {
                const id = await startJob(() => api.verifyModel(model.id), t('models.verifying'))
                notify('info', t('models.verifying') + ' — ' + id)
              } catch (err) {
                notify('danger', err instanceof Error ? err.message : String(err))
              }
            }}
          >
            {t('models.verify')}
          </Button>
          <Button
            size="sm"
            variant="danger"
            icon="trash"
            disabled={loaded}
            onClick={async () => {
              if (!window.confirm(t('models.remove_confirm'))) return
              try {
                await api.deleteModel(model.id)
                notify('ok', t('models.file_not_deleted'))
                onChanged()
              } catch (err) {
                notify('danger', err instanceof Error ? err.message : String(err))
              }
            }}
          >
            {t('models.remove')}
          </Button>
        </div>
      </Card>

      <Card>
        <CardHeader title={t('models.load_config')} icon="bolt" />
        <div className="stack">
          <Field label={`${t('models.context_tokens')} — ${formatNumber(context)}`} htmlFor="ctx">
            <input
              id="ctx"
              type="range"
              min={256}
              max={maxContext}
              step={256}
              value={context}
              onChange={(event) => setContext(Number(event.target.value))}
            />
          </Field>
          <div className="grid cols-2">
            <Field label={t('models.gpu_layers')} htmlFor="ngl">
              <Input id="ngl" type="number" min={0} max={1024} value={gpuLayers} onChange={(event) => setGpuLayers(Number(event.target.value))} />
            </Field>
            <Field label={t('models.threads')} htmlFor="threads" hint={t('models.auto_hint')}>
              <Input id="threads" type="number" min={0} max={128} value={threads} onChange={(event) => setThreads(Number(event.target.value))} />
            </Field>
            <Field label={t('models.flash_attn')} htmlFor="flash">
              <Select id="flash" value={flash} onChange={(event) => setFlash(event.target.value as typeof flash)}>
                <option value="auto">{t('models.auto')}</option>
                <option value="on">{t('models.on')}</option>
                <option value="off">{t('models.off')}</option>
              </Select>
            </Field>
            <Field label={t('models.kv_cache')} htmlFor="kv">
              <Select id="kv" value={kv} onChange={(event) => setKv(event.target.value)}>
                <option value="f16">f16</option>
                <option value="q8_0">q8_0</option>
                <option value="q4_0">q4_0</option>
              </Select>
            </Field>
          </div>

          {fit ? (
            <div className="stack" style={{ gap: 6 }}>
              <div className="row between">
                <span className="small muted">{t('models.load_estimate')}</span>
                <Badge tone={fitTone(fit.status)}>{t(`models.fit_${fit.status}`)}</Badge>
              </div>
              <Progress value={usage} />
              <div className="row between tiny dim">
                <span>{formatBytes(estimated)}</span>
                <span>{formatBytes(fit.safe_budget_bytes)}</span>
              </div>
              <div className="tiny dim">
                {formatBytes(fit.weights_bytes)} {t('models.size').toLowerCase()} + {formatBytes(fit.kv_cache_bytes)} KV + {formatBytes(fit.overhead_bytes)} overhead
              </div>
              {fit.status === 'too_large' ? <Alert tone="danger">{t('models.force_warning')}</Alert> : null}
            </div>
          ) : null}

          {error ? <Alert tone="danger">{error}</Alert> : null}

          {loaded ? (
            <div className="row">
              <Badge tone="ok" dot>{t('models.model_loaded')}</Badge>
              <Button size="sm" onClick={onOpenPlayground} icon="chat">{t('models.open_playground')}</Button>
              <Button
                size="sm"
                variant="ghost"
                icon="stop"
                onClick={async () => {
                  await api.unloadModel()
                  await onUnloaded()
                  notify('info', t('models.unload'))
                }}
              >
                {t('models.unload')}
              </Button>
            </div>
          ) : (
            <div className="stack" style={{ gap: 8 }}>
              <Button variant="primary" loading={busy} onClick={() => void load(force)} disabled={Boolean(status && ['ready', 'starting'].includes(String(status.status)))}>
                {busy ? t('models.loading') : t('models.load')}
              </Button>
              {status && ['ready', 'starting'].includes(String(status.status)) ? <Alert tone="warn">{t('models.already_loaded')}</Alert> : null}
              {fit?.status === 'too_large' ? (
                <label className="check">
                  <input type="checkbox" checked={force} onChange={(event) => setForce(event.target.checked)} />
                  {t('models.force_load')}
                </label>
              ) : null}
            </div>
          )}
        </div>
      </Card>
    </>
  )
}
