import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'

import { useApp } from '../context/AppContext'
import { api, ApiError } from '../lib/api'
import { useAsync, useJob } from '../lib/hooks'
import { formatBytes, formatNumber, timeAgo } from '../lib/format'
import type { HFFile, HFModel } from '../lib/types'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardHeader,
  EmptyState,
  Field,
  Input,
  KV,
  Modal,
  Progress,
  Select,
  Spinner,
} from '../components/ui'

const TASKS = [
  'text-generation',
  'image-text-to-text',
  'text2text-generation',
  'feature-extraction',
  'sentence-similarity',
  'automatic-speech-recognition',
]

export function Hub() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { notify, startJob } = useApp()
  const token = useAsync(() => api.hubToken(), [])
  const [query, setQuery] = useState('qwen3')
  const [task, setTask] = useState('')
  const [sort, setSort] = useState('downloads')
  const [results, setResults] = useState<HFModel[]>([])
  const [searching, setSearching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<HFModel | null>(null)
  const [detail, setDetail] = useState<{ files: HFFile[]; suggested: string[]; total_bytes: number } | null>(null)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [chosen, setChosen] = useState<string[]>([])
  const [tokenOpen, setTokenOpen] = useState(false)
  const [tokenInput, setTokenInput] = useState('')
  const [tokenBusy, setTokenBusy] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const job = useJob(jobId)

  const runSearch = async () => {
    setSearching(true)
    setError(null)
    try {
      const data = await api.hubSearch(query, task || undefined, 24, sort)
      setResults(data.results)
      if (data.results.length === 0) setSelected(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSearching(false)
    }
  }

  useEffect(() => {
    void runSearch()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!selected) {
      setDetail(null)
      return
    }
    let cancelled = false
    setLoadingDetail(true)
    api
      .hubModel(selected.id)
      .then((data) => {
        if (cancelled) return
        setDetail({ files: data.files, suggested: data.suggested, total_bytes: data.total_bytes })
        setChosen(data.suggested)
      })
      .catch((err) => {
        if (!cancelled) notify('danger', err instanceof Error ? err.message : String(err))
      })
      .finally(() => {
        if (!cancelled) setLoadingDetail(false)
      })
    return () => {
      cancelled = true
    }
  }, [selected, notify])

  useEffect(() => {
    if (job.job?.status === 'succeeded') {
      notify('ok', `${job.job.output?.indexed ?? 0} model(s) indexed`)
      setJobId(null)
      navigate('/models')
    }
    if (job.job?.status === 'failed') {
      notify('danger', job.job.error ?? t('common.error'))
      setJobId(null)
    }
  }, [job.job?.status, job.job?.error, job.job?.output, notify, navigate, t])

  const selectedSize = useMemo(() => {
    if (!detail) return 0
    return detail.files.filter((file) => chosen.includes(file.path)).reduce((sum, file) => sum + file.size, 0)
  }, [detail, chosen])

  const download = async () => {
    if (!selected || chosen.length === 0) return
    try {
      const id = await startJob(() => api.hubDownload(selected.id, chosen), `Pull ${selected.id}`)
      setJobId(id)
    } catch (err) {
      if (err instanceof ApiError && err.code === 'auth_required') {
        notify('warn', t('hub.token_required'))
        setTokenOpen(true)
      } else {
        notify('danger', err instanceof Error ? err.message : String(err))
      }
    }
  }

  return (
    <>
      <div className="row between">
        <div>
          <p className="muted small">{t('hub.subtitle')}</p>
        </div>
        <div className="row">
          <Badge tone={token.data?.configured ? 'ok' : 'plain'} dot>
            {token.data?.configured ? t('hub.token_active') : t('hub.anonymous')}
          </Badge>
          <Button size="sm" variant="ghost" icon="shield" onClick={() => setTokenOpen(true)}>
            {t('hub.token')}
          </Button>
        </div>
      </div>

      <Card>
        <div className="row" style={{ gap: 10 }}>
          <div className="grow">
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => event.key === 'Enter' && void runSearch()}
              placeholder={t('hub.search_placeholder')}
              aria-label={t('hub.search_placeholder')}
            />
          </div>
          <Select value={task} onChange={(event) => setTask(event.target.value)} style={{ width: 200 }} aria-label={t('hub.task')}>
            <option value="">{t('hub.all_tasks')}</option>
            {TASKS.map((item) => (
              <option key={item} value={item}>{item}</option>
            ))}
          </Select>
          <Select value={sort} onChange={(event) => setSort(event.target.value)} style={{ width: 150 }} aria-label={t('hub.sort')}>
            <option value="downloads">{t('hub.sort_downloads')}</option>
            <option value="likes">{t('hub.sort_likes')}</option>
            <option value="lastModified">{t('hub.sort_recent')}</option>
          </Select>
          <Button variant="primary" icon="search" loading={searching} onClick={() => void runSearch()}>
            {t('common.search')}
          </Button>
        </div>
        {error ? <div style={{ marginTop: 10 }}><Alert tone="danger">{error}</Alert></div> : null}
        {token.data && !token.data.configured ? (
          <div style={{ marginTop: 10 }}>
            <Alert tone="info">{t('hub.anonymous_note')}</Alert>
          </div>
        ) : null}
      </Card>

      <div className="grid sidebar-right">
        <div className="stack">
          {searching ? (
            <Card><Spinner label={t('common.loading')} /></Card>
          ) : results.length === 0 ? (
            <Card><EmptyState title={t('hub.no_results')} icon="database">{t('hub.no_results_hint')}</EmptyState></Card>
          ) : (
            <div className="hub-grid">
              {results.map((model) => (
                <button
                  key={model.id}
                  type="button"
                  className={`hub-card ${selected?.id === model.id ? 'active' : ''}`}
                  onClick={() => setSelected(model)}
                >
                  <div className="row between" style={{ gap: 8 }}>
                    <strong className="truncate" style={{ fontSize: 13 }}>{model.id}</strong>
                    {model.gated ? <Badge tone="warn">{t('hub.gated')}</Badge> : null}
                  </div>
                  <div className="row" style={{ gap: 10, marginTop: 6 }}>
                    <span className="tiny dim">↓ {formatNumber(model.downloads)}</span>
                    <span className="tiny dim">♥ {formatNumber(model.likes)}</span>
                    {model.pipeline_tag ? <Badge tone="plain">{model.pipeline_tag}</Badge> : null}
                  </div>
                  <div className="pill-row" style={{ marginTop: 8 }}>
                    {model.tags.filter((tag) => ['gguf', 'safetensors', 'transformers', 'text-generation'].includes(tag)).slice(0, 3).map((tag) => (
                      <Badge key={tag} tone="plain">{tag}</Badge>
                    ))}
                    {model.last_modified ? <span className="tiny dim">{timeAgo(model.last_modified)}</span> : null}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="stack">
          {selected ? (
            <>
              <Card>
                <CardHeader title={selected.id} subtitle={selected.author ?? ''} icon="box" actions={
                  selected.gated ? <Badge tone="warn">{t('hub.gated')}</Badge> : null
                } />
                <KV
                  items={[
                    [t('hub.downloads'), formatNumber(selected.downloads)],
                    [t('hub.likes'), formatNumber(selected.likes)],
                    [t('hub.updated'), selected.last_modified ? timeAgo(selected.last_modified) : '—'],
                    [t('hub.library'), selected.library_name ?? '—'],
                  ]}
                />
              </Card>

              <Card>
                <CardHeader
                  title={t('hub.files')}
                  icon="layers"
                  subtitle={detail ? `${detail.files.length} ${t('hub.files').toLowerCase()} · ${formatBytes(detail.total_bytes)}` : undefined}
                />
                {loadingDetail ? (
                  <Spinner />
                ) : detail ? (
                  <div className="stack">
                    <div className="hub-files">
                      {detail.files.slice(0, 40).map((file) => {
                        const checked = chosen.includes(file.path)
                        return (
                          <label key={file.path} className={`hub-file ${checked ? 'checked' : ''}`}>
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={(event) =>
                                setChosen((prev) =>
                                  event.target.checked ? [...prev, file.path] : prev.filter((name) => name !== file.path),
                                )
                              }
                            />
                            <span className="mono tiny grow truncate">{file.path}</span>
                            <span className="tiny dim nowrap">{formatBytes(file.size)}</span>
                            {file.sha256 ? <Badge tone="ok">{t('hub.verified_hash')}</Badge> : null}
                          </label>
                        )
                      })}
                    </div>
                    <div className="row between">
                      <span className="tiny dim">{chosen.length} {t('hub.selected')} · {formatBytes(selectedSize)}</span>
                      <Button
                        variant="primary"
                        icon="download"
                        disabled={chosen.length === 0 || Boolean(jobId)}
                        loading={Boolean(jobId)}
                        onClick={() => void download()}
                      >
                        {t('hub.pull')}
                      </Button>
                    </div>
                    {job.job ? (
                      <div className="stack" style={{ gap: 6 }}>
                        <Progress value={job.job.progress} indeterminate={job.job.progress === 0} />
                        <span className="tiny dim truncate">{job.job.message}</span>
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <EmptyState title={t('hub.select_model')} icon="layers" />
                )}
              </Card>
            </>
          ) : (
            <Card>
              <EmptyState title={t('hub.select_model')} icon="database">{t('hub.select_model_hint')}</EmptyState>
            </Card>
          )}
        </div>
      </div>

      <Modal open={tokenOpen} onClose={() => setTokenOpen(false)} title={t('hub.token')}>
        <div className="stack">
          <Alert tone="info">{t('hub.token_note')}</Alert>
          <KV
            items={[
              [t('common.status'), token.data?.configured ? <Badge tone="ok">{t('hub.token_active')}</Badge> : <Badge tone="plain">{t('hub.anonymous')}</Badge>],
              [t('hub.token_masked'), token.data?.masked ?? '—'],
            ]}
          />
          <Field label={t('hub.token_input')} hint={t('hub.token_input_hint')} htmlFor="hf-token">
            <Input
              id="hf-token"
              type="password"
              value={tokenInput}
              onChange={(event) => setTokenInput(event.target.value)}
              placeholder="hf_…"
              autoComplete="off"
            />
          </Field>
          <div className="row">
            <Button
              variant="primary"
              loading={tokenBusy}
              disabled={tokenInput.trim().length < 4}
              onClick={async () => {
                setTokenBusy(true)
                try {
                  const data = await api.setHubToken(tokenInput.trim())
                  setTokenInput('')
                  token.reload()
                  notify('ok', `${t('hub.token_saved')} · ${String(data.identity?.name ?? '')}`)
                } catch (err) {
                  notify('danger', err instanceof Error ? err.message : String(err))
                } finally {
                  setTokenBusy(false)
                }
              }}
            >
              {t('common.save')}
            </Button>
            <Button
              variant="ghost"
              onClick={async () => {
                try {
                  const data = await api.testHubToken()
                  notify('ok', data.mode === 'anonymous' ? t('hub.anonymous') : `${t('hub.token_active')} · ${String((data.identity as Record<string, unknown> | undefined)?.name ?? '')}`)
                } catch (err) {
                  notify('danger', err instanceof Error ? err.message : String(err))
                }
              }}
            >
              {t('hub.test')}
            </Button>
            <Button
              variant="danger"
              disabled={!token.data?.configured}
              onClick={async () => {
                try {
                  await api.clearHubToken()
                  token.reload()
                  notify('ok', t('hub.token_cleared'))
                } catch (err) {
                  notify('danger', err instanceof Error ? err.message : String(err))
                }
              }}
            >
              {t('common.remove')}
            </Button>
          </div>
          <span className="tiny dim">{t('hub.token_storage')}</span>
        </div>
      </Modal>
    </>
  )
}
