import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { useApp } from '../context/AppContext'
import { api, ApiError } from '../lib/api'
import { useAsync, useJob } from '../lib/hooks'
import { formatBytes, formatDate, formatPercent, statusTone } from '../lib/format'
import type { Dataset, DatasetRecord } from '../lib/types'
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
  Tabs,
  Textarea,
} from '../components/ui'

type Tab = 'datasets' | 'import'

export function DataStudio() {
  const { t } = useTranslation()
  const { notify, startJob } = useApp()
  const [tab, setTab] = useState<Tab>('datasets')
  const datasets = useAsync(() => api.datasets(), [])
  const settings = useAsync(() => api.settings(), [])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [reviewOpen, setReviewOpen] = useState(false)
  const [importJobId, setImportJobId] = useState<string | null>(null)
  const importJob = useJob(importJobId)

  useEffect(() => {
    if (!selectedId && datasets.data?.datasets.length) {
      setSelectedId(datasets.data.datasets[0].id)
    }
  }, [datasets.data, selectedId])

  useEffect(() => {
    if (importJob.job?.status === 'succeeded') {
      const id = importJob.job.output?.dataset_id as string | undefined
      datasets.reload()
      if (id) setSelectedId(id)
      setTab('datasets')
      notify('ok', `${importJob.job.output?.record_count ?? 0} ${t('data.record_count').toLowerCase()}`)
      setImportJobId(null)
    }
    if (importJob.job?.status === 'failed') {
      notify('danger', importJob.job.error ?? t('common.error'))
      setImportJobId(null)
    }
  }, [importJob.job?.status, importJob.job?.error, importJob.job?.output, datasets.reload, notify, t])

  const selected = datasets.data?.datasets.find((dataset) => dataset.id === selectedId) ?? null
  const importRoots = ((settings.data?.effective as Record<string, unknown> | undefined)?.import_roots as string[] | undefined) ?? []

  return (
    <>
      <div className="row between">
        <div>
          <p className="muted small">{t('data.subtitle')}</p>
        </div>
        <Button variant="primary" icon="upload" onClick={() => setTab('import')}>
          {t('data.import')}
        </Button>
      </div>

      <Tabs<Tab>
        tabs={[
          { id: 'datasets', label: `${t('data.datasets')} (${datasets.data?.datasets.length ?? 0})` },
          { id: 'import', label: t('data.import') },
        ]}
        value={tab}
        onChange={setTab}
        label={t('data.title')}
      />

      {tab === 'import' ? (
        <ImportPanel
          importRoots={importRoots}
          busy={importJob.job ? ['queued', 'running'].includes(importJob.job.status) : false}
          progress={importJob.job?.progress ?? 0}
          onStart={async (payload) => {
            try {
              const id = await startJob(() => api.importDataset(payload), t('data.importing'))
              setImportJobId(id)
            } catch (error) {
              notify('danger', error instanceof Error ? error.message : String(error))
            }
          }}
        />
      ) : datasets.loading ? (
        <Card><Spinner label={t('common.loading')} /></Card>
      ) : !datasets.data || datasets.data.datasets.length === 0 ? (
        <Card>
          <EmptyState
            title={t('data.empty')}
            icon="database"
            action={<Button variant="primary" icon="upload" onClick={() => setTab('import')}>{t('data.import')}</Button>}
          >
            {t('data.empty_hint')}
          </EmptyState>
        </Card>
      ) : (
        <div className="grid sidebar-right">
          <Card flush>
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>{t('common.name')}</th>
                    <th>{t('common.status')}</th>
                    <th className="num">{t('data.record_count')}</th>
                    <th className="num">{t('data.synthetic_ratio')}</th>
                    <th>{t('common.created')}</th>
                  </tr>
                </thead>
                <tbody>
                  {datasets.data.datasets.map((dataset) => (
                    <tr
                      key={dataset.id}
                      onClick={() => setSelectedId(dataset.id)}
                      style={{ cursor: 'pointer', background: selectedId === dataset.id ? 'var(--surface-2)' : undefined }}
                    >
                      <td>
                        <div style={{ fontWeight: 560 }}>{dataset.name}</div>
                        <div className="tiny dim">{dataset.provider} · {dataset.id}</div>
                      </td>
                      <td>
                        <div className="row" style={{ gap: 4 }}>
                          <Badge tone={statusTone(dataset.status)}>{dataset.status}</Badge>
                          <Badge tone={dataset.rights_status === 'confirmed' ? 'ok' : 'warn'}>{dataset.rights_status}</Badge>
                        </div>
                      </td>
                      <td className="num">{dataset.record_count}</td>
                      <td className="num">{formatPercent(dataset.stats?.synthetic_ratio ?? 0)}</td>
                      <td className="dim tiny">{formatDate(dataset.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {selected ? (
            <DatasetDetail
              dataset={selected}
              onChanged={() => datasets.reload()}
              onReview={() => setReviewOpen(true)}
              notify={notify}
              startJob={startJob}
            />
          ) : null}
        </div>
      )}

      {selected && reviewOpen ? (
        <ReviewModal dataset={selected} onClose={() => setReviewOpen(false)} onChanged={() => datasets.reload()} />
      ) : null}
    </>
  )
}

function ImportPanel({
  importRoots,
  busy,
  progress,
  onStart,
}: {
  importRoots: string[]
  busy: boolean
  progress: number
  onStart: (payload: Record<string, unknown>) => Promise<void>
}) {
  const { t } = useTranslation()
  const { notify } = useApp()
  const [paths, setPaths] = useState('')
  const [provider, setProvider] = useState('auto')
  const [name, setName] = useState('')
  const [rights, setRights] = useState('needs_review')
  const [consent, setConsent] = useState(false)
  const [preview, setPreview] = useState<Record<string, unknown> | null>(null)
  const [previewing, setPreviewing] = useState(false)

  const pathList = useMemo(() => paths.split('\n').map((value) => value.trim()).filter(Boolean), [paths])

  const runPreview = async () => {
    if (pathList.length === 0) return
    setPreviewing(true)
    try {
      const data = await api.previewDataset(pathList, provider)
      setPreview(data as unknown as Record<string, unknown>)
    } catch (error) {
      notify('danger', error instanceof Error ? error.message : String(error))
    } finally {
      setPreviewing(false)
    }
  }

  const inventory = preview?.inventory as { file_count?: number; total_bytes?: number; files?: Array<Record<string, unknown>>; problems?: Array<Record<string, unknown>> } | undefined
  const samples = (preview?.samples as Array<Record<string, unknown>> | undefined) ?? []
  const errors = (preview?.errors as string[] | undefined) ?? []

  return (
    <div className="grid sidebar-right">
      <div className="stack">
        <Card>
          <CardHeader title={t('data.import_title')} icon="upload" />
          <div className="stack">
            <Field
              label={t('data.paths')}
              hint={`${t('data.paths_hint')}${importRoots.length ? ` — ${importRoots.join(' · ')}` : ''}`}
              htmlFor="import-paths"
            >
              <Textarea id="import-paths" value={paths} onChange={(event) => setPaths(event.target.value)} placeholder={importRoots[0] ?? 'C:\\Users\\me\\.codex\\sessions'} />
            </Field>
            <div className="grid cols-2">
              <Field label={t('data.provider')} htmlFor="provider">
                <Select id="provider" value={provider} onChange={(event) => setProvider(event.target.value)}>
                  <option value="auto">{t('data.provider_auto')}</option>
                  <option value="generic">{t('data.provider_generic')}</option>
                  <option value="claude_code">{t('data.provider_claude')}</option>
                  <option value="codex">{t('data.provider_codex')}</option>
                  <option value="opencode">{t('data.provider_opencode')}</option>
                  <option value="antigravity">{t('data.provider_antigravity')}</option>
                </Select>
              </Field>
              <Field label={t('common.name')} htmlFor="dataset-name">
                <Input id="dataset-name" value={name} onChange={(event) => setName(event.target.value)} placeholder="Session import" />
              </Field>
            </div>
            <div className="row">
              <Button icon="search" loading={previewing} disabled={pathList.length === 0} onClick={() => void runPreview()}>
                {previewing ? t('data.previewing') : t('data.preview')}
              </Button>
              <span className="tiny dim">{pathList.length} path(s)</span>
            </div>
          </div>
        </Card>

        {preview ? (
          <Card>
            <CardHeader title={t('data.inventory')} icon="database" subtitle={`${inventory?.file_count ?? 0} ${t('data.files').toLowerCase()} · ${formatBytes(inventory?.total_bytes ?? 0)}`} />
            {inventory?.files?.length ? (
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>{t('common.name')}</th>
                      <th>{t('data.provider')}</th>
                      <th className="num">{t('common.size')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {inventory.files.slice(0, 12).map((file, index) => (
                      <tr key={index}>
                        <td className="mono tiny truncate" style={{ maxWidth: 320 }}>{String(file.path ?? '')}</td>
                        <td><Badge tone="plain">{String(file.provider_guess ?? '')}</Badge></td>
                        <td className="num">{formatBytes(Number(file.size_bytes ?? 0))}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <Alert tone="warn">{t('data.no_preview')}</Alert>
            )}
            {inventory?.problems?.length ? (
              <div className="stack" style={{ marginTop: 10, gap: 6 }}>
                {inventory.problems.slice(0, 5).map((problem, index) => (
                  <Alert key={index} tone="warn">{String(problem.error ?? problem.path ?? '')}</Alert>
                ))}
              </div>
            ) : null}
            {preview ? (
              <div className="stack" style={{ marginTop: 10, gap: 6 }}>
                <span className="tiny dim">{t('data.privacy_note')}</span>
              </div>
            ) : null}
            {errors.length > 0 ? (
              <div className="stack" style={{ marginTop: 10, gap: 6 }}>
                {errors.slice(0, 5).map((error, index) => (
                  <span key={index} className="tiny dim">• {error}</span>
                ))}
              </div>
            ) : null}
          </Card>
        ) : null}

        {samples.length > 0 ? (
          <Card>
            <CardHeader title={t('data.samples')} icon="chat" />
            <div className="stack">
              {samples.map((sample, index) => (
                <div key={index} className="card tight" style={{ background: 'var(--surface-2)' }}>
                  <div className="row" style={{ gap: 6, marginBottom: 6 }}>
                    <Badge tone="accent">{String(sample.task_type ?? '')}</Badge>
                    <Badge tone="plain">{String(sample.provider ?? '')}</Badge>
                    {sample.has_tool_calls ? <Badge tone="info">tools</Badge> : null}
                  </div>
                  <div className="small" style={{ whiteSpace: 'pre-wrap' }}>{String(sample.user_request ?? '').slice(0, 400)}</div>
                </div>
              ))}
            </div>
          </Card>
        ) : null}
      </div>

      <div className="stack">
        <Card accent>
          <CardHeader title={t('data.consent_title')} icon="shield" />
          <div className="stack">
            <Field label={t('data.rights')} hint={t('data.rights_hint')} htmlFor="rights">
              <Select id="rights" value={rights} onChange={(event) => setRights(event.target.value)}>
                <option value="needs_review">{t('data.rights_needs_review')}</option>
                <option value="confirmed">{t('data.rights_confirmed')}</option>
                <option value="restricted">{t('data.rights_restricted')}</option>
                <option value="excluded">{t('data.rights_excluded')}</option>
              </Select>
            </Field>
            <p className="tiny muted">{t('data.consent_text')}</p>
            <label className="check">
              <input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} />
              {t('data.consent_check')}
            </label>
            {busy ? <Progress value={progress} indeterminate={progress === 0} /> : null}
            <Button
              variant="primary"
              icon="upload"
              block
              disabled={!consent || pathList.length === 0 || busy}
              loading={busy}
              onClick={() =>
                void onStart({
                  name: name.trim() || 'Session import',
                  paths: pathList,
                  provider,
                  rights_status: rights,
                  redact: true,
                  dedupe: true,
                })
              }
            >
              {busy ? t('data.importing') : t('data.start_import')}
            </Button>
            <span className="tiny dim">{t('data.privacy_note')}</span>
          </div>
        </Card>
      </div>
    </div>
  )
}

function DatasetDetail({
  dataset,
  onChanged,
  onReview,
  notify,
  startJob,
}: {
  dataset: Dataset
  onChanged: () => void
  onReview: () => void
  notify: (kind: 'ok' | 'danger' | 'warn' | 'info', message: string) => void
  startJob: (request: () => Promise<{ job_id: string }>, label: string) => Promise<string>
}) {
  const { t } = useTranslation()
  const [busy, setBusy] = useState(false)
  const [enrichJob, setEnrichJob] = useState<string | null>(null)
  const job = useJob(enrichJob)
  const annotations = useAsync(() => api.annotations(dataset.id), [dataset.id])

  useEffect(() => {
    if (job.job?.status === 'succeeded') {
      annotations.reload()
      onChanged()
      notify('ok', `${job.job.output?.created ?? 0} annotations`)
      setEnrichJob(null)
    }
    if (job.job?.status === 'failed') {
      notify('danger', job.job.error ?? t('common.error'))
      setEnrichJob(null)
    }
  }, [job.job?.status, job.job?.error, job.job?.output, annotations.reload, notify, t])

  const splits = dataset.stats?.by_split ?? {}
  const redactions = dataset.stats?.redactions ?? {}
  const totalRedactions = Object.values(redactions).reduce((sum, value) => sum + Number(value), 0)

  return (
    <>
      <Card>
        <CardHeader title={dataset.name} subtitle={`${dataset.provider} · ${dataset.id}`} icon="database" actions={<Badge tone={statusTone(dataset.status)}>{dataset.status}</Badge>} />
        <KV
          items={[
            [t('data.record_count'), dataset.record_count],
            [t('data.synthetic_ratio'), formatPercent(dataset.stats?.synthetic_ratio ?? 0)],
            [t('data.duplicates'), `${dataset.stats?.duplicates_removed ?? 0} / ${dataset.stats?.near_duplicates_removed ?? 0}`],
            [t('data.redactions'), totalRedactions],
            [t('common.created'), formatDate(dataset.created_at)],
            [t('data.holdout'), dataset.holdout_hash ? <span className="mono tiny">{dataset.holdout_hash.slice(0, 20)}…</span> : t('data.no_holdout')],
          ]}
        />
        <div className="pill-row" style={{ marginTop: 10 }}>
          {Object.entries(splits).map(([split, count]) => (
            <Badge key={split} tone={split === 'test' ? 'danger' : split === 'validation' ? 'warn' : 'ok'}>
              {t(`data.split_${split}`)}: {count}
            </Badge>
          ))}
        </div>
      </Card>

      <Card>
        <CardHeader title={t('data.redaction_summary')} icon="shield" />
        {Object.keys(redactions).length === 0 ? (
          <span className="small dim">{t('common.none')}</span>
        ) : (
          <div className="pill-row">
            {Object.entries(redactions).map(([type, count]) => (
              <Badge key={type} tone={type === 'high_entropy' ? 'warn' : 'info'}>{type}: {count}</Badge>
            ))}
          </div>
        )}
      </Card>

      <Card>
        <CardHeader title={t('data.annotations')} icon="sparkles" subtitle={annotations.data ? `${annotations.data.grounded} ${t('data.grounded').toLowerCase()} / ${annotations.data.ungrounded} ${t('data.ungrounded').toLowerCase()}` : undefined} />
        <div className="stack" style={{ gap: 8 }}>
          <Button size="sm" icon="sparkles" disabled={dataset.status === 'frozen'} loading={Boolean(enrichJob)} onClick={async () => {
            try {
              const id = await startJob(() => api.enrichDataset(dataset.id, 'deterministic', 100), t('data.enrich'))
              setEnrichJob(id)
            } catch (error) {
              notify('danger', error instanceof Error ? error.message : String(error))
            }
          }}>
            {t('data.enrich_deterministic')}
          </Button>
          <Button size="sm" variant="ghost" icon="bot" disabled={dataset.status === 'frozen'} onClick={async () => {
            try {
              const id = await startJob(() => api.enrichDataset(dataset.id, 'model', 50), t('data.enrich'))
              setEnrichJob(id)
            } catch (error) {
              if (error instanceof ApiError && error.code === 'no_model') notify('warn', t('play.no_model'))
              else notify('danger', error instanceof Error ? error.message : String(error))
            }
          }}>
            {t('data.enrich_model')}
          </Button>
          {job.job ? <Progress value={job.job.progress} indeterminate={job.job.progress === 0} /> : null}
        </div>
      </Card>

      <Card>
        <CardHeader title={t('data.rights')} icon="check" />
        <Select value={dataset.rights_status} onChange={async (event) => {
          try {
            await api.setRights(dataset.id, event.target.value)
            onChanged()
            notify('ok', event.target.value)
          } catch (error) {
            notify('danger', error instanceof Error ? error.message : String(error))
          }
        }}>
          <option value="needs_review">{t('data.rights_needs_review')}</option>
          <option value="confirmed">{t('data.rights_confirmed')}</option>
          <option value="restricted">{t('data.rights_restricted')}</option>
          <option value="excluded">{t('data.rights_excluded')}</option>
        </Select>
        {dataset.rights_status !== 'confirmed' ? <div style={{ marginTop: 8 }}><Alert tone="warn">{t('data.rights_required')}</Alert></div> : null}
      </Card>

      <Card>
        <div className="stack">
          <Button icon="database" onClick={onReview}>{t('data.review')}</Button>
          <Button
            icon="check"
            disabled={dataset.status === 'frozen' || dataset.rights_status !== 'confirmed' || busy}
            loading={busy}
            onClick={async () => {
              setBusy(true)
              try {
                await api.freezeDataset(dataset.id)
                onChanged()
                notify('ok', t('data.frozen'))
              } catch (error) {
                notify('danger', error instanceof Error ? error.message : String(error))
              } finally {
                setBusy(false)
              }
            }}
          >
            {dataset.status === 'frozen' ? t('data.frozen') : t('data.freeze')}
          </Button>
          <Button
            icon="download"
            disabled={dataset.rights_status !== 'confirmed'}
            onClick={async () => {
              try {
                const { blob, filename } = await api.exportDatasetUrl(dataset.id)
                const url = URL.createObjectURL(blob)
                const anchor = document.createElement('a')
                anchor.href = url
                anchor.download = filename ?? `${dataset.id}.jsonl`
                anchor.click()
                URL.revokeObjectURL(url)
              } catch (error) {
                notify('danger', error instanceof Error ? error.message : String(error))
              }
            }}
          >
            {t('data.export')}
          </Button>
          <Button variant="danger" icon="trash" onClick={async () => {
            if (!window.confirm(t('data.delete_confirm'))) return
            try {
              await api.deleteDataset(dataset.id)
              onChanged()
              notify('ok', t('common.delete'))
            } catch (error) {
              notify('danger', error instanceof Error ? error.message : String(error))
            }
          }}>
            {t('data.delete')}
          </Button>
        </div>
      </Card>

      {dataset.manifest && Object.keys(dataset.manifest).length > 0 ? (
        <Card>
          <CardHeader title={t('data.manifest')} icon="layers" />
          {dataset.manifest.warning ? (
            <div style={{ marginBottom: 10 }}>
              <Alert tone="warn">{String(dataset.manifest.warning_detail ?? dataset.manifest.warning)}</Alert>
            </div>
          ) : null}
          <div className="mono-block">{JSON.stringify(dataset.manifest, null, 2)}</div>
        </Card>
      ) : null}
    </>
  )
}

function ReviewModal({ dataset, onClose, onChanged }: { dataset: Dataset; onClose: () => void; onChanged: () => void }) {
  const { t } = useTranslation()
  const [offset, setOffset] = useState(0)
  const [split, setSplit] = useState('')
  const [synthetic, setSynthetic] = useState('')
  const [includeExcluded, setIncludeExcluded] = useState(true)
  const limit = 10
  const records = useAsync(
    () =>
      api.records(dataset.id, {
        offset,
        limit,
        split: split || undefined,
        synthetic: synthetic === '' ? undefined : synthetic === 'true',
        include_excluded: includeExcluded,
      }),
    [dataset.id, offset, split, synthetic, includeExcluded],
  )

  return (
    <Modal open onClose={onClose} title={`${t('data.review_title')} — ${dataset.name}`} wide>
      <div className="row" style={{ gap: 8, marginBottom: 12 }}>
        <Select value={split} onChange={(event) => { setSplit(event.target.value); setOffset(0) }} style={{ width: 160 }} aria-label={t('data.filters')}>
          <option value="">{t('data.all_splits')}</option>
          <option value="train">{t('data.split_train')}</option>
          <option value="validation">{t('data.split_validation')}</option>
          <option value="test">{t('data.split_test')}</option>
        </Select>
        <Select value={synthetic} onChange={(event) => { setSynthetic(event.target.value); setOffset(0) }} style={{ width: 170 }}>
          <option value="">{t('common.filter')}: {t('common.none')}</option>
          <option value="true">{t('data.synthetic')}</option>
          <option value="false">{t('data.original')}</option>
        </Select>
        <label className="check" style={{ marginLeft: 'auto' }}>
          <input type="checkbox" checked={includeExcluded} onChange={(event) => setIncludeExcluded(event.target.checked)} />
          {t('data.excluded_records')}
        </label>
      </div>

      {records.loading ? (
        <Spinner label={t('common.loading')} />
      ) : !records.data || records.data.items.length === 0 ? (
        <EmptyState title={t('common.none')} icon="database" />
      ) : (
        <div className="stack">
          {records.data.items.map((record) => (
            <RecordCard key={record.id} datasetId={dataset.id} record={record} onChanged={() => { records.reload(); onChanged() }} />
          ))}
        </div>
      )}

      <div className="row between" style={{ marginTop: 14 }}>
        <span className="tiny dim">
          {offset + 1}–{Math.min(offset + limit, records.data?.total ?? 0)} {t('common.of')} {records.data?.total ?? 0}
        </span>
        <div className="row">
          <Button size="sm" variant="ghost" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>{t('data.prev_page')}</Button>
          <Button size="sm" variant="ghost" disabled={offset + limit >= (records.data?.total ?? 0)} onClick={() => setOffset(offset + limit)}>{t('data.next_page')}</Button>
        </div>
      </div>
    </Modal>
  )
}

function RecordCard({ datasetId, record, onChanged }: { datasetId: string; record: DatasetRecord; onChanged: () => void }) {
  const { t } = useTranslation()
  const { notify } = useApp()
  const [open, setOpen] = useState(false)
  const payload = record.payload
  const teaching = (payload.teaching ?? {}) as Record<string, unknown>

  return (
    <div className="card tight" style={{ background: record.excluded ? 'rgba(255,107,107,0.05)' : 'var(--surface-2)', opacity: record.excluded ? 0.7 : 1 }}>
      <div className="row" style={{ gap: 6, marginBottom: 6 }}>
        <Badge tone={record.split === 'test' ? 'danger' : record.split === 'validation' ? 'warn' : 'ok'}>{t(`data.split_${record.split}`)}</Badge>
        {record.synthetic ? <Badge tone="accent">{t('data.synthetic')}</Badge> : null}
        {record.excluded ? <Badge tone="danger">{t('data.excluded')}</Badge> : null}
        <span className="tiny dim mono">{record.record_id.slice(0, 16)}</span>
        <div style={{ marginLeft: 'auto' }} className="row">
          <Button
            size="sm"
            variant="ghost"
            onClick={async () => {
              try {
                await api.excludeRecord(datasetId, record.record_id, !record.excluded, record.excluded ? undefined : 'excluded by reviewer')
                onChanged()
              } catch (error) {
                notify('danger', error instanceof Error ? error.message : String(error))
              }
            }}
          >
            {record.excluded ? t('data.include') : t('data.exclude')}
          </Button>
          <Button size="sm" variant="ghost" icon="chevron" onClick={() => setOpen((value) => !value)}>
            {t('common.details')}
          </Button>
        </div>
      </div>
      <div className="small" style={{ whiteSpace: 'pre-wrap' }}>
        {(payload.task?.user_request ?? '').slice(0, 600)}
      </div>
      {open ? (
        <div className="stack" style={{ marginTop: 10 }}>
          <div>
            <div className="label" style={{ marginBottom: 4 }}>{t('data.trace')}</div>
            <div className="mono-block">
              {(payload.trace ?? []).map((step, index) => `${index + 1}. [${step.kind}] ${step.content.slice(0, 400)}`).join('\n\n')}
            </div>
          </div>
          {teaching.decision_summary ? (
            <div>
              <div className="label" style={{ marginBottom: 4 }}>{t('data.decision_summary')}</div>
              <div className="small muted" style={{ whiteSpace: 'pre-wrap' }}>{String(teaching.decision_summary)}</div>
            </div>
          ) : null}
          {teaching.negative_example ? (
            <div>
              <div className="label" style={{ marginBottom: 4 }}>{t('data.negative_example')}</div>
              <div className="small" style={{ color: 'var(--warn)' }}>{String(teaching.negative_example)}</div>
            </div>
          ) : null}
          {!teaching.decision_summary && record.synthetic ? <Alert tone="warn">{t('data.unknown_evidence')}</Alert> : null}
        </div>
      ) : null}
    </div>
  )
}
