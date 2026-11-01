import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { useApp } from '../context/AppContext'
import { api } from '../lib/api'
import { useAsync, useJob } from '../lib/hooks'
import { formatDate, formatPercent, statusTone } from '../lib/format'
import type { Candidate } from '../lib/types'
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
  Textarea,
} from '../components/ui'

export function UpgradeLab() {
  const { t } = useTranslation()
  const { notify, startJob } = useApp()
  const runs = useAsync(() => api.benchmarkRuns(), [])
  const datasets = useAsync(() => api.datasets(), [])
  const candidates = useAsync(() => api.candidates(), [])
  const taxonomy = useAsync(() => api.upgradeTaxonomy(), [])
  const lineage = useAsync(() => api.lineage(), [])
  const [runId, setRunId] = useState('')
  const [analysis, setAnalysis] = useState<Record<string, unknown> | null>(null)
  const [analysing, setAnalysing] = useState(false)
  const [dataset, setDataset] = useState('')
  const [kind, setKind] = useState('prompt')
  const [note, setNote] = useState('')
  const [creating, setCreating] = useState(false)
  const [evalJobId, setEvalJobId] = useState<string | null>(null)
  const job = useJob(evalJobId)
  const [promoteTarget, setPromoteTarget] = useState<Candidate | null>(null)
  const [approver, setApprover] = useState('')
  const [threshold, setThreshold] = useState(0.05)
  const [active, setActive] = useState<Record<string, unknown> | null>(null)

  useEffect(() => {
    if (job.job?.status === 'succeeded') {
      candidates.reload()
      notify('ok', t('up.status.evaluated'))
      setEvalJobId(null)
    }
    if (job.job?.status === 'failed') {
      notify('danger', job.job.error ?? t('common.error'))
      setEvalJobId(null)
    }
  }, [job.job?.status, job.job?.error, candidates.reload, notify, t])

  useEffect(() => {
    setActive((taxonomy.data?.active as Record<string, unknown> | null) ?? null)
  }, [taxonomy.data])

  const frozen = (datasets.data?.datasets ?? []).filter((item) => item.status === 'frozen' && item.rights_status === 'confirmed')

  return (
    <>
      <div>
        <p className="muted small">{t('up.subtitle')}</p>
      </div>

      <div className="grid sidebar-right">
        <div className="stack">
          <Card>
            <CardHeader title={t('up.analyze')} icon="search" />
            <div className="row">
              <div className="grow">
                <Select value={runId} onChange={(event) => { setRunId(event.target.value); setAnalysis(null) }} aria-label={t('up.select_run')}>
                  <option value="">{t('up.select_run')}</option>
                  {(runs.data?.runs ?? []).map((run) => (
                    <option key={run.id} value={run.id}>
                      {run.suite_id} · {formatPercent(Number((run.results?.aggregate as Record<string, unknown> | undefined)?.score ?? 0))} · {run.id.slice(-8)}
                    </option>
                  ))}
                </Select>
              </div>
              <Button
                icon="chart"
                loading={analysing}
                disabled={!runId}
                onClick={async () => {
                  setAnalysing(true)
                  try {
                    setAnalysis(await api.analyzeRun(runId))
                  } catch (error) {
                    notify('danger', error instanceof Error ? error.message : String(error))
                  } finally {
                    setAnalysing(false)
                  }
                }}
              >
                {t('up.analyze')}
              </Button>
            </div>

            {analysis ? (
              <div className="stack" style={{ marginTop: 12 }}>
                <div className="row between">
                  <span className="small">{t('up.failures')}</span>
                  <Badge tone={Number(analysis.failure_count) > 0 ? 'warn' : 'ok'}>{String(analysis.failure_count)}</Badge>
                </div>
                {(analysis.failures as Array<Record<string, unknown>> | undefined)?.length ? (
                  <div className="table-wrap">
                    <table className="table">
                      <thead>
                        <tr>
                          <th>{t('bench.case')}</th>
                          <th>{t('up.taxonomy')}</th>
                          <th>{t('common.evidence')}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(analysis.failures as Array<Record<string, unknown>>).map((failure, index) => (
                          <tr key={index}>
                            <td className="mono tiny">{String(failure.case_id)}</td>
                            <td><Badge tone={failure.critical ? 'danger' : 'warn'}>{String(failure.label)}</Badge></td>
                            <td className="tiny">{String(failure.evidence ?? '')}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <Alert tone="ok">{t('up.no_failures')}</Alert>
                )}
              </div>
            ) : null}
          </Card>

          <Card>
            <CardHeader title={t('up.create_title')} icon="sparkles" subtitle={t('up.requires_frozen')} />
            <div className="stack">
              <div className="grid cols-2">
                <Field label={t('up.dataset')} htmlFor="cand-dataset">
                  <Select id="cand-dataset" value={dataset} onChange={(event) => setDataset(event.target.value)}>
                    <option value="">—</option>
                    {frozen.map((item) => (
                      <option key={item.id} value={item.id}>{item.name}</option>
                    ))}
                  </Select>
                </Field>
                <Field label={t('up.kind')} htmlFor="cand-kind">
                  <Select id="cand-kind" value={kind} onChange={(event) => setKind(event.target.value)}>
                    <option value="prompt">{t('up.kind_prompt')}</option>
                    <option value="runtime">{t('up.kind_runtime')}</option>
                    <option value="dataset">{t('up.kind_dataset')}</option>
                  </Select>
                </Field>
              </div>
              <Field label={t('up.note')} htmlFor="cand-note">
                <Textarea id="cand-note" value={note} onChange={(event) => setNote(event.target.value)} style={{ minHeight: 60 }} />
              </Field>
              <Button
                variant="primary"
                icon="sparkles"
                loading={creating}
                disabled={!runId || !dataset}
                onClick={async () => {
                  setCreating(true)
                  try {
                    await api.createCandidate({ parent_run_id: runId, dataset_id: dataset, kind, note: note || undefined })
                    candidates.reload()
                    notify('ok', t('up.status.proposed'))
                  } catch (error) {
                    notify('danger', error instanceof Error ? error.message : String(error))
                  } finally {
                    setCreating(false)
                  }
                }}
              >
                {t('up.create')}
              </Button>
              <Alert tone="info">{t('up.review_note')}</Alert>
            </div>
          </Card>

          <Card>
            <CardHeader title={t('up.candidates')} icon="layers" />
            {candidates.loading ? (
              <Spinner />
            ) : !candidates.data || candidates.data.candidates.length === 0 ? (
              <EmptyState title={t('up.empty')} icon="sparkles" />
            ) : (
              <div className="stack">
                {candidates.data.candidates.map((candidate) => (
                  <div key={candidate.id} className="card tight" style={{ background: 'var(--surface-2)' }}>
                    <div className="row between" style={{ marginBottom: 8 }}>
                      <div>
                        <div className="small" style={{ fontWeight: 560 }}>{String(candidate.config.kind ?? 'prompt')} · {candidate.id}</div>
                        <div className="tiny dim">← {candidate.parent_run_id} · {formatDate(candidate.created_at)}</div>
                      </div>
                      <Badge tone={statusTone(candidate.status)}>{t(`up.status.${candidate.status === 'rolled_back' ? 'rolled_back' : candidate.status}`)}</Badge>
                    </div>
                    {candidate.config.patches?.length ? (
                      <div className="pill-row" style={{ marginBottom: 8 }}>
                        {candidate.config.source_labels?.map((label) => <Badge key={label} tone="plain">{label}</Badge>)}
                      </div>
                    ) : null}
                    {candidate.metrics.delta !== undefined ? (
                      <div className="row" style={{ gap: 16, marginBottom: 8 }}>
                        <span className="tiny dim">{t('up.baseline')}: {formatPercent(Number(candidate.metrics.baseline_score ?? 0))}</span>
                        <span className="tiny dim">{t('up.candidate')}: {formatPercent(Number(candidate.metrics.candidate_score ?? 0))}</span>
                        <Badge tone={Number(candidate.metrics.delta) > 0 ? 'ok' : 'warn'}>
                          {t('up.delta')} {formatPercent(Number(candidate.metrics.delta ?? 0), 1)}
                        </Badge>
                        {candidate.metrics.critical_regressions?.length ? <Badge tone="danger">{t('up.critical_regressions')}: {candidate.metrics.critical_regressions.length}</Badge> : null}
                      </div>
                    ) : null}
                    <div className="row">
                      <Button
                        size="sm"
                        icon="play"
                        loading={Boolean(evalJobId)}
                        disabled={candidate.status !== 'proposed'}
                        onClick={async () => {
                          try {
                            const id = await startJob(() => api.evaluateCandidate(candidate.id), t('up.evaluate'))
                            setEvalJobId(id)
                          } catch (error) {
                            notify('danger', error instanceof Error ? error.message : String(error))
                          }
                        }}
                      >
                        {t('up.evaluate')}
                      </Button>
                      <Button
                        size="sm"
                        variant="primary"
                        disabled={candidate.status !== 'evaluated'}
                        onClick={() => { setPromoteTarget(candidate); setApprover('') }}
                      >
                        {t('up.promote')}
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        disabled={candidate.status !== 'promoted'}
                        onClick={async () => {
                          try {
                            await api.rollbackCandidate(candidate.id)
                            candidates.reload()
                            lineage.reload()
                            notify('ok', t('up.rollback'))
                          } catch (error) {
                            notify('danger', error instanceof Error ? error.message : String(error))
                          }
                        }}
                      >
                        {t('up.rollback')}
                      </Button>
                    </div>
                    {candidate.decision.reasons?.length ? (
                      <div style={{ marginTop: 8 }}>
                        <Alert tone="danger" title={t('up.reasons')}>
                          <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
                            {candidate.decision.reasons.map((reason, index) => <li key={index}>{reason}</li>)}
                          </ul>
                        </Alert>
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            )}
            {job.job ? <div style={{ marginTop: 10 }}><Progress value={job.job.progress} indeterminate={job.job.progress === 0} /></div> : null}
          </Card>
        </div>

        <div className="stack">
          <Card accent>
            <CardHeader title={t('up.active')} icon="check" />
            {active ? (
              <KV
                items={[
                  [t('up.candidate_id'), String(active.candidate_id ?? '—')],
                  [t('up.approver_name'), String((active as Record<string, unknown>).approver ?? '—')],
                  [t('up.promoted_at'), formatDate(String((active as Record<string, unknown>).promoted_at ?? ''))],
                ]}
              />
            ) : (
              <span className="small dim">{t('common.none')}</span>
            )}
          </Card>

          <Card>
            <CardHeader title={t('up.holdout_guard')} icon="shield" />
            <Alert tone="ok">{t('up.holdout_ok')}</Alert>
          </Card>

          <Card>
            <CardHeader title={t('up.lineage')} icon="history" />
            {lineage.data?.lineage.length ? (
              <div className="stack" style={{ gap: 8 }}>
                {lineage.data.lineage.slice(0, 12).map((entry, index) => (
                  <div key={index} className="stack" style={{ gap: 2 }}>
                    <div className="row between">
                      <Badge tone="plain">{String(entry.kind)}</Badge>
                      <span className="tiny dim">{formatDate(String(entry.created_at ?? ''))}</span>
                    </div>
                    <span className="mono tiny truncate">{String(entry.ref)}</span>
                  </div>
                ))}
              </div>
            ) : (
              <span className="small dim">{t('common.none')}</span>
            )}
          </Card>

          <Card>
            <CardHeader title={t('up.taxonomy')} icon="layers" />
            <div className="stack" style={{ gap: 6 }}>
              {Object.entries(taxonomy.data?.taxonomy ?? {}).map(([label, description]) => (
                <div key={label}>
                  <span className="tiny mono" style={{ color: 'var(--accent)' }}>{label}</span>
                  <div className="tiny dim">{description}</div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>

      {promoteTarget ? (
        <Modal
          open
          onClose={() => setPromoteTarget(null)}
          title={t('up.promote')}
          footer={
            <>
              <Button variant="ghost" onClick={() => setPromoteTarget(null)}>{t('common.cancel')}</Button>
              <Button
                variant="primary"
                disabled={approver.trim().length < 2}
                onClick={async () => {
                  try {
                    const result = await api.promoteCandidate(promoteTarget.id, approver.trim(), threshold)
                    if (result.promoted) {
                      notify('ok', t('up.promoted'))
                      setPromoteTarget(null)
                    } else {
                      notify('warn', String((result.reasons as string[] | undefined)?.join(' · ') ?? t('up.rejected')))
                    }
                    candidates.reload()
                    lineage.reload()
                  } catch (error) {
                    notify('danger', error instanceof Error ? error.message : String(error))
                  }
                }}
              >
                {t('up.promote')}
              </Button>
            </>
          }
        >
          <div className="stack">
            <Field label={t('up.approver')} htmlFor="approver">
              <Input id="approver" value={approver} onChange={(event) => setApprover(event.target.value)} />
            </Field>
            <Field label={`${t('up.threshold')} — ${formatPercent(threshold)}`} htmlFor="threshold">
              <input id="threshold" type="range" min={0} max={0.5} step={0.01} value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
            </Field>
            <Alert tone="info">{t('up.review_note')}</Alert>
          </div>
        </Modal>
      ) : null}
    </>
  )
}
