import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'

import { useApp } from '../context/AppContext'
import { api } from '../lib/api'
import { useAsync, useJob } from '../lib/hooks'
import { formatDate, formatPercent, statusTone } from '../lib/format'
import type { BenchmarkRun } from '../lib/types'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardHeader,
  EmptyState,
  KV,
  Modal,
  Progress,
  Spinner,
  Toggle,
  CopyButton,
} from '../components/ui'

export function Benchmarks() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { notify, startJob } = useApp()
  const suites = useAsync(() => api.suites(), [])
  const runs = useAsync(() => api.benchmarkRuns(), [])
  const [judge, setJudge] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const job = useJob(jobId)
  const [detail, setDetail] = useState<BenchmarkRun | null>(null)
  const [selected, setSelected] = useState<string[]>([])
  const [comparison, setComparison] = useState<Record<string, unknown> | null>(null)

  useEffect(() => {
    if (job.job?.status === 'succeeded') {
      runs.reload()
      const score = job.job.output?.score
      notify('ok', score === null || score === undefined ? t('bench.result') : `${t('bench.score')}: ${formatPercent(Number(score))}`)
      setJobId(null)
    }
    if (job.job?.status === 'failed') {
      notify('danger', job.job.error ?? t('common.error'))
      setJobId(null)
    }
  }, [job.job?.status, job.job?.error, job.job?.output, runs.reload, notify, t])

  const toggleSelected = (id: string) => {
    setSelected((prev) => (prev.includes(id) ? prev.filter((item) => item !== id) : [...prev.slice(-1), id]))
  }

  return (
    <>
      <div className="row between">
        <div>
          <p className="muted small">{t('bench.subtitle')}</p>
        </div>
        <div className="row">
          <Button variant="ghost" icon="chart" onClick={() => navigate('/upgrades')}>{t('nav.upgrades')}</Button>
        </div>
      </div>

      <Card>
        <CardHeader title={t('bench.suites')} icon="chart" subtitle={t('bench.judge_note')} actions={
          <div style={{ minWidth: 260 }}>
            <Toggle checked={judge} onChange={setJudge} label={t('bench.judge')} />
          </div>
        } />
        {suites.loading ? (
          <Spinner />
        ) : (
          <div className="grid auto">
            {(suites.data?.suites ?? []).map((suite) => (
              <div key={suite.id} className="card tight" style={{ background: 'var(--surface-2)' }}>
                <div className="row between" style={{ marginBottom: 6 }}>
                  <strong className="small">{suite.name}</strong>
                  <Badge tone="plain">{t(`bench.kind.${suite.kind}`)}</Badge>
                </div>
                <p className="tiny muted" style={{ minHeight: 48 }}>{suite.description}</p>
                <div className="row between" style={{ marginTop: 8 }}>
                  <span className="tiny dim">{suite.cases ?? '—'} {t('bench.cases').toLowerCase()} · v{suite.version}</span>
                  {suite.external ? (
                    <CopyButton value={suite.command ?? 'cd web && npm run test:e2e'} label={t('bench.copy_command')} />
                  ) : (
                    <Button
                      size="sm"
                      variant={suite.available ? 'primary' : 'ghost'}
                      disabled={!suite.available || Boolean(jobId)}
                      title={suite.unavailable_reason}
                      onClick={async () => {
                        try {
                          const id = await startJob(() => api.startBenchmark(suite.id, judge), suite.name)
                          setJobId(id)
                        } catch (error) {
                          notify('danger', error instanceof Error ? error.message : String(error))
                        }
                      }}
                    >
                      {suite.available ? t('bench.run_suite') : t('bench.not_available')}
                    </Button>
                  )}
                </div>
                {!suite.available && suite.unavailable_reason ? <div className="tiny dim" style={{ marginTop: 6 }}>{suite.unavailable_reason}</div> : null}
              </div>
            ))}
          </div>
        )}
        {job.job ? (
          <div className="stack" style={{ marginTop: 12 }}>
            <Progress value={job.job.progress} indeterminate={job.job.progress === 0} />
            <span className="tiny dim">{job.job.message}</span>
          </div>
        ) : null}
      </Card>

      <Card>
        <CardHeader
          title={t('bench.runs')}
          icon="history"
          actions={
            <>
              <span className="tiny dim">{t('bench.select_two')}</span>
              <Button
                size="sm"
                disabled={selected.length !== 2}
                onClick={async () => {
                  try {
                    const result = await api.compareBenchmarks(selected[0], selected[1])
                    setComparison(result)
                  } catch (error) {
                    notify('danger', error instanceof Error ? error.message : String(error))
                  }
                }}
              >
                {t('bench.compare')}
              </Button>
            </>
          }
        />
        {runs.loading ? (
          <Spinner />
        ) : !runs.data || runs.data.runs.length === 0 ? (
          <EmptyState title={t('bench.empty')} icon="chart" />
        ) : (
          <div className="table-wrap">
            <table className="table responsive">
              <thead>
                <tr>
                  <th />
                  <th>{t('runs.type')}</th>
                  <th>{t('bench.model')}</th>
                  <th className="num">{t('bench.score')}</th>
                  <th className="num">{t('bench.cases')}</th>
                  <th>{t('common.created')}</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {runs.data.runs.map((run) => {
                  const aggregate = (run.results?.aggregate ?? {}) as Record<string, unknown>
                  return (
                    <tr key={run.id}>
                      <td>
                        <input
                          type="checkbox"
                          checked={selected.includes(run.id)}
                          onChange={() => toggleSelected(run.id)}
                          aria-label={`${t('bench.compare')} ${run.id}`}
                        />
                      </td>
                      <td data-label={t('runs.type')}>
                        <div style={{ fontWeight: 560 }}>{run.suite_id}</div>
                        <div className="tiny dim">v{run.suite_version} · {run.comparability}</div>
                      </td>
                      <td data-label={t('bench.model')} className="mono tiny">{run.model_id ?? '—'}</td>
                      <td data-label={t('bench.score')} className="num">{aggregate.score !== undefined && aggregate.score !== null ? formatPercent(Number(aggregate.score)) : '—'}</td>
                      <td data-label={t('bench.cases')} className="num">{aggregate.passed !== undefined && aggregate.passed !== null ? `${String(aggregate.passed)}/${String(aggregate.total ?? '—')}` : '—'}</td>
                      <td data-label={t('common.created')} className="dim tiny">{formatDate(run.created_at)}</td>
                      <td className="num">
                        <Button size="sm" variant="ghost" onClick={async () => {
                          try {
                            const data = await api.benchmarkRun(run.id)
                            setDetail(data.run)
                          } catch (error) {
                            notify('danger', error instanceof Error ? error.message : String(error))
                          }
                        }}>
                          {t('bench.view_run')}
                        </Button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {comparison ? (
        <Card>
          <CardHeader title={t('bench.compare')} icon="layers" actions={<Button size="sm" variant="ghost" icon="close" onClick={() => setComparison(null)} />} />
          <KV
            items={[
              ['A', `${String((comparison.a as Record<string, unknown>)?.suite)} · ${formatPercent(Number((comparison.a as Record<string, unknown>)?.score ?? 0))}`],
              ['B', `${String((comparison.b as Record<string, unknown>)?.suite)} · ${formatPercent(Number((comparison.b as Record<string, unknown>)?.score ?? 0))}`],
              [t('bench.delta'), comparison.score_delta !== null ? formatPercent(Number(comparison.score_delta), 1) : '—'],
              [t('bench.comparability'), String(comparison.note ?? '')],
            ]}
          />
          {(comparison.changed_cases as Array<Record<string, unknown>> | undefined)?.length ? (
            <div className="table-wrap" style={{ marginTop: 12 }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>{t('bench.case')}</th>
                    <th>{t('common.status')}</th>
                  </tr>
                </thead>
                <tbody>
                  {(comparison.changed_cases as Array<Record<string, unknown>>).map((item, index) => (
                    <tr key={index}>
                      <td className="mono tiny">{String(item.case_id)}</td>
                      <td><Badge tone="warn">{String(item.before)} → {String(item.after)}</Badge></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </Card>
      ) : null}

      {detail ? (
        <Modal open onClose={() => setDetail(null)} title={`${detail.suite_id} — ${detail.id}`} wide footer={
          <>
            <Button size="sm" variant="ghost" icon="download" onClick={async () => {
              const { blob, filename } = await api.benchmarkExport(detail.id, 'md')
              const url = URL.createObjectURL(blob)
              const anchor = document.createElement('a')
              anchor.href = url
              anchor.download = filename ?? `${detail.id}.md`
              anchor.click()
              URL.revokeObjectURL(url)
            }}>
              Markdown
            </Button>
            <Button size="sm" variant="ghost" icon="download" onClick={async () => {
              const { blob, filename } = await api.benchmarkExport(detail.id, 'json')
              const url = URL.createObjectURL(blob)
              const anchor = document.createElement('a')
              anchor.href = url
              anchor.download = filename ?? `${detail.id}.json`
              anchor.click()
              URL.revokeObjectURL(url)
            }}>
              JSON
            </Button>
          </>
        }>
          <div className="stack">
            <KV
              items={[
                [t('bench.score'), (() => {
                  const score = (detail.results?.aggregate as Record<string, unknown> | undefined)?.score
                  return score === null || score === undefined ? '—' : formatPercent(Number(score))
                })()],
                [t('bench.model'), detail.model_id ?? '—'],
                [t('bench.suite_version'), detail.suite_version],
                [t('bench.comparability'), detail.comparability],
                [t('models.hash'), detail.model_hash ?? '—'],
              ]}
            />
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>{t('bench.case')}</th>
                    <th>{t('bench.result')}</th>
                    <th>{t('bench.detail')}</th>
                  </tr>
                </thead>
                <tbody>
                  {(detail.cases ?? []).map((item) => (
                    <tr key={item.case_id}>
                      <td className="mono tiny">{item.case_id}</td>
                      <td><Badge tone={statusTone(item.status)}>{item.status}</Badge></td>
                      <td className="tiny">{String(item.detail?.detail ?? '')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Alert tone="info">{t('bench.judge_note')}</Alert>
          </div>
        </Modal>
      ) : null}
    </>
  )
}
