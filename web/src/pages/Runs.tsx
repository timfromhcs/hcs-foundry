import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { useApp } from '../context/AppContext'
import { api } from '../lib/api'
import { useAsync } from '../lib/hooks'
import { formatDate, formatDuration, jobDuration, statusTone } from '../lib/format'
import type { Job, JobEvent } from '../lib/types'
import { Badge, Button, Card, EmptyState, Modal, Progress, Select, Spinner } from '../components/ui'

const TYPES = [
  'models.scan',
  'models.verify',
  'datasets.import',
  'datasets.enrich',
  'benchmarks.run',
  'training.smoke',
  'upgrades.evaluate',
  'agent.task',
]

export function Runs() {
  const { t } = useTranslation()
  const [type, setType] = useState('')
  const runs = useAsync(() => api.runs(60, type || undefined), [type])
  const [detail, setDetail] = useState<{ run: Job; events: JobEvent[] } | null>(null)
  const { notify } = useApp()

  useEffect(() => {
    const timer = window.setInterval(() => runs.reload(), 4000)
    return () => window.clearInterval(timer)
  }, [runs])

  return (
    <>
      <div className="row between">
        <div>
          <p className="muted small">{t('runs.subtitle')}</p>
        </div>
        <div className="row">
          <Select value={type} onChange={(event) => setType(event.target.value)} style={{ width: 220 }} aria-label={t('runs.filter_type')}>
            <option value="">{t('runs.all_types')}</option>
            {TYPES.map((item) => (
              <option key={item} value={item}>{item}</option>
            ))}
          </Select>
          <Button variant="ghost" icon="refresh" onClick={() => runs.reload()}>{t('common.refresh')}</Button>
        </div>
      </div>

      <Card flush>
        {runs.loading ? (
          <div style={{ padding: 16 }}><Spinner /></div>
        ) : !runs.data || runs.data.runs.length === 0 ? (
          <div style={{ padding: 16 }}>
            <EmptyState title={t('runs.empty')} icon="history" />
          </div>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>{t('runs.type')}</th>
                  <th>{t('common.status')}</th>
                  <th style={{ width: '20%' }}>{t('common.progress')}</th>
                  <th>{t('runs.duration')}</th>
                  <th>{t('common.created')}</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {runs.data.runs.map((job) => (
                  <tr key={job.id} onClick={() => setDetail({ run: job, events: [] })} style={{ cursor: 'pointer' }}>
                    <td>
                      <div className="mono tiny">{job.type}</div>
                      <div className="tiny dim truncate" style={{ maxWidth: 260 }}>{job.id}</div>
                    </td>
                    <td><Badge tone={statusTone(job.status)}>{t(`common.${job.status}`)}</Badge></td>
                    <td>
                      <Progress value={job.progress} indeterminate={['queued', 'running'].includes(job.status) && job.progress === 0} />
                    </td>
                    <td className="tiny">{formatDuration(jobDuration(job.started_at, job.finished_at))}</td>
                    <td className="tiny dim">{formatDate(job.created_at)}</td>
                    <td className="num">
                      <Button size="sm" variant="ghost" onClick={async (event) => {
                        event.stopPropagation()
                        try {
                          const data = await api.run(job.id)
                          setDetail(data)
                        } catch (error) {
                          notify('danger', error instanceof Error ? error.message : String(error))
                        }
                      }}>
                        {t('common.details')}
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {detail ? (
        <Modal open onClose={() => setDetail(null)} title={`${detail.run.type} — ${detail.run.id}`} wide>
          <div className="stack">
            <div className="row">
              <Badge tone={statusTone(detail.run.status)}>{t(`common.${detail.run.status}`)}</Badge>
              <span className="tiny dim">{t('runs.started')}: {formatDate(detail.run.started_at)}</span>
              <span className="tiny dim">{t('runs.finished')}: {formatDate(detail.run.finished_at)}</span>
              {['queued', 'running'].includes(detail.run.status) ? (
                <Button size="sm" variant="danger" onClick={async () => {
                  try {
                    await api.cancelRun(detail.run.id)
                    const data = await api.run(detail.run.id)
                    setDetail(data)
                  } catch (error) {
                    notify('danger', error instanceof Error ? error.message : String(error))
                  }
                }}>
                  {t('runs.cancel')}
                </Button>
              ) : null}
            </div>

            {detail.run.error ? (
              <div>
                <div className="label" style={{ marginBottom: 4 }}>{t('runs.error')}</div>
                <div className="mono-block" style={{ color: 'var(--danger)' }}>{detail.run.error}</div>
              </div>
            ) : null}

            <div>
              <div className="label" style={{ marginBottom: 4 }}>{t('runs.events')}</div>
              {detail.events.length === 0 ? (
                <span className="small dim">{t('runs.no_events')}</span>
              ) : (
                <div className="mono-block">
                  {detail.events.map((event) => `${event.ts} [${event.kind}] ${event.message}`).join('\n')}
                </div>
              )}
            </div>

            {Object.keys(detail.run.output ?? {}).length > 0 ? (
              <div>
                <div className="label" style={{ marginBottom: 4 }}>{t('runs.output')}</div>
                <div className="mono-block">{JSON.stringify(detail.run.output, null, 2)}</div>
              </div>
            ) : null}
          </div>
        </Modal>
      ) : null}
    </>
  )
}
