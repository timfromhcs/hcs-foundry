import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'

import { useApp } from '../context/AppContext'
import { api } from '../lib/api'
import { useAsync } from '../lib/hooks'
import { formatBytes, formatDuration, formatNumber, jobDuration, statusTone, timeAgo } from '../lib/format'
import type { Job } from '../lib/types'
import { Alert, Badge, Button, Card, CardHeader, EmptyState, KV, Progress, Stat } from '../components/ui'
import { Icon } from '../components/icons'

export function Overview() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { profile, budget, scanMachine, notify, jobs } = useApp()
  const [status, setStatus] = useState<Record<string, unknown> | null>(null)
  const [scanning, setScanning] = useState(false)
  const runs = useAsync(() => api.runs(8), [])

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const data = await api.inferenceStatus()
        if (!cancelled) setStatus(data)
      } catch {
        if (!cancelled) setStatus(null)
      }
    }
    void load()
    const timer = window.setInterval(load, 4000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [])

  const active = jobs.filter((job) => ['queued', 'running'].includes(job.status))
  const model = status?.model as Record<string, unknown> | null | undefined
  const gpu = profile?.gpu?.devices?.[0]
  const ramUsed = profile ? (profile.memory.total_bytes ?? 0) - (profile.memory.available_bytes ?? 0) : 0
  const datasets = useAsync(() => api.datasets(), [])
  const trainingRuns = useAsync(() => api.trainingRuns(), [])
  const checklist = [
    { done: Boolean(profile), label: t('overview.step_scan'), to: '/machine' },
    { done: Boolean(profile?.runtimes?.llama_server?.candidates?.length), label: t('overview.step_models'), to: '/models' },
    { done: Boolean(model), label: t('overview.step_load'), to: '/models' },
    { done: Boolean((datasets.data?.datasets ?? []).length), label: t('overview.step_data'), to: '/data' },
    { done: Boolean((trainingRuns.data?.runs ?? []).length), label: t('overview.step_train'), to: '/training' },
    { done: false, label: t('overview.step_play'), to: '/playground' },
  ]

  return (
    <>
      <div className="grid cols-4">
        <Card>
          <Stat
            label={t('overview.tier')}
            value={profile ? `Tier ${profile.tier.tier}` : '—'}
            hint={profile ? profile.tier.label : t('machine.no_profile')}
            small
          />
        </Card>
        <Card>
          <Stat
            label={t('overview.ram')}
            value={profile ? formatBytes(profile.memory.available_bytes) : '—'}
            hint={profile ? `${formatBytes(ramUsed)} / ${formatBytes(profile.memory.total_bytes)}` : t('common.not_measured')}
            small
          />
        </Card>
        <Card>
          <Stat
            label={t('overview.vram')}
            value={profile ? formatBytes(profile.gpu.dedicated_vram_bytes ?? 0) : '—'}
            hint={gpu?.unified_memory ? t('machine.unified') : gpu?.name}
            small
          />
        </Card>
        <Card>
          <Stat
            label={t('overview.safe_budget')}
            value={budget ? formatBytes(budget.safe_budget_bytes) : '—'}
            hint={t('overview.available')}
            small
          />
        </Card>
      </div>

      <div className="grid sidebar-right">
        <div className="stack">
          <Card>
            <CardHeader
              title={t('overview.model')}
              subtitle={model ? String(model.name) : t('overview.no_model')}
              icon="box"
              actions={
                model ? (
                  <>
                    <Button size="sm" variant="ghost" icon="chat" onClick={() => navigate('/playground')}>
                      {t('nav.playground')}
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      icon="stop"
                      onClick={async () => {
                        try {
                          await api.unloadModel()
                          setStatus(null)
                          notify('info', t('models.unload'))
                        } catch (error) {
                          notify('danger', error instanceof Error ? error.message : String(error))
                        }
                      }}
                    >
                      {t('models.unload')}
                    </Button>
                  </>
                ) : (
                  <Button size="sm" icon="box" onClick={() => navigate('/models')}>
                    {t('overview.load_model')}
                  </Button>
                )
              }
            />
            {model ? (
              <div className="grid cols-3">
                <Stat label={t('models.quant')} value={String(model.quantization ?? '—')} small />
                <Stat label={t('models.context')} value={formatNumber(Number(model.context_max ?? 0))} small />
                <Stat
                  label={t('models.gpu_layers')}
                  value={String((status?.config as Record<string, unknown> | undefined)?.gpu_layers ?? '—')}
                  small
                />
              </div>
            ) : (
              <EmptyState title={t('overview.no_model')} icon="box">
                {t('play.error_no_model')}
              </EmptyState>
            )}
            {model ? (
              <div style={{ marginTop: 12 }}>
                <KV
                  items={[
                    [t('models.path'), <span className="mono tiny">{String(model.id)}</span>],
                    [t('models.size'), formatBytes(Number(model.size_bytes ?? 0))],
                    [t('common.status'), <Badge tone={statusTone(String(status?.status ?? ''))}>{String(status?.status ?? '—')}</Badge>],
                  ]}
                />
              </div>
            ) : null}
          </Card>

          <Card>
            <CardHeader
              title={t('overview.active_jobs')}
              subtitle={active.length ? `${active.length} ${t('overview.jobs_running', { count: active.length })}` : t('jobs.none')}
              icon="history"
              actions={
                <Button size="sm" variant="ghost" onClick={() => navigate('/runs')}>
                  {t('common.view_all')}
                </Button>
              }
            />
            {active.length === 0 ? (
              <EmptyState title={t('jobs.none')} icon="history" />
            ) : (
              <div className="stack">
                {active.map((job) => (
                  <div key={job.id} className="stack" style={{ gap: 6 }}>
                    <div className="row between">
                      <span className="small truncate">{job.label}</span>
                      <Badge tone={statusTone(job.status)}>{t(`common.${job.status}`)}</Badge>
                    </div>
                    <Progress value={job.progress} indeterminate={job.status === 'queued'} />
                    <span className="tiny dim truncate">{job.message ?? ''}</span>
                  </div>
                ))}
              </div>
            )}
          </Card>

          <Card>
            <CardHeader title={t('overview.recent_runs')} icon="history" actions={<Button size="sm" variant="ghost" onClick={() => navigate('/runs')}>{t('common.view_all')}</Button>} />
            {runs.loading ? (
              <div className="skeleton" style={{ height: 60 }} />
            ) : !runs.data || runs.data.runs.length === 0 ? (
              <EmptyState title={t('overview.no_runs')} icon="history" />
            ) : (
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>{t('runs.type')}</th>
                      <th>{t('common.status')}</th>
                      <th>{t('runs.duration')}</th>
                      <th>{t('common.created')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.data.runs.map((job: Job) => (
                      <tr key={job.id}>
                        <td className="mono tiny">{job.type}</td>
                        <td><Badge tone={statusTone(job.status)}>{t(`common.${job.status}`)}</Badge></td>
                        <td>{formatDuration(jobDuration(job.started_at, job.finished_at))}</td>
                        <td className="dim">{timeAgo(job.created_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>

        <div className="stack">
          <Card accent>
            <CardHeader title={t('overview.quick_actions')} icon="bolt" />
            <div className="stack">
              <Button block variant="primary" icon="cpu" loading={scanning} onClick={async () => {
                setScanning(true)
                try {
                  await scanMachine()
                } catch (error) {
                  notify('danger', error instanceof Error ? error.message : String(error))
                } finally {
                  setScanning(false)
                }
              }}>
                {t('overview.scan_machine')}
              </Button>
              <Button block icon="box" onClick={() => navigate('/models')}>{t('overview.load_model')}</Button>
              <Button block icon="chat" onClick={() => navigate('/playground')}>{t('overview.open_playground')}</Button>
              <Button block icon="chart" onClick={() => navigate('/benchmarks')}>{t('nav.benchmarks')}</Button>
            </div>
          </Card>

          <Card>
            <CardHeader title={t('overview.getting_started')} icon="check" />
            <div className="stack" style={{ gap: 8 }}>
              {checklist.map((item, index) => (
                <Link key={index} to={item.to} className="row" style={{ gap: 10, color: 'inherit', textDecoration: 'none' }}>
                  <span className={`dot ${item.done ? 'ok' : ''}`} aria-hidden="true" />
                  <span className="small grow" style={{ textDecoration: item.done ? 'line-through' : 'none', opacity: item.done ? 0.6 : 1 }}>
                    {item.label}
                  </span>
                  <Icon name="chevron" size={13} style={{ color: 'var(--text-3)' }} />
                </Link>
              ))}
            </div>
            <p className="small muted" style={{ marginTop: 12, marginBottom: 0 }}>
              {t('overview.tutorial_hint')}{' '}
              <a href="https://github.com/timfromhcs/hcs-foundry/blob/main/docs/TUTORIAL.md" target="_blank" rel="noopener noreferrer">
                {t('overview.tutorial_link')}
              </a>
            </p>
          </Card>

          <Card>
            <CardHeader title={t('overview.warnings')} icon="alert" />
            {profile && profile.warnings.length > 0 ? (
              <div className="stack" style={{ gap: 8 }}>
                {profile.warnings.map((warning, index) => (
                  <Alert key={index} tone="warn">{warning}</Alert>
                ))}
              </div>
            ) : (
              <span className="small dim">{t('overview.none_warnings')}</span>
            )}
          </Card>

          <Card>
            <CardHeader title={t('overview.machine')} icon="cpu" actions={<Button size="sm" variant="ghost" onClick={() => navigate('/machine')}>{t('common.details')}</Button>} />
            {profile ? (
              <KV
                items={[
                  [t('machine.cpu'), `${profile.cpu.model ?? '—'}`],
                  [t('machine.cores'), `${profile.cpu.physical_cores ?? '—'} / ${profile.cpu.logical_cores ?? '—'}`],
                  [t('overview.gpu'), gpu?.name ?? '—'],
                  [t('machine.runtimes'), profile.runtimes.llama_server?.candidates?.length ? `${profile.runtimes.llama_server.candidates.length}` : t('common.none')],
                ]}
              />
            ) : (
              <EmptyState title={t('machine.no_profile')} icon="cpu">{t('machine.no_profile_hint')}</EmptyState>
            )}
          </Card>
        </div>
      </div>
    </>
  )
}
