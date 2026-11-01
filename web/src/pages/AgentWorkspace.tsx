import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { useApp } from '../context/AppContext'
import { api } from '../lib/api'
import { useAsync } from '../lib/hooks'
import { formatDate, statusTone } from '../lib/format'
import type { AgentTask } from '../lib/types'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardHeader,
  EmptyState,
  Field,
  KV,
  Select,
  Spinner,
  Textarea,
} from '../components/ui'

export function AgentWorkspace() {
  const { t } = useTranslation()
  const { notify } = useApp()
  const tasks = useAsync(() => api.agentTasks(), [])
  const tools = useAsync(() => api.agentTools(), [])
  const [goal, setGoal] = useState('')
  const [autonomy, setAutonomy] = useState('supervised')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [detail, setDetail] = useState<{ task: AgentTask; events: Array<Record<string, unknown>> } | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!selectedId && tasks.data?.tasks.length) setSelectedId(tasks.data.tasks[0].id)
  }, [tasks.data, selectedId])

  useEffect(() => {
    if (!selectedId) {
      setDetail(null)
      return
    }
    let cancelled = false
    const load = async () => {
      try {
        const data = await api.agentTask(selectedId)
        if (!cancelled) setDetail(data as unknown as { task: AgentTask; events: Array<Record<string, unknown>> })
      } catch {
        /* keep previous */
      }
    }
    void load()
    const timer = window.setInterval(load, 2500)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [selectedId])

  useEffect(() => {
    const timer = window.setInterval(() => tasks.reload(), 5000)
    return () => window.clearInterval(timer)
  }, [tasks.reload])

  const task = detail?.task ?? tasks.data?.tasks.find((item) => item.id === selectedId) ?? null

  return (
    <>
      <div>
        <p className="muted small">{t('agent.subtitle')}</p>
      </div>

      <div className="grid sidebar-right">
        <div className="stack">
          <Card accent>
            <CardHeader title={t('agent.goal')} icon="bot" subtitle={t('agent.autonomy_hint')} />
            <div className="stack">
              <Field label={t('agent.goal')} htmlFor="agent-goal">
                <Textarea id="agent-goal" value={goal} onChange={(event) => setGoal(event.target.value)} placeholder={t('agent.goal_hint')} style={{ minHeight: 80 }} />
              </Field>
              <div className="grid cols-2">
                <Field label={t('agent.autonomy')} htmlFor="agent-autonomy">
                  <Select id="agent-autonomy" value={autonomy} onChange={(event) => setAutonomy(event.target.value)}>
                    <option value="suggest">{t('agent.suggest')}</option>
                    <option value="supervised">{t('agent.supervised')}</option>
                    <option value="trusted">{t('agent.trusted')}</option>
                  </Select>
                </Field>
                <div className="row" style={{ alignItems: 'flex-end' }}>
                  <Button
                    variant="primary"
                    icon="play"
                    block
                    loading={busy}
                    disabled={goal.trim().length < 3}
                    onClick={async () => {
                      setBusy(true)
                      try {
                        const data = await api.createAgentTask(goal.trim(), autonomy)
                        setGoal('')
                        tasks.reload()
                        setSelectedId(data.task.id)
                      } catch (error) {
                        notify('danger', error instanceof Error ? error.message : String(error))
                      } finally {
                        setBusy(false)
                      }
                    }}
                  >
                    {t('agent.run')}
                  </Button>
                </div>
              </div>
            </div>
          </Card>

          <Card>
            <CardHeader title={t('agent.tasks')} icon="history" />
            {tasks.loading ? (
              <Spinner />
            ) : !tasks.data || tasks.data.tasks.length === 0 ? (
              <EmptyState title={t('agent.empty')} icon="bot" />
            ) : (
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>{t('agent.goal')}</th>
                      <th>{t('agent.autonomy')}</th>
                      <th>{t('common.status')}</th>
                      <th>{t('common.created')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tasks.data.tasks.map((item) => (
                      <tr
                        key={item.id}
                        onClick={() => setSelectedId(item.id)}
                        style={{ cursor: 'pointer', background: selectedId === item.id ? 'var(--surface-2)' : undefined }}
                      >
                        <td className="truncate" style={{ maxWidth: 360 }}>{item.goal}</td>
                        <td><Badge tone="plain">{item.autonomy}</Badge></td>
                        <td><Badge tone={statusTone(item.status)}>{item.status}</Badge></td>
                        <td className="dim tiny">{formatDate(item.created_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          {task ? (
            <Card>
              <CardHeader
                title={task.goal}
                subtitle={`${task.id} · ${task.autonomy}`}
                icon="bot"
                actions={<Badge tone={statusTone(task.status)}>{task.status}</Badge>}
              />
              <div className="stack">
                <div>
                  <div className="label" style={{ marginBottom: 6 }}>{t('agent.plan')}</div>
                  {task.plan.length === 0 ? (
                    <span className="small dim">{t('agent.no_plan')}</span>
                  ) : (
                    <div className="stack" style={{ gap: 6 }}>
                      {task.plan.map((step, index) => (
                        <div key={index} className="row between card tight" style={{ background: 'var(--surface-2)' }}>
                          <div className="row" style={{ gap: 8 }}>
                            <span className="tiny dim">{index + 1}.</span>
                            <span className="mono tiny">{step.tool}</span>
                            <span className="tiny dim truncate" style={{ maxWidth: 320 }}>{step.description}</span>
                          </div>
                          <div className="row" style={{ gap: 4 }}>
                            <Badge tone={step.mutating ? 'warn' : 'ok'}>{step.mutating ? t('agent.mutating') : t('agent.read_only')}</Badge>
                            {step.requires_approval ? <Badge tone="danger">{t('agent.requires_approval')}</Badge> : null}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="row">
                  {['awaiting_approval', 'suggested'].includes(task.status) ? (
                    <Button
                      variant="primary"
                      icon="check"
                      onClick={async () => {
                        try {
                          await api.approveAgentTask(task.id)
                          tasks.reload()
                          notify('ok', t('agent.approve_run'))
                        } catch (error) {
                          notify('danger', error instanceof Error ? error.message : String(error))
                        }
                      }}
                    >
                      {t('agent.approve_run')}
                    </Button>
                  ) : null}
                  {['queued', 'running'].includes(task.status) ? (
                    <Button
                      variant="danger"
                      icon="stop"
                      onClick={async () => {
                        try {
                          await api.cancelAgentTask(task.id)
                          tasks.reload()
                        } catch (error) {
                          notify('danger', error instanceof Error ? error.message : String(error))
                        }
                      }}
                    >
                      {t('agent.cancel')}
                    </Button>
                  ) : null}
                </div>

                {task.result?.summary ? (
                  <div>
                    <div className="label" style={{ marginBottom: 6 }}>{t('agent.summary')}</div>
                    <Alert tone="ok">{String(task.result.summary)}</Alert>
                  </div>
                ) : null}

                <div>
                  <div className="label" style={{ marginBottom: 6 }}>{t('agent.timeline')}</div>
                  {!detail?.events?.length ? (
                    <span className="small dim">{t('agent.no_events')}</span>
                  ) : (
                    <div className="mono-block">
                      {detail.events
                        .map((event) => `${String(event.ts ?? '')} [${String(event.kind ?? '')}] ${String(event.message ?? '')}`)
                        .join('\n')}
                    </div>
                  )}
                </div>
              </div>
            </Card>
          ) : null}
        </div>

        <div className="stack">
          <Card>
            <CardHeader title={t('agent.tools')} icon="terminal" />
            <div className="stack" style={{ gap: 8 }}>
              {(tools.data?.tools ?? []).map((tool) => (
                <div key={tool.name} className="row between">
                  <div className="row" style={{ gap: 6 }}>
                    <span className={`dot ${tool.mutating ? 'warn' : 'ok'}`} />
                    <span className="mono tiny">{tool.name}</span>
                  </div>
                  <Badge tone={tool.mutating ? 'warn' : 'plain'}>{tool.mutating ? t('agent.mutating') : t('agent.read_only')}</Badge>
                </div>
              ))}
            </div>
          </Card>

          <Card>
            <CardHeader title={t('agent.risk')} icon="shield" />
            <div className="stack" style={{ gap: 6 }}>
              <span className="tiny muted">• {t('agent.autonomy_hint')}</span>
              <span className="tiny muted">• {t('agent.retry_note')}</span>
              <span className="tiny muted">• {t('agent.no_shell')}</span>
            </div>
          </Card>

          {task ? (
            <Card>
              <CardHeader title={t('common.details')} icon="layers" />
              <KV
                items={[
                  [t('common.status'), task.status],
                  [t('common.created'), formatDate(task.created_at)],
                  [t('runs.finished'), task.finished_at ? formatDate(task.finished_at) : '—'],
                ]}
              />
            </Card>
          ) : null}
        </div>
      </div>
    </>
  )
}
