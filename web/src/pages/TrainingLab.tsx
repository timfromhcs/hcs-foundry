import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { useApp } from '../context/AppContext'
import { api } from '../lib/api'
import { useAsync, useJob } from '../lib/hooks'
import { formatBytes, formatDate, statusTone } from '../lib/format'
import type { TrainingEnvironment } from '../lib/types'
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
  Progress,
  Select,
  Spinner,
} from '../components/ui'

export function TrainingLab() {
  const { t } = useTranslation()
  const { notify, startJob } = useApp()
  const env = useAsync(() => api.trainingEnvironment(), [])
  const datasets = useAsync(() => api.datasets(), [])
  const runs = useAsync(() => api.trainingRuns(), [])
  const [smokeJobId, setSmokeJobId] = useState<string | null>(null)
  const smoke = useJob(smokeJobId)
  const [dataset, setDataset] = useState('')
  const [baseModel, setBaseModel] = useState('')
  const [method, setMethod] = useState('lora_sft')
  const [preflight, setPreflight] = useState<Record<string, unknown> | null>(null)
  const [checking, setChecking] = useState(false)

  useEffect(() => {
    if (smoke.job?.status === 'succeeded') {
      runs.reload()
      notify('ok', t('training.smoke_passed'))
      setSmokeJobId(null)
    }
    if (smoke.job?.status === 'failed') {
      notify('danger', smoke.job.error ?? t('training.smoke_failed'))
      setSmokeJobId(null)
    }
  }, [smoke.job?.status, smoke.job?.error, runs.reload, notify, t])

  const environment = env.data as TrainingEnvironment | null
  const frozen = (datasets.data?.datasets ?? []).filter((item) => item.status === 'frozen' && item.rights_status === 'confirmed')

  return (
    <>
      <div>
        <p className="muted small">{t('training.subtitle')}</p>
      </div>

      <div className="grid sidebar-right">
        <div className="stack">
          <Card accent>
            <CardHeader
              title={t('training.environment')}
              icon="graduation"
              actions={environment ? (
                <Badge tone={environment.state === 'gpu' ? 'ok' : environment.state === 'cpu-experimental' ? 'warn' : 'danger'}>
                  {t(`training.${environment.state === 'gpu' ? 'gpu' : environment.state === 'cpu-experimental' ? 'cpu' : 'unsupported'}`)}
                </Badge>
              ) : null}
            />
            {env.loading ? (
              <Spinner />
            ) : environment ? (
              <div className="stack">
                <p className="small muted">{environment.reason}</p>
                <KV
                  items={[
                    [t('training.device'), environment.device],
                    ['GPU', environment.gpu_name ?? t('common.none')],
                    [t('training.free_disk'), environment.disk ? formatBytes(environment.disk.free_bytes) : '—'],
                    [t('training.verified'), <Badge tone={environment.can_finetune_verified ? 'ok' : 'warn'}>{environment.can_finetune_verified ? t('common.yes') : t('training.not_verified')}</Badge>],
                  ]}
                />
                <div>
                  <div className="label" style={{ marginBottom: 6 }}>{t('training.packages')}</div>
                  <div className="pill-row">
                    {Object.entries(environment.packages).map(([name, version]) => {
                      const usable = environment.importable?.[name]
                      const broken = Boolean(version) && usable === false
                      return (
                        <Badge key={name} tone={!version ? 'plain' : broken ? 'danger' : 'ok'} title={broken ? environment.import_errors?.[name] : undefined}>
                          {name}{version ? ` ${version.split('+')[0]}` : ''}{broken ? ' ⚠' : ''}
                        </Badge>
                      )
                    })}
                  </div>
                </div>
                {environment.missing_packages.length > 0 ? (
                  <Alert tone="warn" title={t('training.missing')}>
                    {environment.missing_packages.join(', ')}
                    {environment.install_hint ? <div className="mono tiny" style={{ marginTop: 6 }}>{environment.install_hint}</div> : null}
                  </Alert>
                ) : null}
              </div>
            ) : (
              <Alert tone="danger">{env.error}</Alert>
            )}
          </Card>

          <Card>
            <CardHeader title={t('training.smoke_run')} icon="bolt" subtitle={t('training.smoke_hint')} />
            <div className="stack">
              <Button
                variant="primary"
                icon="play"
                loading={Boolean(smokeJobId)}
                disabled={!environment?.can_smoke}
                onClick={async () => {
                  try {
                    const id = await startJob(() => api.trainingSmoke(), t('training.smoke_run'))
                    setSmokeJobId(id)
                  } catch (error) {
                    notify('danger', error instanceof Error ? error.message : String(error))
                  }
                }}
              >
                {t('training.smoke')}
              </Button>
              {smoke.job ? (
                <>
                  <Progress value={smoke.job.progress} indeterminate={smoke.job.progress === 0} />
                  <div className="row between">
                    <span className="tiny dim">{smoke.job.message}</span>
                    <Badge tone={statusTone(smoke.job.status)}>{t(`common.${smoke.job.status}`)}</Badge>
                  </div>
                  {smoke.job.output && Object.keys(smoke.job.output).length > 0 ? (
                    <div className="mono-block">{JSON.stringify(smoke.job.output, null, 2)}</div>
                  ) : null}
                </>
              ) : null}
              <Alert tone="info">{t('training.honesty_note')}</Alert>
            </div>
          </Card>

          <Card>
            <CardHeader title={t('training.preflight_title')} icon="shield" />
            <div className="stack">
              <div className="grid cols-2">
                <Field label={t('training.dataset')} htmlFor="train-dataset" hint={frozen.length === 0 ? t('training.no_dataset') : undefined}>
                  <Select id="train-dataset" value={dataset} onChange={(event) => setDataset(event.target.value)}>
                    <option value="">—</option>
                    {frozen.map((item) => (
                      <option key={item.id} value={item.id}>{item.name} ({item.record_count})</option>
                    ))}
                  </Select>
                </Field>
                <Field label={t('training.method')} htmlFor="train-method">
                  <Select id="train-method" value={method} onChange={(event) => setMethod(event.target.value)}>
                    <option value="lora_sft">{t('training.method_lora')}</option>
                    <option value="dpo">{t('training.method_dpo')}</option>
                  </Select>
                </Field>
              </div>
              <Field label={t('training.base_model')} hint={t('training.base_model_hint')} htmlFor="train-base">
                <Input id="train-base" value={baseModel} onChange={(event) => setBaseModel(event.target.value)} placeholder="C:\\models\\Qwen3-4B" />
              </Field>
              <Button
                icon="search"
                loading={checking}
                disabled={!dataset}
                onClick={async () => {
                  setChecking(true)
                  try {
                    const plan = await api.trainingPreflight({ dataset_id: dataset, base_model: baseModel || null, method })
                    setPreflight(plan)
                  } catch (error) {
                    notify('danger', error instanceof Error ? error.message : String(error))
                  } finally {
                    setChecking(false)
                  }
                }}
              >
                {t('training.preflight')}
              </Button>

              {preflight ? (
                <div className="stack">
                  <div className="row between">
                    <span className="small">{preflight.can_start ? t('training.can_start') : t('training.cannot_start')}</span>
                    <Badge tone={preflight.can_start ? 'ok' : 'danger'}>{preflight.can_start ? t('common.yes') : t('common.no')}</Badge>
                  </div>
                  {((preflight.blockers as string[]) ?? []).length > 0 ? (
                    <Alert tone="danger" title={t('training.blockers')}>
                      <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
                        {(preflight.blockers as string[]).map((item, index) => <li key={index}>{item}</li>)}
                      </ul>
                    </Alert>
                  ) : null}
                  {((preflight.warnings as string[]) ?? []).length > 0 ? (
                    <Alert tone="warn" title={t('training.warnings')}>
                      <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
                        {(preflight.warnings as string[]).map((item, index) => <li key={index}>{item}</li>)}
                      </ul>
                    </Alert>
                  ) : null}
                  <div>
                    <div className="label" style={{ marginBottom: 6 }}>{t('training.preset')}</div>
                    <div className="mono-block">{JSON.stringify(preflight.preset, null, 2)}</div>
                  </div>
                </div>
              ) : null}
            </div>
          </Card>

          <Card>
            <CardHeader title={t('training.runs')} icon="history" />
            {runs.loading ? (
              <Spinner />
            ) : !runs.data || runs.data.runs.length === 0 ? (
              <EmptyState title={t('training.empty')} icon="graduation" />
            ) : (
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>{t('runs.type')}</th>
                      <th>{t('common.status')}</th>
                      <th>{t('training.loss')}</th>
                      <th>{t('common.created')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.data.runs.map((run, index) => {
                      const metrics = (run.metrics as Record<string, unknown> | undefined) ?? {}
                      return (
                        <tr key={index}>
                          <td className="mono tiny">{String(run.backend ?? '')}</td>
                          <td><Badge tone={statusTone(String(run.status ?? ''))}>{String(run.status ?? '')}</Badge></td>
                          <td className="tiny">
                            {metrics.loss_first !== undefined ? `${Number(metrics.loss_first).toFixed(4)} → ${Number(metrics.loss_last).toFixed(4)}` : '—'}
                          </td>
                          <td className="dim tiny">{formatDate(String(run.created_at ?? ''))}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>

        <div className="stack">
          <Card>
            <CardHeader title={t('training.honesty_note')} icon="shield" />
            <p className="small muted">
              {t('training.base_model_hint')}
            </p>
          </Card>
        </div>
      </div>
    </>
  )
}
