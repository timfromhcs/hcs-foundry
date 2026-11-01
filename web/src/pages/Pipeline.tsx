import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { useApp } from '../context/AppContext'
import { api } from '../lib/api'
import { useAsync, useJob } from '../lib/hooks'
import { formatBytes, statusTone } from '../lib/format'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardHeader,
  EmptyState,
  Field,
  Input,
  Progress,
  Select,
  Spinner,
} from '../components/ui'

const QUANTS = ['F16', 'BF16', 'Q8_0', 'Q6_K', 'Q5_K_M', 'Q4_K_M', 'Q4_K_S', 'Q3_K_M', 'Q2_K']
const STAGES = ['pull', 'dataset', 'train', 'merge', 'export', 'quantize', 'moe', 'report']

export function Pipeline() {
  const { t } = useTranslation()
  const { notify, startJob } = useApp()
  const models = useAsync(() => api.models(), [])
  const datasets = useAsync(() => api.datasets(), [])
  const tools = useAsync(() => api.ggufTools(), [])
  const sources = useAsync(() => api.sources(), [])
  const playbook = useAsync(() => api.trainingPlaybook(), [])

  const [modelId, setModelId] = useState('')
  const [sourceFile, setSourceFile] = useState('')
  const [method, setMethod] = useState('Q4_K_M')
  const [experts, setExperts] = useState(4)
  const [topK, setTopK] = useState(2)
  const [ngl, setNgl] = useState('0,16,99')
  const [datasetId, setDatasetId] = useState('')
  const [modelPath, setModelPath] = useState('')
  const [algorithm, setAlgorithm] = useState('auto')
  const [teacher, setTeacher] = useState('local')
  const [steps, setSteps] = useState(8)
  const [stages, setStages] = useState<string[]>(['export', 'quantize', 'report'])
  const [jobId, setJobId] = useState<string | null>(null)
  const job = useJob(jobId)

  useEffect(() => {
    if (!modelId && models.data?.models.length) setModelId(models.data.models[0].id)
  }, [models.data, modelId])

  useEffect(() => {
    if (!datasetId && datasets.data?.datasets.length) setDatasetId(datasets.data.datasets[0].id)
  }, [datasets.data, datasetId])

  useEffect(() => {
    if (job.job?.status === 'succeeded') {
      notify('ok', t('common.succeeded'))
      playbook.reload()
      sources.reload()
      models.reload()
      setJobId(null)
    }
    if (job.job?.status === 'failed') {
      notify('danger', job.job.error ?? t('common.failed'))
      setJobId(null)
    }
  }, [job.job?.status, job.job?.error, notify, playbook, sources, models, t])

  const run = async (label: string, request: () => Promise<{ job_id: string }>) => {
    try {
      setJobId(await startJob(request, label))
    } catch (error) {
      notify('danger', error instanceof Error ? error.message : String(error))
    }
  }

  const modelOptions = (models.data?.models ?? []).map((model) => (
    <option key={model.id} value={model.id}>{model.name}</option>
  ))

  return (
    <>
      <div>
        <p className="muted small">{t('pipeline.subtitle')}</p>
      </div>

      {job.job ? (
        <Card>
          <div className="row between">
            <span className="small">{job.job.type}</span>
            <Badge tone={statusTone(job.job.status)}>{t(`common.${job.job.status}`)}</Badge>
          </div>
          <Progress value={job.job.progress} indeterminate={job.job.progress === 0} />
          <div className="tiny dim truncate" style={{ marginTop: 6 }}>{job.job.message}</div>
          {job.events.length > 0 ? (
            <div className="mono-block" style={{ maxHeight: 140, marginTop: 8 }}>
              {job.events.slice(-8).map((event) => `${event.ts} ${event.message}`).join('\n')}
            </div>
          ) : null}
        </Card>
      ) : null}

      <div className="grid cols-2">
        <Card>
          <CardHeader title={t('pipeline.gguf')} icon="layers" subtitle={tools.data?.ready ? 'llama.cpp tools ready' : t('pipeline.tools_missing')} />
          <div className="stack">
            <Field label={t('common.name')} htmlFor="pl-model">
              <Select id="pl-model" value={modelId} onChange={(event) => setModelId(event.target.value)}>
                <option value="">—</option>
                {modelOptions}
              </Select>
            </Field>
            <Button variant="primary" icon="download" disabled={!modelId || !tools.data?.ready} onClick={() => void run(t('pipeline.export'), () => api.ggufExport(modelId))}>
              {t('pipeline.export')}
            </Button>
            <div className="divider" />
            <Field label={t('pipeline.source')} htmlFor="pl-source">
              <Input id="pl-source" value={sourceFile} onChange={(event) => setSourceFile(event.target.value)} placeholder="E:\\data\\gguf\\model-f16.gguf" />
            </Field>
            <div className="grid cols-2">
              <Field label={t('pipeline.method')} htmlFor="pl-method">
                <Select id="pl-method" value={method} onChange={(event) => setMethod(event.target.value)}>
                  {QUANTS.map((item) => <option key={item} value={item}>{item}</option>)}
                </Select>
              </Field>
              <div className="row" style={{ alignItems: 'flex-end' }}>
                <Button icon="bolt" disabled={!sourceFile} onClick={() => void run(t('pipeline.quantize'), () => api.ggufQuantize(sourceFile, method))}>
                  {t('pipeline.quantize')}
                </Button>
              </div>
            </div>
            <Button variant="ghost" icon="refresh" disabled={!sourceFile} onClick={() => void run(t('pipeline.dequantize'), () => api.ggufDequantize(sourceFile, 'F16'))}>
              {t('pipeline.dequantize')}
            </Button>
          </div>
        </Card>

        <Card>
          <CardHeader title={t('pipeline.moe')} icon="sparkles" subtitle={t('pipeline.moe_note')} />
          <div className="stack">
            <Field label={t('common.name')} htmlFor="pl-moe-model">
              <Select id="pl-moe-model" value={modelId} onChange={(event) => setModelId(event.target.value)}>
                <option value="">—</option>
                {modelOptions}
              </Select>
            </Field>
            <div className="grid cols-2">
              <Field label={t('pipeline.experts')} htmlFor="pl-experts">
                <Input id="pl-experts" type="number" min={2} max={16} value={experts} onChange={(event) => setExperts(Number(event.target.value))} />
              </Field>
              <Field label={t('pipeline.topk')} htmlFor="pl-topk">
                <Input id="pl-topk" type="number" min={1} max={experts} value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
              </Field>
            </div>
            <Button variant="primary" icon="layers" disabled={!modelId} onClick={() => void run(t('pipeline.build_moe'), () => api.moeBuild(modelId, experts, topK))}>
              {t('pipeline.build_moe')}
            </Button>
            <Alert tone="warn">{t('pipeline.note_4b')}</Alert>
          </div>
        </Card>

        <Card>
          <CardHeader title={t('pipeline.vulkan')} icon="bolt" />
          <div className="stack">
            <Field label={t('common.name')} htmlFor="pl-tune-model">
              <Select id="pl-tune-model" value={modelId} onChange={(event) => setModelId(event.target.value)}>
                <option value="">—</option>
                {modelOptions}
              </Select>
            </Field>
            <Field label={t('pipeline.ngl')} htmlFor="pl-ngl" hint="comma separated">
              <Input id="pl-ngl" value={ngl} onChange={(event) => setNgl(event.target.value)} />
            </Field>
            <Button icon="chart" disabled={!modelId} onClick={() => void run(t('pipeline.sweep'), () => api.tuneVulkan(modelId, ngl.split(',').map((v) => Number(v.trim())).filter((v) => !Number.isNaN(v)), 2048))}>
              {t('pipeline.sweep')}
            </Button>
          </div>
        </Card>

        <Card>
          <CardHeader
            title={t('pipeline.logs')}
            icon="database"
            actions={<Button size="sm" variant="ghost" icon="refresh" onClick={() => sources.reload()}>{t('pipeline.scan')}</Button>}
          />
          {sources.loading ? (
            <Spinner />
          ) : (
            <div className="stack">
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>{t('data.provider')}</th>
                      <th className="num">{t('pipeline.files')}</th>
                      <th className="num">{t('common.size')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(sources.data?.sources ?? []).map((source) => (
                      <tr key={String(source.provider)}>
                        <td>
                          <div className="row" style={{ gap: 6 }}>
                            <span className={`dot ${source.exists ? 'ok' : ''}`} />
                            <span className="small">{String(source.label)}</span>
                          </div>
                        </td>
                        <td className="num">{String(source.files ?? 0)}</td>
                        <td className="num">{formatBytes(Number(source.bytes ?? 0))}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Button
                variant="primary"
                icon="upload"
                onClick={() => void run(t('pipeline.import_all'), () => api.importSources(['claude_code', 'codex', 'antigravity', 'opencode'], 'needs_review'))}
              >
                {t('pipeline.import_all')}
              </Button>
              <Button
                icon="shield"
                onClick={() => void run(t('pipeline.clean'), () => api.cleanSources(['claude_code', 'codex', 'antigravity', 'opencode'], 'confirmed'))}
              >
                {t('pipeline.clean')}
              </Button>
              <span className="tiny dim">{t('pipeline.clean_note')}</span>
            </div>
          )}
        </Card>
      </div>

      <Card>
        <CardHeader title={t('pipeline.training')} icon="graduation" />
        <div className="grid cols-2">
          <Field label={t('training.dataset')} htmlFor="pl-dataset">
            <Select id="pl-dataset" value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>
              <option value="">—</option>
              {(datasets.data?.datasets ?? []).filter((item) => item.status === 'frozen' && item.rights_status === 'confirmed').map((item) => (
                <option key={item.id} value={item.id}>{item.name} ({item.record_count})</option>
              ))}
            </Select>
          </Field>
          <Field label={t('training.model_path')} htmlFor="pl-model-path">
            <Input id="pl-model-path" value={modelPath} onChange={(event) => setModelPath(event.target.value)} placeholder="E:\\data\\models\\Qwen\\Qwen3-0.6B" />
          </Field>
          <Field label={t('pipeline.algorithm')} htmlFor="pl-algo">
            <Select id="pl-algo" value={algorithm} onChange={(event) => setAlgorithm(event.target.value)}>
              <option value="auto">{t('pipeline.auto')}</option>
              <option value="int4">{t('pipeline.int4')}</option>
              <option value="dpo">{t('pipeline.dpo')}</option>
              <option value="distill">{t('pipeline.distill')}</option>
            </Select>
          </Field>
          <Field label={t('pipeline.teacher')} htmlFor="pl-teacher">
            <Select id="pl-teacher" value={teacher} onChange={(event) => setTeacher(event.target.value)}>
              <option value="local">local</option>
              <option value="gateway">vercel-ai-gateway</option>
            </Select>
          </Field>
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <Field label={`${t('training.max_steps')} — ${steps}`} htmlFor="pl-steps">
            <input id="pl-steps" type="range" min={1} max={200} value={steps} onChange={(event) => setSteps(Number(event.target.value))} />
          </Field>
          <Button
            variant="primary"
            icon="play"
            disabled={!datasetId || !modelPath}
            onClick={() =>
              void run(t('pipeline.start'), () =>
                api.trainingAuto({ dataset_id: datasetId, model_path: modelPath, max_steps: steps, algorithm, teacher }),
              )
            }
          >
            {t('pipeline.start')}
          </Button>
          <Button
            icon="bolt"
            disabled={!datasetId || !modelPath}
            onClick={() =>
              void run(t('pipeline.full_train'), () =>
                api.trainingUnified({ dataset_id: datasetId, model_path: modelPath, max_steps: steps, quant: method }),
              )
            }
          >
            {t('pipeline.full_train')}
          </Button>
        </div>
      </Card>

      <Card>
        <CardHeader title={t('pipeline.stages')} icon="layers" />
        <div className="pill-row" style={{ marginBottom: 10 }}>
          {STAGES.map((stage) => (
            <label key={stage} className="check" style={{ border: '1px solid var(--border)', borderRadius: 8, padding: '4px 8px' }}>
              <input
                type="checkbox"
                checked={stages.includes(stage)}
                onChange={(event) => setStages((prev) => (event.target.checked ? [...prev, stage] : prev.filter((item) => item !== stage)))}
              />
              {t(`pipeline.stage.${stage}`)}
            </label>
          ))}
        </div>
        <Button
          variant="primary"
          icon="play"
          disabled={stages.length === 0}
          onClick={() =>
            void run(t('pipeline.run_pipeline'), () =>
              api.pipelineRun(stages, {
                dataset_id: datasetId || undefined,
                model_path: modelPath || undefined,
                max_steps: steps,
                quant: method,
                experts,
                top_k: topK,
              }),
            )
          }
        >
          {t('pipeline.run_pipeline')}
        </Button>
      </Card>

      <Card>
        <CardHeader title={t('pipeline.playbook')} icon="history" />
        {!playbook.data || playbook.data.entries.length === 0 ? (
          <EmptyState title={t('pipeline.no_playbook')} icon="history" />
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>{t('common.created')}</th>
                  <th>LR</th>
                  <th>Rank</th>
                  <th className="num">Best eval</th>
                  <th className="num">{t('training.steps')}</th>
                </tr>
              </thead>
              <tbody>
                {playbook.data.entries.slice(0, 10).map((entry, index) => {
                  const config = (entry.config ?? {}) as Record<string, unknown>
                  return (
                    <tr key={index}>
                      <td className="tiny dim">{String(entry.created_at ?? '')}</td>
                      <td className="mono tiny">{String(config.learning_rate ?? '—')}</td>
                      <td className="mono tiny">{String(config.lora_r ?? '—')}</td>
                      <td className="num">{entry.best_eval_loss !== null ? Number(entry.best_eval_loss).toFixed(4) : '—'}</td>
                      <td className="num">{String(entry.steps ?? '—')}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Alert tone="info">{t('pipeline.note_4b')}</Alert>
    </>
  )
}
