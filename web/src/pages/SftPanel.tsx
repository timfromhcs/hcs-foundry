import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'

import { useApp } from '../context/AppContext'
import { api } from '../lib/api'
import { useAsync, useJob } from '../lib/hooks'
import { formatBytes, statusTone } from '../lib/format'
import type { TrainingEnvironment } from '../lib/types'
import { Alert, Badge, Button, Card, CardHeader, Field, Input, KV, Progress, Select } from '../components/ui'

export function SftPanel({
  frozen,
  environment,
  notify,
  onFinished,
}: {
  frozen: Array<{ id: string; name: string; record_count: number }>
  environment: TrainingEnvironment | null
  notify: (kind: 'ok' | 'danger' | 'warn' | 'info', message: string) => void
  onFinished: () => void
}) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { startJob } = useApp()
  const recipes = useAsync(() => api.trainingRecipes(), [])
  const [datasetId, setDatasetId] = useState('')
  const [modelPath, setModelPath] = useState('')
  const [recipeId, setRecipeId] = useState('tiny-proof')
  const [maxSteps, setMaxSteps] = useState(20)
  const [seqLen, setSeqLen] = useState(128)
  const [projection, setProjection] = useState<Record<string, unknown> | null>(null)
  const [projecting, setProjecting] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const job = useJob(jobId)

  useEffect(() => {
    const recipe = (recipes.data?.recipes ?? []).find((item) => item.id === recipeId)
    const config = (recipe?.config ?? {}) as Record<string, number>
    if (config.max_steps) setMaxSteps(config.max_steps)
    if (config.max_seq_len) setSeqLen(config.max_seq_len)
  }, [recipeId, recipes.data])

  useEffect(() => {
    if (job.job?.status === 'succeeded') {
      notify('ok', t('training.sft_done'))
      onFinished()
      setJobId(null)
    }
    if (job.job?.status === 'failed') {
      notify('danger', job.job.error ?? t('training.sft_failed'))
      setJobId(null)
    }
  }, [job.job?.status, job.job?.error, notify, onFinished, t])

  const payload = {
    dataset_id: datasetId,
    model_path: modelPath,
    max_steps: maxSteps,
    max_seq_len: seqLen,
  }

  const selectedRecipe = (recipes.data?.recipes ?? []).find((item) => item.id === recipeId)
  const projectionInfo = (selectedRecipe?.projection ?? {}) as Record<string, unknown>
  const resources = (projection?.resources ?? {}) as Record<string, unknown>
  const output = (job.job?.output ?? {}) as Record<string, unknown>
  const manifest = (output.manifest ?? null) as Record<string, unknown> | null

  return (
    <Card accent>
      <CardHeader
        title={t('training.sft_title')}
        icon="graduation"
        subtitle={t('training.model_path_hint')}
        actions={
          <Badge tone={environment?.can_finetune ? 'ok' : 'warn'}>
            {environment?.can_finetune ? t('training.can_start') : t('training.requires_stack')}
          </Badge>
        }
      />
      <div className="stack">
        <div className="grid cols-2">
          <Field label={t('training.dataset')} htmlFor="sft-dataset" hint={frozen.length === 0 ? t('training.no_dataset') : undefined}>
            <Select id="sft-dataset" value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>
              <option value="">—</option>
              {frozen.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name} ({item.record_count})
                </option>
              ))}
            </Select>
          </Field>
          <Field label={t('training.recipe')} htmlFor="sft-recipe">
            <Select id="sft-recipe" value={recipeId} onChange={(event) => setRecipeId(event.target.value)}>
              {(recipes.data?.recipes ?? []).map((recipe) => (
                <option key={String(recipe.id)} value={String(recipe.id)}>
                  {String(recipe.name)}
                </option>
              ))}
            </Select>
          </Field>
        </div>

        <Field label={t('training.model_path')} hint={t('training.model_path_hint')} htmlFor="sft-model">
          <Input
            id="sft-model"
            value={modelPath}
            onChange={(event) => setModelPath(event.target.value)}
            placeholder="E:\\models\\Qwen3-0.6B"
          />
        </Field>

        <div className="grid cols-2">
          <Field label={`${t('training.max_steps')} — ${maxSteps}`} htmlFor="sft-steps">
            <input
              id="sft-steps"
              type="range"
              min={1}
              max={2000}
              step={1}
              value={maxSteps}
              onChange={(event) => setMaxSteps(Number(event.target.value))}
            />
          </Field>
          <Field label={`${t('training.seq_len')} — ${seqLen}`} htmlFor="sft-seq">
            <input
              id="sft-seq"
              type="range"
              min={32}
              max={2048}
              step={32}
              value={seqLen}
              onChange={(event) => setSeqLen(Number(event.target.value))}
            />
          </Field>
        </div>

        <div className="row">
          <Button
            icon="chart"
            loading={projecting}
            disabled={!modelPath || !datasetId}
            onClick={async () => {
              setProjecting(true)
              try {
                const data = await api.trainingProject(payload)
                setProjection(data as unknown as Record<string, unknown>)
              } catch (error) {
                notify('danger', error instanceof Error ? error.message : String(error))
              } finally {
                setProjecting(false)
              }
            }}
          >
            {t('common.estimate')}
          </Button>
          <Button
            variant="primary"
            icon="play"
            loading={Boolean(jobId)}
            disabled={!environment?.can_finetune || !modelPath || !datasetId || Boolean(jobId)}
            onClick={async () => {
              try {
                const id = await startJob(() => api.trainingSft(payload), t('training.sft_running'))
                setJobId(id)
              } catch (error) {
                notify('danger', error instanceof Error ? error.message : String(error))
              }
            }}
          >
            {t('training.start_sft')}
          </Button>
          <Button variant="ghost" icon="download" onClick={() => navigate('/hub')}>
            {t('training.pull_model')}
          </Button>
        </div>

        {projection ? (
          <div className="card tight" style={{ background: 'var(--surface-2)' }}>
            <KV
              items={[
                [
                  t('training.memory'),
                  `${formatBytes(Number(resources.estimated_low_bytes ?? 0))} – ${formatBytes(Number(resources.estimated_high_bytes ?? 0))}`,
                ],
                [t('training.projection'), `${String(projectionInfo.hours ?? '—')} h (${String(projectionInfo.confidence ?? '—')})`],
                [t('common.why'), String(projectionInfo.basis ?? t('training.projection_hint'))],
              ]}
            />
          </div>
        ) : null}

        {selectedRecipe?.warning ? <Alert tone="warn">{String(selectedRecipe.warning)}</Alert> : null}

        {job.job ? (
          <div className="stack" style={{ gap: 6 }}>
            <Progress value={job.job.progress} indeterminate={job.job.progress === 0} />
            <div className="row between">
              <span className="tiny dim truncate">{job.job.message}</span>
              <Badge tone={statusTone(job.job.status)}>{t(`common.${job.job.status}`)}</Badge>
            </div>
            {job.events.length > 0 ? (
              <div className="mono-block" style={{ maxHeight: 160 }}>
                {job.events
                  .slice(-12)
                  .map((event) => `${event.ts} ${event.message}`)
                  .join('\n')}
              </div>
            ) : null}
          </div>
        ) : null}

        {manifest ? (
          <KV
            items={[
              [t('training.steps'), String(manifest.steps ?? '—')],
              [t('training.loss_first'), String(manifest.loss_first ?? '—')],
              [t('training.loss_last'), String(manifest.loss_last ?? '—')],
              [t('training.adapter'), <span className="mono tiny">{String(manifest.adapter_dir ?? '—')}</span>],
            ]}
          />
        ) : null}
      </div>
    </Card>
  )
}
