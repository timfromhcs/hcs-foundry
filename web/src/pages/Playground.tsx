import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'

import { useApp } from '../context/AppContext'
import { api } from '../lib/api'
import { formatNumber } from '../lib/format'
import { Alert, Badge, Button, Card, EmptyState, Field, Select, Textarea } from '../components/ui'

interface Message {
  role: 'user' | 'assistant'
  content: string
  reasoning?: string
}

const PRESETS: Array<{ id: string; prompt: string }> = [
  { id: 'explain', prompt: 'Explain in three short paragraphs how quantized GGUF inference works on an integrated GPU.' },
  { id: 'code', prompt: 'Write a Python function that merges overlapping intervals and include three unit tests.' },
  { id: 'translate', prompt: 'Translate the following into German and French, preserving technical terms:\n\nThe model runs locally.' },
  { id: 'summarize', prompt: 'Summarize the trade-offs between weight quantization and KV-cache quantization in five bullets.' },
  { id: 'json', prompt: 'Return only a JSON object with the keys "status" (string) and "items" (array of three strings).' },
]

export function Playground() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { notify, refreshInfo } = useApp()
  const [status, setStatus] = useState<Record<string, unknown> | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [systemPrompt, setSystemPrompt] = useState('You are a precise local assistant. If you do not know something, say so.')
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(512)
  const [busy, setBusy] = useState(false)
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null)
  const [showSettings, setShowSettings] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api.inferenceStatus().then(setStatus).catch(() => setStatus(null))
  }, [])

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  const ready = status?.status === 'ready'
  const model = status?.model as Record<string, unknown> | undefined

  const send = async () => {
    const text = input.trim()
    if (!text || busy || !ready) return
    const history: Message[] = [...messages, { role: 'user', content: text }]
    setMessages([...history, { role: 'assistant', content: '', reasoning: '' }])
    setInput('')
    setBusy(true)
    setMetrics(null)
    const controller = new AbortController()
    abortRef.current = controller
    let answer = ''
    let reasoning = ''
    try {
      await api.chat(
        [{ role: 'system', content: systemPrompt }, ...history],
        { temperature, max_tokens: maxTokens },
        (event) => {
          if (event.type === 'delta' && typeof event.text === 'string') {
            answer += event.text
            setMessages((prev) => {
              const next = [...prev]
              next[next.length - 1] = { role: 'assistant', content: answer, reasoning }
              return next
            })
          } else if (event.type === 'reasoning' && typeof event.text === 'string') {
            reasoning += event.text
            setMessages((prev) => {
              const next = [...prev]
              next[next.length - 1] = { role: 'assistant', content: answer, reasoning }
              return next
            })
          } else if (event.type === 'done') {
            if (event.metrics) setMetrics(event.metrics as Record<string, unknown>)
            if (typeof event.text === 'string' && !answer) answer = event.text
            setMessages((prev) => {
              const next = [...prev]
              next[next.length - 1] = { role: 'assistant', content: answer, reasoning }
              return next
            })
          } else if (event.type === 'error') {
            notify('danger', String(event.message ?? t('common.error')))
          }
        },
        controller.signal,
      )
    } catch (error) {
      if (!controller.signal.aborted) {
        notify('danger', error instanceof Error ? error.message : String(error))
      }
    } finally {
      setBusy(false)
      abortRef.current = null
      void refreshInfo()
    }
  }

  const stop = async () => {
    abortRef.current?.abort()
    try {
      await api.cancelInference()
    } catch {
      /* stopping is best effort */
    }
    setBusy(false)
  }

  if (!ready) {
    return (
      <>
        <div>
          <p className="muted small">{t('play.subtitle')}</p>
        </div>
        <Card>
          <EmptyState
            title={t('play.no_model')}
            icon="chat"
            action={<Button variant="primary" icon="box" onClick={() => navigate('/models')}>{t('play.load_model')}</Button>}
          >
            {t('play.error_no_model')}
          </EmptyState>
        </Card>
      </>
    )
  }

  return (
    <>
      <div className="row between">
        <div>
          <p className="muted small">{t('play.subtitle')}</p>
        </div>
        <div className="row">
          <Badge tone="ok" dot>{String(model?.name ?? t('models.model_loaded'))}</Badge>
          <Badge tone="plain">{String(model?.quantization ?? '')}</Badge>
          <Button size="sm" variant="ghost" icon="settings" onClick={() => setShowSettings((value) => !value)}>
            {t('models.load_config')}
          </Button>
        </div>
      </div>

      {showSettings ? (
        <Card>
          <div className="grid cols-2">
            <Field label={t('play.system_prompt')} htmlFor="system">
              <Textarea id="system" value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value)} />
            </Field>
            <div className="stack">
              <Field label={`${t('play.temperature')} — ${temperature.toFixed(2)}`} htmlFor="temp">
                <input id="temp" type="range" min={0} max={2} step={0.05} value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} />
              </Field>
              <Field label={`${t('play.max_tokens')} — ${maxTokens}`} htmlFor="maxtok">
                <input id="maxtok" type="range" min={32} max={4096} step={32} value={maxTokens} onChange={(event) => setMaxTokens(Number(event.target.value))} />
              </Field>
              <Field label={t('play.presets')} htmlFor="preset">
                <Select
                  id="preset"
                  value=""
                  onChange={(event) => {
                    const preset = PRESETS.find((item) => item.id === event.target.value)
                    if (preset) setInput(preset.prompt)
                  }}
                >
                  <option value="">—</option>
                  {PRESETS.map((preset) => (
                    <option key={preset.id} value={preset.id}>{t(`play.preset.${preset.id}`)}</option>
                  ))}
                </Select>
              </Field>
            </div>
          </div>
        </Card>
      ) : null}

      <Card className="chat-card">
        <div className="chat">
          <div className="chat-log" ref={logRef} aria-live="polite">
            {messages.length === 0 ? (
              <EmptyState title={t('play.empty')} icon="chat">{t('play.empty_hint')}</EmptyState>
            ) : (
              messages.map((message, index) => (
                <div key={index} className={`msg ${message.role}`}>
                  <div className="avatar" aria-hidden="true">{message.role === 'user' ? 'You' : 'AI'}</div>
                  <div className="bubble">
                    {message.reasoning ? (
                      <details>
                        <summary className="tiny dim" style={{ cursor: 'pointer', marginBottom: 4 }}>{t('play.reasoning')}</summary>
                        <div className="reasoning">{message.reasoning}</div>
                      </details>
                    ) : null}
                    {message.content || (busy && index === messages.length - 1 ? <span className="dim">{t('play.thinking')}</span> : null)}
                  </div>
                </div>
              ))
            )}
          </div>

          {metrics ? (
            <div className="row" style={{ gap: 14, padding: '4px 2px' }}>
              <span className="tiny dim">{t('play.metrics')}</span>
              <span className="tiny"><strong>{formatNumber(Number(metrics.tokens_per_second ?? 0))}</strong> {t('play.tokens_per_sec')}</span>
              <span className="tiny"><strong>{formatNumber(Number(metrics.ttft_seconds ?? 0))}</strong> s {t('play.ttft').toLowerCase()}</span>
              <span className="tiny dim">{String(metrics.completion_tokens ?? 0)} {t('common.tokens')}</span>
            </div>
          ) : null}

          <div className="chat-input">
            <Textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  void send()
                }
              }}
              placeholder={t('play.input_placeholder')}
              aria-label={t('play.input_placeholder')}
            />
            {busy ? (
              <Button variant="danger" icon="stop" onClick={() => void stop()}>{t('play.stop')}</Button>
            ) : (
              <Button variant="primary" icon="play" onClick={() => void send()} disabled={!input.trim()}>{t('play.send')}</Button>
            )}
            <Button variant="ghost" icon="trash" onClick={() => { setMessages([]); setMetrics(null) }} aria-label={t('play.clear')} />
          </div>
        </div>
      </Card>

      <Alert tone="info">{t('settings.privacy_text')}</Alert>
    </>
  )
}
