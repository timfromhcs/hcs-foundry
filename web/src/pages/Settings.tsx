import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { useApp } from '../context/AppContext'
import { api } from '../lib/api'
import { useAsync } from '../lib/hooks'
import { formatBytes, formatDate } from '../lib/format'
import { Alert, Badge, Button, Card, CardHeader, Field, Input, KV, Select, Spinner, Textarea, Toggle } from '../components/ui'
import { LANGUAGES } from '../i18n'

export function Settings() {
  const { t } = useTranslation()
  const { notify, lang, setLang, theme, setTheme, info, refreshInfo } = useApp()
  const settings = useAsync(() => api.settings(), [])
  const tokens = useAsync(() => api.request<{ tokens: Array<{ id: string; label: string; created_at: string; revoked: boolean }> }>('/system/tokens'), [])
  const [modelRoots, setModelRoots] = useState('')
  const [importRoots, setImportRoots] = useState('')
  const [reserve, setReserve] = useState(15)
  const [headroom, setHeadroom] = useState(1)
  const [concurrent, setConcurrent] = useState(false)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    const effective = settings.data?.effective as Record<string, unknown> | undefined
    if (!effective) return
    setModelRoots(((effective.model_roots as string[] | undefined) ?? []).join('\n'))
    setImportRoots(((effective.import_roots as string[] | undefined) ?? []).join('\n'))
    setReserve(Math.round(Number(effective.os_reserve_ratio ?? 0.15) * 100))
    setHeadroom(Math.round(Number(effective.gpu_headroom_bytes ?? 1073741824) / 1024 ** 3))
    setConcurrent(Boolean(effective.allow_concurrent_heavy_jobs))
  }, [settings.data])

  return (
    <>
      <div>
        <p className="muted small">{t('settings.subtitle')}</p>
      </div>

      <div className="grid cols-2">
        <Card>
          <CardHeader title={t('settings.diagnostics')} icon="cpu" />
          {settings.loading ? (
            <Spinner />
          ) : (
            <div className="stack">
              <Field label={t('settings.model_roots')} hint="One folder per line">
                <Textarea value={modelRoots} onChange={(event) => setModelRoots(event.target.value)} />
              </Field>
              <Field label={t('settings.import_roots')} hint="One folder per line. Imports are only accepted from these roots.">
                <Textarea value={importRoots} onChange={(event) => setImportRoots(event.target.value)} />
              </Field>
              <div className="grid cols-2">
                <Field label={`${t('settings.reserve')} — ${reserve}%`} htmlFor="reserve">
                  <input id="reserve" type="range" min={5} max={50} step={1} value={reserve} onChange={(event) => setReserve(Number(event.target.value))} />
                </Field>
                <Field label={`${t('settings.headroom')} — ${headroom} GiB`} htmlFor="headroom">
                  <Input id="headroom" type="number" min={0} max={16} value={headroom} onChange={(event) => setHeadroom(Number(event.target.value))} />
                </Field>
              </div>
              <Toggle
                checked={concurrent}
                onChange={setConcurrent}
                label={t('settings.concurrent')}
                description={t('machine.concurrent_hint')}
              />
              <Toggle checked={false} onChange={() => notify('warn', t('settings.telemetry_note'))} label={t('settings.telemetry')} description={t('settings.telemetry_note')} disabled />
              <Button
                variant="primary"
                loading={saving}
                onClick={async () => {
                  setSaving(true)
                  try {
                    await api.updateSettings({
                      model_roots: modelRoots.split('\n').map((value) => value.trim()).filter(Boolean),
                      import_roots: importRoots.split('\n').map((value) => value.trim()).filter(Boolean),
                      os_reserve_ratio: reserve / 100,
                      gpu_headroom_bytes: headroom * 1024 ** 3,
                      allow_concurrent_heavy_jobs: concurrent,
                    })
                    settings.reload()
                    notify('ok', t('settings.saved'))
                  } catch (error) {
                    notify('danger', error instanceof Error ? error.message : String(error))
                  } finally {
                    setSaving(false)
                  }
                }}
              >
                {t('settings.save')}
              </Button>
              <span className="tiny dim">{t('settings.restart_required')}</span>
            </div>
          )}
        </Card>

        <div className="stack">
          <Card>
            <CardHeader title={t('settings.language')} icon="globe" />
            <div className="grid cols-2">
              <Field label={t('settings.language')} htmlFor="settings-lang">
                <Select id="settings-lang" value={lang} onChange={(event) => setLang(event.target.value as typeof lang)}>
                  {LANGUAGES.map((language) => (
                    <option key={language.code} value={language.code}>{language.label}</option>
                  ))}
                </Select>
              </Field>
              <Field label={t('settings.theme')} htmlFor="settings-theme">
                <Select id="settings-theme" value={theme} onChange={(event) => setTheme(event.target.value as typeof theme)}>
                  <option value="dark">{t('common.dark')}</option>
                  <option value="light">{t('common.light')}</option>
                </Select>
              </Field>
            </div>
          </Card>

          <Card>
            <CardHeader title={t('settings.security')} icon="shield" />
            <div className="stack">
              <KV
                items={[
                  [t('settings.paired_tokens'), <Badge tone="ok">{tokens.data?.tokens.filter((token) => !token.revoked).length ?? 0}</Badge>],
                  [t('settings.local_only'), <Badge tone="accent">{t('common.yes')}</Badge>],
                  [t('settings.telemetry'), <Badge tone="ok">{t('common.no')}</Badge>],
                ]}
              />
              {tokens.data?.tokens.length ? (
                <div className="stack" style={{ gap: 4 }}>
                  {tokens.data.tokens.slice(-4).map((token) => (
                    <div key={token.id} className="row between">
                      <span className="tiny mono">{token.label}</span>
                      <span className="tiny dim">{formatDate(token.created_at)}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <span className="small dim">{t('settings.no_tokens')}</span>
              )}
              <Button
                variant="danger"
                icon="shield"
                onClick={async () => {
                  if (!window.confirm(t('settings.revoke_confirm'))) return
                  try {
                    await api.revokeTokens()
                    tokens.reload()
                    notify('ok', t('settings.revoked'))
                  } catch (error) {
                    notify('danger', error instanceof Error ? error.message : String(error))
                  }
                }}
              >
                {t('settings.revoke')}
              </Button>
            </div>
          </Card>

          <Card>
            <CardHeader title={t('settings.about')} icon="box" />
            <KV
              items={[
                [t('settings.version'), info?.version ?? '—'],
                [t('settings.data_dir'), <span className="mono tiny">{info?.data_dir ?? '—'}</span>],
                ['Schema', String(info?.schema_version ?? '—')],
                [t('settings.license'), t('settings.license_value')],
              ]}
            />
            <div style={{ marginTop: 10 }}>
              <Alert tone="info" title={t('settings.privacy')}>{t('settings.privacy_text')}</Alert>
            </div>
          </Card>

          <Card>
            <CardHeader title={t('settings.reset')} icon="refresh" />
            <div className="row">
              <Button
                variant="ghost"
                onClick={async () => {
                  await refreshInfo()
                  settings.reload()
                  notify('ok', t('common.refresh'))
                }}
              >
                {t('common.refresh')}
              </Button>
              <Button
                variant="ghost"
                onClick={() => {
                  setReserve(15)
                  setHeadroom(1)
                  setConcurrent(false)
                }}
              >
                {t('settings.reset')}
              </Button>
            </div>
          </Card>
        </div>
      </div>

      {settings.data ? (
        <Card>
          <CardHeader title={t('settings.effective')} icon="terminal" />
          <div className="mono-block" style={{ maxHeight: 320 }}>
            {JSON.stringify(settings.data.effective, null, 2)}
          </div>
        </Card>
      ) : null}

      <Card>
        <CardHeader title={t('settings.diagnostics')} icon="history" />
        <div className="row">
          <span className="tiny dim">{t('overview.safe_budget')}: {formatBytes(Number((settings.data?.effective as Record<string, unknown> | undefined)?.gpu_headroom_bytes ?? 0))} headroom</span>
        </div>
      </Card>
    </>
  )
}
