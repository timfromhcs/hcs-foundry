import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { useApp } from '../context/AppContext'
import { api } from '../lib/api'
import { useAsync } from '../lib/hooks'
import { formatDate, statusTone } from '../lib/format'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardHeader,
  EmptyState,
  Field,
  Input,
  Select,
  Spinner,
  Textarea,
} from '../components/ui'

type Card = Record<string, unknown> & {
  name?: string
  purpose?: string
  explicit_facts?: string[]
  generated_proposals?: string[]
  voice?: Record<string, unknown>
  values?: string[]
  competence?: string[]
  boundaries?: string[]
  unknowns?: string[]
  languages?: string[]
  approved?: boolean
}

export function IdentityMaker() {
  const { t } = useTranslation()
  const { notify } = useApp()
  const profiles = useAsync(() => api.identityList(), [])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [profile, setProfile] = useState<Record<string, unknown> | null>(null)
  const [prompt, setPrompt] = useState('')
  const [name, setName] = useState('')
  const [languages, setLanguages] = useState('en')
  const [busy, setBusy] = useState(false)
  const [editing, setEditing] = useState(false)
  const [draftCard, setDraftCard] = useState<Card | null>(null)
  const [evalResult, setEvalResult] = useState<Record<string, unknown> | null>(null)

  useEffect(() => {
    if (!selectedId && profiles.data?.profiles.length) setSelectedId(profiles.data.profiles[0].id)
  }, [profiles.data, selectedId])

  useEffect(() => {
    if (!selectedId) {
      setProfile(null)
      return
    }
    api
      .identityGet(selectedId)
      .then((data) => setProfile(data.profile))
      .catch(() => setProfile(null))
  }, [selectedId])

  const card = (profile?.card ?? null) as Card | null
  const status = String(profile?.status ?? '')
  const approved = Boolean(card?.approved)

  const reloadProfile = async (id: string) => {
    const data = await api.identityGet(id)
    setProfile(data.profile)
    profiles.reload()
  }

  return (
    <>
      <div className="row between">
        <div>
          <p className="muted small">{t('identity.subtitle')}</p>
        </div>
      </div>

      <div className="grid sidebar-right">
        <div className="stack">
          <Card accent>
            <CardHeader title={t('identity.prompt')} icon="fingerprint" />
            <div className="stack">
              <Field label={t('identity.prompt')} hint={t('identity.prompt_hint')} htmlFor="identity-prompt">
                <Textarea
                  id="identity-prompt"
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  placeholder="You are Mara, a calm senior engineer who explains trade-offs honestly, prefers short answers, and never invents facts."
                  style={{ minHeight: 120 }}
                />
              </Field>
              <div className="grid cols-2">
                <Field label={t('identity.name')} htmlFor="identity-name">
                  <Input id="identity-name" value={name} onChange={(event) => setName(event.target.value)} placeholder="Mara" />
                </Field>
                <Field label={t('identity.languages')} htmlFor="identity-langs">
                  <Select id="identity-langs" value={languages} onChange={(event) => setLanguages(event.target.value)}>
                    <option value="en">English</option>
                    <option value="de">Deutsch</option>
                    <option value="fr">Français</option>
                    <option value="en,de">English + Deutsch</option>
                    <option value="en,de,fr">English + Deutsch + Français</option>
                  </Select>
                </Field>
              </div>
              <Button
                variant="primary"
                loading={busy}
                disabled={prompt.trim().length < 12}
                onClick={async () => {
                  setBusy(true)
                  try {
                    const data = await api.identityDraft(prompt, name || undefined, languages.split(','))
                    setSelectedId(data.id)
                    await reloadProfile(data.id)
                    setPrompt('')
                    notify('ok', t('identity.card'))
                  } catch (error) {
                    notify('danger', error instanceof Error ? error.message : String(error))
                  } finally {
                    setBusy(false)
                  }
                }}
              >
                {busy ? t('identity.creating') : t('identity.create')}
              </Button>
            </div>
          </Card>

          {profile && card ? (
            <>
              <Card>
                <CardHeader
                  title={card.name ?? t('identity.card')}
                  subtitle={card.purpose}
                  icon="fingerprint"
                  actions={
                    <>
                      <Badge tone={approved ? 'ok' : statusTone(status)}>{approved ? t('identity.approved') : status}</Badge>
                      {!approved ? (
                        <Button size="sm" variant="ghost" icon="edit" onClick={() => { setDraftCard(card); setEditing(true) }}>
                          {t('identity.edit')}
                        </Button>
                      ) : null}
                    </>
                  }
                />

                {editing && draftCard ? (
                  <div className="stack">
                    <Field label={t('identity.name')} htmlFor="edit-name">
                      <Input id="edit-name" value={draftCard.name ?? ''} onChange={(event) => setDraftCard({ ...draftCard, name: event.target.value })} />
                    </Field>
                    <Field label="Purpose" htmlFor="edit-purpose">
                      <Textarea id="edit-purpose" value={draftCard.purpose ?? ''} onChange={(event) => setDraftCard({ ...draftCard, purpose: event.target.value })} />
                    </Field>
                    <Field label={t('identity.boundaries')} hint="One per line">
                      <Textarea
                        value={(draftCard.boundaries ?? []).join('\n')}
                        onChange={(event) => setDraftCard({ ...draftCard, boundaries: event.target.value.split('\n').map((v) => v.trim()).filter(Boolean) })}
                      />
                    </Field>
                    <div className="row end">
                      <Button variant="ghost" onClick={() => setEditing(false)}>{t('common.cancel')}</Button>
                      <Button
                        variant="primary"
                        onClick={async () => {
                          if (!selectedId) return
                          try {
                            await api.identityPatch(selectedId, {
                              name: draftCard.name,
                              purpose: draftCard.purpose,
                              boundaries: draftCard.boundaries,
                            })
                            await reloadProfile(selectedId)
                            setEditing(false)
                            notify('ok', t('identity.save'))
                          } catch (error) {
                            notify('danger', error instanceof Error ? error.message : String(error))
                          }
                        }}
                      >
                        {t('identity.save')}
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="stack">
                    <div>
                      <div className="label" style={{ marginBottom: 6 }}>{t('identity.facts')}</div>
                      {card.explicit_facts?.length ? (
                        <ul className="small muted" style={{ margin: 0, paddingLeft: 18 }}>
                          {card.explicit_facts.map((fact, index) => <li key={index}>{fact}</li>)}
                        </ul>
                      ) : (
                        <span className="small dim">{t('common.none')}</span>
                      )}
                    </div>
                    <div>
                      <div className="label" style={{ marginBottom: 6 }}>{t('identity.proposals')}</div>
                      <div className="stack" style={{ gap: 4 }}>
                        {(card.generated_proposals ?? []).map((proposal, index) => (
                          <div key={index} className="row" style={{ gap: 6 }}>
                            <Badge tone="accent">AI</Badge>
                            <span className="small">{proposal}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div className="grid cols-2">
                      <div>
                        <div className="label" style={{ marginBottom: 6 }}>{t('identity.values')}</div>
                        <div className="pill-row">{(card.values ?? []).map((value) => <Badge key={value} tone="plain">{value}</Badge>)}</div>
                      </div>
                      <div>
                        <div className="label" style={{ marginBottom: 6 }}>{t('identity.languages')}</div>
                        <div className="pill-row">{(card.languages ?? []).map((lang) => <Badge key={lang} tone="info">{lang}</Badge>)}</div>
                      </div>
                    </div>
                    <div>
                      <div className="label" style={{ marginBottom: 6 }}>{t('identity.boundaries')}</div>
                      <ul className="small muted" style={{ margin: 0, paddingLeft: 18 }}>
                        {(card.boundaries ?? []).map((rule, index) => <li key={index}>{rule}</li>)}
                      </ul>
                    </div>
                    <Alert tone="info">{t('identity.disclaimer')}</Alert>
                  </div>
                )}
              </Card>

              <Card>
                <CardHeader title={t('identity.dataset')} icon="database" />
                <div className="stack">
                  {profile.dataset_id ? (
                    <Alert tone="ok">{t('identity.dataset_created')} — <span className="mono tiny">{String(profile.dataset_id)}</span></Alert>
                  ) : null}
                  <Button
                    variant="primary"
                    icon="database"
                    disabled={status !== 'approved' || Boolean(profile.dataset_id)}
                    onClick={async () => {
                      if (!selectedId) return
                      try {
                        const result = await api.identityDataset(selectedId)
                        await reloadProfile(selectedId)
                        notify('ok', `${result.record_count} ${t('data.record_count').toLowerCase()}`)
                      } catch (error) {
                        notify('danger', error instanceof Error ? error.message : String(error))
                      }
                    }}
                  >
                    {t('identity.create_dataset')}
                  </Button>
                  {!approved ? (
                    <Button
                      icon="check"
                      onClick={async () => {
                        if (!selectedId) return
                        try {
                          await api.identityApprove(selectedId)
                          await reloadProfile(selectedId)
                          notify('ok', t('identity.approved'))
                        } catch (error) {
                          notify('danger', error instanceof Error ? error.message : String(error))
                        }
                      }}
                    >
                      {t('identity.approve')}
                    </Button>
                  ) : null}
                  <Button
                    icon="chart"
                    variant="ghost"
                    onClick={async () => {
                      if (!selectedId) return
                      try {
                        const result = await api.identityEvaluate(selectedId)
                        setEvalResult(result as unknown as Record<string, unknown>)
                      } catch (error) {
                        if (error instanceof Error && error.message.includes('Load a model')) notify('warn', t('identity.requires_model'))
                        else notify('danger', error instanceof Error ? error.message : String(error))
                      }
                    }}
                  >
                    {t('identity.evaluate')}
                  </Button>
                  {evalResult ? (
                    <div className="stack" style={{ gap: 6 }}>
                      <div className="row between">
                        <span className="small">{t('identity.score')}</span>
                        <Badge tone={Number(evalResult.score) >= 0.75 ? 'ok' : 'warn'}>{Math.round(Number(evalResult.score ?? 0) * 100)}%</Badge>
                      </div>
                      {(evalResult.cases as Array<Record<string, unknown>> | undefined)?.map((item, index) => (
                        <div key={index} className="row between">
                          <span className="tiny">{String(item.id)}</span>
                          <Badge tone={item.status === 'pass' ? 'ok' : item.status === 'fail' ? 'danger' : 'warn'}>{String(item.status)}</Badge>
                        </div>
                      ))}
                    </div>
                  ) : null}
                  <span className="tiny dim">{t('identity.generated_notice')}</span>
                </div>
              </Card>
            </>
          ) : (
            <Card>
              <EmptyState title={t('identity.empty')} icon="fingerprint">{t('identity.select')}</EmptyState>
            </Card>
          )}
        </div>

        <Card flush>
          <div style={{ padding: 14 }}>
            <CardHeader title={t('identity.profiles')} icon="history" />
          </div>
          {profiles.loading ? (
            <div style={{ padding: 14 }}><Spinner /></div>
          ) : !profiles.data || profiles.data.profiles.length === 0 ? (
            <div style={{ padding: 14 }}><span className="small dim">{t('identity.empty')}</span></div>
          ) : (
            <div className="stack" style={{ gap: 0 }}>
              {profiles.data.profiles.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className="row between"
                  style={{
                    width: '100%',
                    background: selectedId === item.id ? 'var(--surface-2)' : 'transparent',
                    border: 'none',
                    borderTop: '1px solid var(--border)',
                    padding: '10px 14px',
                    color: 'inherit',
                    cursor: 'pointer',
                    textAlign: 'left',
                  }}
                  onClick={() => setSelectedId(item.id)}
                >
                  <span>
                    <span className="small" style={{ display: 'block', fontWeight: 560 }}>{item.name}</span>
                    <span className="tiny dim">{formatDate(item.created_at)}</span>
                  </span>
                  <Badge tone={item.status === 'approved' ? 'ok' : 'plain'}>{item.status}</Badge>
                </button>
              ))}
            </div>
          )}
        </Card>
      </div>
    </>
  )
}
