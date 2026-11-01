import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'

import { useApp } from '../context/AppContext'
import { api } from '../lib/api'
import { Alert, Button, Field, Input } from '../components/ui'
import { Icon } from '../components/icons'

export function Onboarding() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { connect, connecting, paired, profile, scanMachine, notify } = useApp()
  const [code, setCode] = useState('')
  const [backendUp, setBackendUp] = useState<boolean | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [scanning, setScanning] = useState(false)

  useEffect(() => {
    api
      .health()
      .then(() => setBackendUp(true))
      .catch(() => setBackendUp(false))
  }, [])

  const steps = [
    { key: 'step1', done: paired },
    { key: 'step2', done: Boolean(profile) },
    { key: 'step3', done: false },
    { key: 'step4', done: false },
  ]

  return (
    <div className="onboarding">
      <div className="onboarding-card">
        <div className="row" style={{ gap: 14 }}>
          <div className="brand-mark" style={{ width: 44, height: 44, fontSize: 20 }} aria-hidden="true">H</div>
          <div>
            <h1 style={{ fontSize: 24 }}>{t('onboard.title')}</h1>
            <p className="muted" style={{ marginTop: 4 }}>{t('onboard.subtitle')}</p>
          </div>
        </div>

        <div style={{ marginTop: 22 }}>
          <Alert tone={backendUp === false ? 'danger' : 'info'} title={backendUp === false ? t('onboard.backend_unreachable') : t('onboard.local_first')}>
            {backendUp === false ? t('onboard.pair_failed') : t('onboard.local_first_body')}
          </Alert>
        </div>

        <div className="onboarding-steps">
          {steps.map((step, index) => (
            <div key={step.key} className={`onboarding-step ${step.done ? 'done' : ''} ${!step.done && (index === 0 || steps[index - 1]?.done) ? 'active' : ''}`}>
              <div className="num">{step.done ? <Icon name="check" size={13} /> : index + 1}</div>
              <div className="grow">
                <div style={{ fontWeight: 560 }}>{t(`onboard.${step.key}`)}</div>
                {index === 0 && !step.done ? (
                  <div className="stack" style={{ marginTop: 10, maxWidth: 460 }}>
                    <Field label={t('onboard.code_label')} hint={t('onboard.code_hint')} htmlFor="pair-code">
                      <Input
                        id="pair-code"
                        value={code}
                        onChange={(event) => setCode(event.target.value)}
                        placeholder="123456"
                        autoComplete="off"
                        inputMode="numeric"
                      />
                    </Field>
                    <div>
                      <Button
                        variant="primary"
                        loading={connecting}
                        onClick={async () => {
                          setError(null)
                          try {
                            await connect(code.trim() || undefined)
                            notify('ok', t('common.connected'))
                          } catch (err) {
                            setError(err instanceof Error ? err.message : String(err))
                          }
                        }}
                      >
                        {connecting ? t('onboard.pairing') : t('onboard.pair')}
                      </Button>
                    </div>
                    {error ? <Alert tone="danger">{error}</Alert> : null}
                  </div>
                ) : null}
                {index === 1 && step.done === false && paired ? (
                  <div style={{ marginTop: 10 }}>
                    <Button
                      variant="primary"
                      loading={scanning}
                      onClick={async () => {
                        setScanning(true)
                        try {
                          await scanMachine()
                        } catch (err) {
                          notify('danger', err instanceof Error ? err.message : String(err))
                        } finally {
                          setScanning(false)
                        }
                      }}
                    >
                      {scanning ? t('onboard.scanning') : t('onboard.scan')}
                    </Button>
                  </div>
                ) : null}
                {index >= 2 && paired ? (
                  <div style={{ marginTop: 8 }}>
                    <Button
                      variant="ghost"
                      onClick={() => navigate(index === 2 ? '/models' : '/playground')}
                    >
                      {index === 2 ? t('nav.models') : t('nav.playground')}
                    </Button>
                  </div>
                ) : null}
              </div>
            </div>
          ))}
        </div>

        <p className="tiny dim" style={{ marginTop: 18 }}>
          {t('onboard.privacy_note')}
        </p>
        {paired ? (
          <div className="row end" style={{ marginTop: 10 }}>
            <Button variant="primary" onClick={() => navigate('/')}>
              {t('onboard.open')}
            </Button>
          </div>
        ) : null}
      </div>
    </div>
  )
}
