import { useState } from 'react'
import { useTranslation } from 'react-i18next'

import { useApp } from '../context/AppContext'
import { api } from '../lib/api'
import { formatBytes, formatDate } from '../lib/format'
import { Alert, Badge, Button, Card, CardHeader, EmptyState, Field, Input, KV, Progress, Stat, Textarea } from '../components/ui'

function statusTone(status: string | undefined) {
  if (status === 'detected') return 'ok'
  if (status === 'not_detected') return 'plain'
  if (status === 'error') return 'danger'
  return 'warn'
}

const ACCELERATOR_LABELS: Record<string, string> = {
  vulkan: 'Vulkan',
  cuda: 'CUDA',
  rocm: 'ROCm',
  directml: 'DirectML',
  npu: 'NPU',
}

export function Machine() {
  const { t } = useTranslation()
  const { profile, budget, scanMachine, refreshMachine, notify } = useApp()
  const [scanning, setScanning] = useState(false)
  const [runtimeRoots, setRuntimeRoots] = useState('')
  const [found, setFound] = useState<Array<Record<string, unknown>>>([])
  const [manualPath, setManualPath] = useState('')
  const [busy, setBusy] = useState(false)

  const gpu = profile?.gpu?.devices?.[0]
  const browsers = getBrowserCaps()

  return (
    <>
      <div className="row between">
        <div>
          <p className="muted small">{t('machine.subtitle')}</p>
        </div>
        <div className="row">
          <Button
            icon="refresh"
            variant="primary"
            loading={scanning}
            onClick={async () => {
              setScanning(true)
              try {
                await scanMachine()
                await refreshMachine()
              } catch (error) {
                notify('danger', error instanceof Error ? error.message : String(error))
              } finally {
                setScanning(false)
              }
            }}
          >
            {scanning ? t('machine.scanning') : t('machine.scan')}
          </Button>
        </div>
      </div>

      {!profile ? (
        <Card>
          <EmptyState title={t('machine.no_profile')} icon="cpu" action={<Button variant="primary" loading={scanning} onClick={() => void scanMachine()}>{t('machine.scan')}</Button>}>
            {t('machine.no_profile_hint')}
          </EmptyState>
        </Card>
      ) : (
        <>
          <div className="grid cols-4">
            <Card accent>
              <Stat label={t('machine.tier')} value={`Tier ${profile.tier.tier}`} hint={profile.tier.label} small />
              <div style={{ marginTop: 8 }}>
                <Badge tone={profile.tier.confidence === 'verified' ? 'ok' : 'warn'}>{profile.tier.confidence}</Badge>
              </div>
            </Card>
            <Card>
              <Stat label={t('machine.cpu')} value={`${profile.cpu.physical_cores ?? '—'}c / ${profile.cpu.logical_cores ?? '—'}t`} hint={profile.cpu.model ?? ''} small />
            </Card>
            <Card>
              <Stat
                label={t('machine.ram')}
                value={formatBytes(profile.memory.total_bytes)}
                hint={`${formatBytes(profile.memory.available_bytes)} ${t('machine.free')}`}
                small
              />
            </Card>
            <Card>
              <Stat
                label={t('machine.safe_budget')}
                value={budget ? formatBytes(budget.safe_budget_bytes) : '—'}
                hint={budget ? `${t('machine.reserve')}: ${formatBytes(budget.os_reserve_bytes)}` : ''}
                small
              />
            </Card>
          </div>

          <div className="grid cols-2">
            <Card>
              <CardHeader title={t('machine.gpu')} icon="layers" />
              {profile.gpu.devices?.length ? (
                <div className="stack">
                  {profile.gpu.devices.map((device, index) => (
                    <div key={index} className="stack" style={{ gap: 8 }}>
                      <div className="row between">
                        <strong>{device.name}</strong>
                        <Badge tone={device.unified_memory ? 'info' : 'accent'}>
                          {device.unified_memory ? t('machine.unified') : t('machine.dedicated_vram')}
                        </Badge>
                      </div>
                      <KV
                        items={[
                          [t('machine.dedicated_vram'), formatBytes(device.dedicated_vram_bytes ?? 0)],
                          [t('machine.shared_memory'), formatBytes(device.shared_gpu_memory_bytes ?? 0)],
                          [t('machine.driver'), String(device.driver_version ?? '—')],
                          [t('machine.evidence'), String(device.evidence ?? '—')],
                        ]}
                      />
                    </div>
                  ))}
                  {gpu?.unified_memory ? (
                    <Alert tone="warn">{t('machine.gpu_note', { shared: formatBytes(gpu.shared_gpu_memory_bytes ?? 0) })}</Alert>
                  ) : null}
                </div>
              ) : (
                <EmptyState title={t('common.not_detected')} icon="layers" />
              )}
            </Card>

            <Card>
              <CardHeader title={t('machine.accelerators')} icon="bolt" />
              <div className="stack" style={{ gap: 10 }}>
                {Object.entries(profile.accelerators ?? {}).map(([name, value]) => (
                  <div key={name} className="row between">
                    <div className="row" style={{ gap: 8 }}>
                      <span className={`dot ${value.status === 'detected' ? 'ok' : value.status === 'error' ? 'danger' : ''}`} />
                      <span className="small">{ACCELERATOR_LABELS[name] ?? name}</span>
                    </div>
                    <div className="row" style={{ gap: 8 }}>
                      {value.instance_version ? <span className="tiny dim">{String(value.instance_version)}</span> : null}
                      <Badge tone={statusTone(value.status)}>{t(`common.${value.status === 'detected' ? 'detected' : value.status === 'not_detected' ? 'not_detected' : 'unknown'}`)}</Badge>
                    </div>
                  </div>
                ))}
              </div>
              {profile.accelerators?.cuda?.status !== 'detected' ? (
                <div style={{ marginTop: 10 }}>
                  <Alert tone="warn">{String(profile.accelerators?.cuda?.reason ?? '')}</Alert>
                </div>
              ) : null}
            </Card>
          </div>

          <Card>
            <CardHeader
              title={t('machine.runtimes')}
              icon="terminal"
              subtitle={profile.runtimes?.llama_server?.status === 'detected' ? `${profile.runtimes.llama_server.candidates?.length ?? 0} × llama-server` : t('machine.no_runtimes')}
            />
            {profile.runtimes?.llama_server?.candidates?.length ? (
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>{t('models.path')}</th>
                      <th>{t('machine.version')}</th>
                      <th>{t('machine.backend')}</th>
                      <th>{t('common.status')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {profile.runtimes.llama_server.candidates.map((candidate, index) => (
                      <tr key={index}>
                        <td className="mono tiny">{String(candidate.path ?? '')}</td>
                        <td>{String(candidate.version ?? '—')}</td>
                        <td><Badge tone="accent">{String(candidate.accelerator ?? '—')}</Badge></td>
                        <td><Badge tone={candidate.status === 'verified' ? 'ok' : 'warn'}>{String(candidate.status ?? '—')}</Badge></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <EmptyState title={t('machine.no_runtimes')} icon="terminal" />
            )}
            <div className="divider" style={{ margin: '14px 0' }} />
            <div className="grid cols-2">
              <Field label={t('machine.runtime_paths')} hint={t('machine.folders_hint')}>
                <Textarea value={runtimeRoots} onChange={(event) => setRuntimeRoots(event.target.value)} placeholder={'C:\\llama.cpp\nD:\\AI'} />
              </Field>
              <div className="stack" style={{ justifyContent: 'flex-end' }}>
                <Button
                  icon="search"
                  loading={busy}
                  onClick={async () => {
                    const roots = runtimeRoots.split('\n').map((value) => value.trim()).filter(Boolean)
                    if (roots.length === 0) {
                      notify('warn', t('machine.runtime_paths'))
                      return
                    }
                    setBusy(true)
                    try {
                      const data = await api.scanRuntimes(roots)
                      setFound(data.candidates)
                      notify('ok', `${data.candidates.length} llama-server`)
                    } catch (error) {
                      notify('danger', error instanceof Error ? error.message : String(error))
                    } finally {
                      setBusy(false)
                    }
                  }}
                >
                  {t('machine.runtime_scan')}
                </Button>
                <Field label={t('machine.add_runtime')} htmlFor="runtime-path">
                  <Input id="runtime-path" value={manualPath} onChange={(event) => setManualPath(event.target.value)} placeholder="C:\\llama.cpp\\build\\bin\\llama-server.exe" />
                </Field>
                <Button
                  variant="ghost"
                  disabled={!manualPath.trim()}
                  onClick={async () => {
                    try {
                      await api.addRuntime(manualPath.trim())
                      await refreshMachine()
                      notify('ok', t('machine.runtime_added'))
                    } catch (error) {
                      notify('danger', error instanceof Error ? error.message : String(error))
                    }
                  }}
                >
                  {t('common.add')}
                </Button>
              </div>
            </div>
            {found.length > 0 ? (
              <div className="stack" style={{ marginTop: 12 }}>
                {found.map((candidate, index) => (
                  <div key={index} className="row between card tight">
                    <div className="grow">
                      <div className="mono tiny truncate">{String(candidate.path ?? '')}</div>
                      <div className="tiny dim">{String(candidate.version ?? '')} · {String(candidate.accelerator ?? '')} · {String(candidate.status ?? '')}</div>
                    </div>
                    <Button
                      size="sm"
                      onClick={async () => {
                        try {
                          await api.addRuntime(String(candidate.path))
                          await refreshMachine()
                          notify('ok', t('machine.runtime_added'))
                        } catch (error) {
                          notify('danger', error instanceof Error ? error.message : String(error))
                        }
                      }}
                    >
                      {t('common.add')}
                    </Button>
                  </div>
                ))}
              </div>
            ) : null}
          </Card>

          <div className="grid cols-2">
            <Card>
              <CardHeader title={t('machine.budget')} icon="shield" />
              {budget ? (
                <>
                  <KV
                    items={[
                      [t('machine.safe_budget'), formatBytes(budget.safe_budget_bytes)],
                      [t('machine.reserve'), formatBytes(budget.os_reserve_bytes)],
                      [t('machine.usable_ram'), formatBytes(budget.usable_ram_bytes)],
                      [t('machine.gpu_cap'), formatBytes(budget.gpu_shared_bytes)],
                      [t('machine.measured_at'), formatDate(budget.measured_at)],
                    ]}
                  />
                  <div style={{ marginTop: 12 }}>
                    <Progress value={(budget.safe_budget_bytes / Math.max(1, budget.total_ram_bytes)) * 100} />
                    <div className="tiny dim" style={{ marginTop: 4 }}>
                      {formatBytes(budget.safe_budget_bytes)} {t('common.of')} {formatBytes(budget.total_ram_bytes)}
                    </div>
                  </div>
                  {budget.notes.length > 0 ? (
                    <div className="stack" style={{ marginTop: 12, gap: 6 }}>
                      {budget.notes.map((note, index) => (
                        <span key={index} className="tiny dim">• {note}</span>
                      ))}
                    </div>
                  ) : null}
                </>
              ) : (
                <EmptyState title={t('common.not_measured')} icon="shield" />
              )}
            </Card>

            <Card>
              <CardHeader title={t('machine.browser_caps')} icon="globe" />
              <KV
                items={[
                  [t('machine.webgpu'), <Badge tone={browsers.webgpu ? 'ok' : 'warn'}>{browsers.webgpu ? t('common.yes') : t('common.no')}</Badge>],
                  [t('machine.wasm'), <Badge tone={browsers.wasm ? 'ok' : 'warn'}>{browsers.wasm ? t('common.yes') : t('common.no')}</Badge>],
                  [t('machine.sab'), <Badge tone={browsers.sab ? 'ok' : 'plain'}>{browsers.sab ? t('common.yes') : t('common.no')}</Badge>],
                  ['User agent', <span className="tiny mono">{navigator.userAgent.slice(0, 90)}</span>],
                ]}
              />
              <p className="tiny dim" style={{ marginTop: 10 }}>
                Browser inference is optional. The local service remains the primary runtime.
              </p>
            </Card>
          </div>

          <Card>
            <CardHeader title={t('machine.disks')} icon="database" />
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>{t('machine.mount')}</th>
                    <th>{t('machine.fs')}</th>
                    <th className="num">{t('machine.total')}</th>
                    <th className="num">{t('machine.free')}</th>
                    <th style={{ width: '30%' }}>{t('common.progress')}</th>
                  </tr>
                </thead>
                <tbody>
                  {profile.disks.map((disk) => (
                    <tr key={disk.mount}>
                      <td className="mono">{disk.mount}</td>
                      <td className="dim">{disk.fstype}</td>
                      <td className="num">{formatBytes(disk.total_bytes)}</td>
                      <td className="num">{formatBytes(disk.free_bytes)}</td>
                      <td>
                        <Progress value={((disk.total_bytes - disk.free_bytes) / Math.max(1, disk.total_bytes)) * 100} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card>
            <CardHeader title={t('machine.toolchains')} icon="terminal" />
            <div className="grid auto" style={{ gap: 10 }}>
              {Object.entries(profile.toolchains ?? {})
                .filter(([name]) => name !== 'python_packages')
                .map(([name, value]) => (
                  <div key={name} className="row between card tight" style={{ padding: '8px 10px' }}>
                    <span className="small mono">{name}</span>
                    <Badge tone={value.status === 'detected' ? 'ok' : 'plain'}>
                      {value.status === 'detected' ? String(value.version ?? t('common.detected')).slice(0, 34) : t('machine.missing')}
                    </Badge>
                  </div>
                ))}
            </div>
            {profile.toolchains?.python_packages ? (
              <div style={{ marginTop: 12 }}>
                <div className="label" style={{ marginBottom: 6 }}>{t('machine.python_packages')}</div>
                <div className="pill-row">
                  {Object.entries((profile.toolchains.python_packages as Record<string, unknown>).packages as Record<string, string | null> ?? {}).map(([name, version]) => (
                    <Badge key={name} tone={version ? 'ok' : 'plain'}>{name}{version ? ` ${version}` : ''}</Badge>
                  ))}
                </div>
              </div>
            ) : null}
          </Card>

          {profile.warnings.length > 0 ? (
            <Card>
              <CardHeader title={t('machine.warnings')} icon="alert" />
              <div className="stack" style={{ gap: 8 }}>
                {profile.warnings.map((warning, index) => (
                  <Alert key={index} tone="warn">{warning}</Alert>
                ))}
              </div>
            </Card>
          ) : null}

          <Card>
            <CardHeader title={t('machine.evidence')} icon="check" actions={
              <Button size="sm" variant="ghost" icon="copy" onClick={() => void navigator.clipboard.writeText(JSON.stringify(profile, null, 2))}>
                {t('machine.copy_profile')}
              </Button>
            } />
            <div className="mono-block" style={{ maxHeight: 260 }}>
              {JSON.stringify({ tier: profile.tier, scanned_at: profile.scanned_at, duration_ms: profile.duration_ms, gpu: profile.gpu, accelerators: profile.accelerators }, null, 2)}
            </div>
          </Card>
        </>
      )}
    </>
  )
}

function getBrowserCaps() {
  const nav = navigator as Navigator & { gpu?: unknown }
  const wasm = typeof WebAssembly === 'object' && typeof WebAssembly.validate === 'function'
  return {
    webgpu: Boolean(nav.gpu),
    wasm,
    sab: typeof SharedArrayBuffer !== 'undefined',
  }
}
