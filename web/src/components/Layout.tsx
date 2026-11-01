import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useTranslation } from 'react-i18next'

import { useApp } from '../context/AppContext'
import { LANGUAGES } from '../i18n'
import { formatPercent, statusTone } from '../lib/format'
import { Badge, Button, Progress } from './ui'
import { Icon, type IconName } from './icons'
import { CommandPalette } from './CommandPalette'
import { BottomNav } from './BottomNav'
import { Onboarding } from '../pages/Onboarding'

const NAV: Array<{ group: string; items: Array<{ to: string; label: string; icon: IconName }> }> = [
  {
    group: 'nav.group.workspace',
    items: [
      { to: '/', label: 'nav.overview', icon: 'grid' },
      { to: '/models', label: 'nav.models', icon: 'box' },
      { to: '/hub', label: 'nav.hub', icon: 'download' },
      { to: '/playground', label: 'nav.playground', icon: 'chat' },
    ],
  },
  {
    group: 'nav.group.lab',
    items: [
      { to: '/data', label: 'nav.data', icon: 'database' },
      { to: '/identity', label: 'nav.identity', icon: 'fingerprint' },
      { to: '/training', label: 'nav.training', icon: 'graduation' },
      { to: '/benchmarks', label: 'nav.benchmarks', icon: 'chart' },
      { to: '/upgrades', label: 'nav.upgrades', icon: 'sparkles' },
      { to: '/pipeline', label: 'nav.pipeline', icon: 'layers' },
      { to: '/agent', label: 'nav.agent', icon: 'bot' },
    ],
  },
  {
    group: 'nav.group.system',
    items: [
      { to: '/machine', label: 'nav.machine', icon: 'cpu' },
      { to: '/runs', label: 'nav.runs', icon: 'history' },
      { to: '/settings', label: 'nav.settings', icon: 'settings' },
    ],
  },
]

function pageTitle(pathname: string): string {
  const map: Record<string, string> = {
    '/': 'nav.overview',
    '/machine': 'nav.machine',
    '/models': 'nav.models',
    '/hub': 'nav.hub',
    '/playground': 'nav.playground',
    '/data': 'nav.data',
    '/identity': 'nav.identity',
    '/training': 'nav.training',
    '/benchmarks': 'nav.benchmarks',
    '/upgrades': 'nav.upgrades',
    '/pipeline': 'nav.pipeline',
    '/agent': 'nav.agent',
    '/runs': 'nav.runs',
    '/settings': 'nav.settings',
  }
  return map[pathname] ?? 'nav.overview'
}

function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { t } = useTranslation()
  const { sseConnected, info } = useApp()
  return (
    <>
      <div className={`sidebar-backdrop ${open ? 'open' : ''}`} onClick={onClose} aria-hidden="true" />
      <aside className={`sidebar ${open ? 'open' : ''}`} aria-label="Primary">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">H</div>
          <div className="brand-text">
            <strong>{t('app.name')}</strong>
            <span>local model lab</span>
          </div>
        </div>
        <nav className="nav">
          {NAV.map((group) => (
            <div key={group.group}>
              <div className="nav-group">{t(group.group)}</div>
              {group.items.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === '/'}
                  className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                  onClick={onClose}
                >
                  <span className="ico"><Icon name={item.icon} size={17} /></span>
                  {t(item.label)}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="conn">
            <span className={`dot ${sseConnected ? 'ok pulse' : 'warn'}`} aria-hidden="true" />
            <span>{sseConnected ? t('common.connected') : t('common.disconnected')}</span>
          </div>
          <div className="conn" title={info?.data_dir}>
            <Icon name="shield" size={13} />
            <span>{t('settings.local_only')} · v{info?.version ?? '—'}</span>
          </div>
        </div>
      </aside>
    </>
  )
}

function Topbar({ onMenu, onPalette }: { onMenu: () => void; onPalette: () => void }) {
  const { t } = useTranslation()
  const { lang, setLang, theme, setTheme, info } = useApp()
  const location = useLocation()
  return (
    <header className="topbar">
      <Button variant="ghost" size="sm" icon="menu" className="nav-toggle" onClick={onMenu} aria-label={t('nav.overview')} />
      <div className="topbar-title">
        <h1>{t(pageTitle(location.pathname))}</h1>
      </div>
      <div className="topbar-actions">
        <button type="button" className="palette-trigger" onClick={onPalette} aria-label={t('palette.open')}>
          <Icon name="search" size={14} />
          <span className="palette-trigger-label">{t('common.search')}</span>
          <kbd>{t('palette.hint')}</kbd>
        </button>
        {info?.model_loaded ? <Badge tone="ok" dot>{t('models.model_loaded')}</Badge> : null}
        <label className="sr-only" htmlFor="lang-select">{t('common.language')}</label>
        <select
          id="lang-select"
          className="select"
          style={{ width: 110, height: 30, minHeight: 30, fontSize: 12 }}
          value={lang}
          onChange={(event) => setLang(event.target.value as typeof lang)}
        >
          {LANGUAGES.map((language) => (
            <option key={language.code} value={language.code}>
              {language.label}
            </option>
          ))}
        </select>
        <Button
          variant="ghost"
          size="sm"
          icon={theme === 'dark' ? 'sun' : 'moon'}
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          aria-label={t('common.theme')}
          title={t('common.theme')}
        />
      </div>
    </header>
  )
}

function JobTray() {
  const { t } = useTranslation()
  const { jobs, cancelJob } = useApp()
  const [collapsed, setCollapsed] = useState(false)
  const active = jobs.filter((job) => ['queued', 'running'].includes(job.status))
  const recent = jobs.filter((job) => !['queued', 'running'].includes(job.status)).slice(0, 2)
  const shown = [...active, ...recent].slice(0, 3)
  if (shown.length === 0) return null

  return (
    <aside className="job-tray" aria-label={t('jobs.active')} aria-live="polite">
      <div className="row between" style={{ padding: '0 4px' }}>
        <span className="tiny dim">
          {active.length > 0 ? `${active.length} ${t('jobs.active').toLowerCase()}` : t('jobs.none')}
        </span>
        <button type="button" className="linkish" onClick={() => setCollapsed((value) => !value)}>
          {collapsed ? t('common.open') : t('common.dismiss')}
        </button>
      </div>
      {!collapsed &&
        shown.map((job) => {
          const running = ['queued', 'running'].includes(job.status)
          return (
            <div className="job-card" key={job.id}>
              <div className="job-head">
                <span className={`dot ${running ? 'accent pulse' : statusTone(job.status)}`} aria-hidden="true" />
                <span className="job-title grow truncate">{job.label}</span>
                <Badge tone={statusTone(job.status)}>{t(`common.${job.status}`)}</Badge>
                {running ? (
                  <Button size="sm" variant="ghost" icon="stop" onClick={() => void cancelJob(job.id)} aria-label={t('jobs.cancel')} />
                ) : null}
              </div>
              {running ? (
                <Progress value={job.progress} indeterminate={job.progress === 0 && job.status === 'queued'} />
              ) : null}
              <div className="job-msg truncate">{job.error ?? job.message ?? formatPercent(job.progress / 100)}</div>
            </div>
          )
        })}
    </aside>
  )
}

function Toasts() {
  const { toasts, dismissToast } = useApp()
  if (toasts.length === 0) return null
  return (
    <div className="toasts" role="status" aria-live="polite">
      {toasts.map((toast) => (
        <div key={toast.id} className={`toast ${toast.kind}`}>
          <Icon name={toast.kind === 'danger' ? 'alert' : toast.kind === 'ok' ? 'check' : 'shield'} size={15} style={{ marginTop: 2, flex: 'none' }} />
          <span>{toast.message}</span>
          <button type="button" className="btn ghost sm icon" onClick={() => dismissToast(toast.id)} aria-label="Dismiss">
            <Icon name="close" size={12} />
          </button>
        </div>
      ))}
    </div>
  )
}

export function Layout() {
  const { t } = useTranslation()
  const { paired } = useApp()
  const [mobileOpen, setMobileOpen] = useState(false)
  const [paletteOpen, setPaletteOpen] = useState(false)

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        setPaletteOpen((value) => !value)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  if (!paired) return <Onboarding />

  return (
    <div className="shell">
      <a className="skip-link" href="#main">{t('common.skip')}</a>
      <Sidebar open={mobileOpen} onClose={() => setMobileOpen(false)} />
      <div className="main">
        <Topbar onMenu={() => setMobileOpen(true)} onPalette={() => setPaletteOpen(true)} />
        <main id="main" className="page">
          <Outlet />
        </main>
      </div>
      <BottomNav onMore={() => setMobileOpen(true)} />
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
      <JobTray />
      <Toasts />
    </div>
  )
}
