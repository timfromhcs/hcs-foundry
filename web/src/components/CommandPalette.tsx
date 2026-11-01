import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { createPortal } from 'react-dom'

import { useApp } from '../context/AppContext'
import { api } from '../lib/api'
import { LANGUAGES } from '../i18n'
import { Icon, type IconName } from './icons'

interface Command {
  id: string
  label: string
  group: string
  icon: IconName
  run: () => void
}

export function CommandPalette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { setTheme, theme, setLang, lang, startJob, notify, scanMachine } = useApp()
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)

  const commands = useMemo<Command[]>(() => {
    const routes: Array<{ path: string; key: string; icon: IconName }> = [
      { path: '/', key: 'nav.overview', icon: 'grid' },
      { path: '/models', key: 'nav.models', icon: 'box' },
      { path: '/hub', key: 'nav.hub', icon: 'download' },
      { path: '/playground', key: 'nav.playground', icon: 'chat' },
      { path: '/data', key: 'nav.data', icon: 'database' },
      { path: '/identity', key: 'nav.identity', icon: 'fingerprint' },
      { path: '/training', key: 'nav.training', icon: 'graduation' },
      { path: '/benchmarks', key: 'nav.benchmarks', icon: 'chart' },
      { path: '/upgrades', key: 'nav.upgrades', icon: 'sparkles' },
      { path: '/agent', key: 'nav.agent', icon: 'bot' },
      { path: '/machine', key: 'nav.machine', icon: 'cpu' },
      { path: '/runs', key: 'nav.runs', icon: 'history' },
      { path: '/settings', key: 'nav.settings', icon: 'settings' },
    ]
    const list: Command[] = routes.map((route) => ({
      id: `go:${route.path}`,
      label: t(route.key),
      group: t('palette.navigate'),
      icon: route.icon,
      run: () => navigate(route.path),
    }))
    list.push(
      {
        id: 'action:scan-machine',
        label: t('palette.scan_machine'),
        group: t('palette.actions'),
        icon: 'cpu',
        run: () => {
          void scanMachine().catch((error) => notify('danger', error instanceof Error ? error.message : String(error)))
        },
      },
      {
        id: 'action:scan-models',
        label: t('palette.scan_models'),
        group: t('palette.actions'),
        icon: 'search',
        run: () => {
          void startJob(() => api.scanModels(), t('palette.scan_models')).catch((error) =>
            notify('danger', error instanceof Error ? error.message : String(error)),
          )
        },
      },
      {
        id: 'action:theme',
        label: t('palette.toggle_theme'),
        group: t('palette.actions'),
        icon: theme === 'dark' ? 'sun' : 'moon',
        run: () => setTheme(theme === 'dark' ? 'light' : 'dark'),
      },
      {
        id: 'action:language',
        label: `${t('palette.language')} (${LANGUAGES.find((item) => item.code === lang)?.label})`,
        group: t('palette.actions'),
        icon: 'globe',
        run: () => {
          const index = LANGUAGES.findIndex((item) => item.code === lang)
          setLang(LANGUAGES[(index + 1) % LANGUAGES.length].code)
        },
      },
    )
    return list
  }, [t, navigate, scanMachine, startJob, notify, theme, setTheme, lang, setLang])

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle) return commands
    return commands.filter((command) => `${command.group} ${command.label}`.toLowerCase().includes(needle))
  }, [commands, query])

  useEffect(() => {
    if (open) {
      setQuery('')
      setActive(0)
      window.setTimeout(() => inputRef.current?.focus(), 10)
    }
  }, [open])

  useEffect(() => {
    if (!open) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onClose()
      } else if (event.key === 'ArrowDown') {
        event.preventDefault()
        setActive((value) => Math.min(filtered.length - 1, value + 1))
      } else if (event.key === 'ArrowUp') {
        event.preventDefault()
        setActive((value) => Math.max(0, value - 1))
      } else if (event.key === 'Enter') {
        event.preventDefault()
        const command = filtered[active]
        if (command) {
          command.run()
          onClose()
        }
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open, filtered, active, onClose])

  if (!open) return null

  let lastGroup = ''

  return createPortal(
    <div className="palette-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className="palette" role="dialog" aria-modal="true" aria-label={t('palette.open')}>
        <div className="palette-input">
          <Icon name="search" size={16} />
          <input
            ref={inputRef}
            value={query}
            onChange={(event) => {
              setQuery(event.target.value)
              setActive(0)
            }}
            placeholder={t('palette.placeholder')}
            aria-label={t('palette.placeholder')}
          />
          <kbd>esc</kbd>
        </div>
        <div className="palette-list" role="listbox">
          {filtered.length === 0 ? <div className="palette-empty">{t('palette.no_results')}</div> : null}
          {filtered.map((command, index) => {
            const showGroup = command.group !== lastGroup
            lastGroup = command.group
            return (
              <div key={command.id}>
                {showGroup ? <div className="palette-group">{command.group}</div> : null}
                <button
                  type="button"
                  role="option"
                  aria-selected={index === active}
                  className={`palette-item ${index === active ? 'active' : ''}`}
                  onMouseEnter={() => setActive(index)}
                  onClick={() => {
                    command.run()
                    onClose()
                  }}
                >
                  <Icon name={command.icon} size={15} />
                  <span>{command.label}</span>
                </button>
              </div>
            )
          })}
        </div>
      </div>
    </div>,
    document.body,
  )
}
