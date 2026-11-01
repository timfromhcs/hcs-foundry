import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'

import { api, ApiError } from '../lib/api'
import type { Budget, Job, MachineProfile } from '../lib/types'
import { setLanguage, type LanguageCode } from '../i18n'

type Theme = 'dark' | 'light'
type ToastKind = 'ok' | 'danger' | 'warn' | 'info'

interface Toast {
  id: number
  kind: ToastKind
  message: string
}

interface SystemInfo {
  app: string
  version: string
  schema_version: number
  data_dir: string
  telemetry: boolean
  local_only: boolean
  paired_tokens: number
  has_profile: boolean
  model_loaded: boolean
  time: string
}

interface AppContextValue {
  theme: Theme
  setTheme: (theme: Theme) => void
  lang: LanguageCode
  setLang: (lang: LanguageCode) => void
  paired: boolean
  connecting: boolean
  connect: (code?: string) => Promise<void>
  disconnect: () => void
  info: SystemInfo | null
  refreshInfo: () => Promise<void>
  profile: MachineProfile | null
  budget: Budget | null
  refreshMachine: () => Promise<void>
  scanMachine: () => Promise<void>
  toasts: Toast[]
  notify: (kind: ToastKind, message: string) => void
  dismissToast: (id: number) => void
  jobs: Array<Job & { label: string }>
  startJob: (request: () => Promise<{ job_id: string }>, label: string) => Promise<string>
  cancelJob: (id: string) => Promise<void>
  sseConnected: boolean
  lastError: string | null
}

const AppContext = createContext<AppContextValue | null>(null)

function readTheme(): Theme {
  try {
    const stored = localStorage.getItem('foundry.theme')
    if (stored === 'light' || stored === 'dark') return stored
  } catch {
    /* ignore */
  }
  return 'dark'
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(readTheme)
  const [lang, setLangState] = useState<LanguageCode>(
    (() => {
      try {
        const stored = localStorage.getItem('foundry.lang')
        if (stored === 'en' || stored === 'de' || stored === 'fr') return stored
      } catch {
        /* ignore */
      }
      return 'en'
    })(),
  )
  const [paired, setPaired] = useState<boolean>(Boolean(api.token))
  const [connecting, setConnecting] = useState(false)
  const [info, setInfo] = useState<SystemInfo | null>(null)
  const [profile, setProfile] = useState<MachineProfile | null>(null)
  const [budget, setBudget] = useState<Budget | null>(null)
  const [toasts, setToasts] = useState<Toast[]>([])
  const [jobMap, setJobMap] = useState<Record<string, Job & { label: string }>>({})
  const [sseConnected, setSseConnected] = useState(false)
  const [lastError, setLastError] = useState<string | null>(null)
  const toastSeq = useRef(1)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem('foundry.theme', theme)
    } catch {
      /* ignore */
    }
  }, [theme])

  const setTheme = useCallback((next: Theme) => setThemeState(next), [])

  const setLang = useCallback((next: LanguageCode) => {
    setLangState(next)
    setLanguage(next)
  }, [])

  const notify = useCallback((kind: ToastKind, message: string) => {
    const id = toastSeq.current++
    setToasts((prev) => [...prev.slice(-4), { id, kind, message }])
    window.setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id))
    }, kind === 'danger' ? 9000 : 5000)
  }, [])

  const dismissToast = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  const refreshInfo = useCallback(async () => {
    try {
      const data = await api.info()
      setInfo(data)
      setLastError(null)
      setPaired(true)
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        setPaired(false)
        api.setToken(null)
      }
      setLastError(error instanceof Error ? error.message : String(error))
    }
  }, [])

  const refreshMachine = useCallback(async () => {
    try {
      const data = await api.machineProfile()
      setProfile(data.profile)
      setBudget(data.budget)
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        setProfile(null)
        setBudget(null)
        return
      }
      if (!(error instanceof ApiError && error.status === 401)) {
        setLastError(error instanceof Error ? error.message : String(error))
      }
    }
  }, [])

  const scanMachine = useCallback(async () => {
    const data = await api.scanMachine()
    setProfile(data.profile)
    try {
      setBudget(await api.budget())
    } catch {
      /* budget is informational */
    }
    notify('ok', data.profile.tier.label)
  }, [notify])

  const connect = useCallback(
    async (code?: string) => {
      setConnecting(true)
      try {
        await api.pair('web ui', code)
        setPaired(true)
        await refreshInfo()
        await refreshMachine()
        setLastError(null)
      } catch (error) {
        setPaired(false)
        const message = error instanceof Error ? error.message : String(error)
        setLastError(message)
        throw error
      } finally {
        setConnecting(false)
      }
    },
    [refreshInfo, refreshMachine],
  )

  const disconnect = useCallback(() => {
    api.setToken(null)
    setPaired(false)
    setInfo(null)
  }, [])

  const upsertJob = useCallback((job: Job, label?: string) => {
    setJobMap((prev) => ({
      ...prev,
      [job.id]: { ...job, label: label ?? prev[job.id]?.label ?? job.type },
    }))
  }, [])

  const startJob = useCallback(
    async (request: () => Promise<{ job_id: string }>, label: string) => {
      const { job_id } = await request()
      try {
        const { run } = await api.run(job_id)
        upsertJob(run, label)
      } catch {
        setJobMap((prev) => ({
          ...prev,
          [job_id]: {
            id: job_id,
            type: 'job',
            status: 'queued',
            progress: 0,
            message: label,
            created_at: new Date().toISOString(),
            started_at: null,
            finished_at: null,
            error: null,
            input: {},
            output: {},
            cancel_requested: false,
            label,
          },
        }))
      }
      return job_id
    },
    [upsertJob],
  )

  const cancelJob = useCallback(
    async (id: string) => {
      try {
        await api.cancelRun(id)
        notify('info', 'Cancellation requested')
      } catch (error) {
        notify('danger', error instanceof Error ? error.message : String(error))
      }
    },
    [notify],
  )

  // -- initial load -------------------------------------------------------
  useEffect(() => {
    if (!api.token) return
    void refreshInfo()
    void refreshMachine()
  }, [refreshInfo, refreshMachine])

  // -- server-sent events -------------------------------------------------
  useEffect(() => {
    if (!paired) {
      setSseConnected(false)
      return
    }
    let source: EventSource | null = null
    let closed = false
    let retry: number | null = null

    const open = () => {
      if (closed) return
      source = new EventSource(api.eventsUrl())
      source.onopen = () => setSseConnected(true)
      source.onerror = () => {
        setSseConnected(false)
        source?.close()
        if (!closed) retry = window.setTimeout(open, 4000)
      }
      source.onmessage = (message) => {
        try {
          const event = JSON.parse(message.data) as Record<string, unknown>
          const jobId = event.job_id as string | undefined
          if (!jobId) return
          setJobMap((prev) => {
            const existing = prev[jobId]
            if (!existing) return prev
            if (event.type === 'job_event') {
              const data = (event.data as Record<string, unknown>) ?? {}
              return {
                ...prev,
                [jobId]: {
                  ...existing,
                  progress: typeof data.progress === 'number' ? data.progress : existing.progress,
                  message: (event.message as string) ?? existing.message,
                },
              }
            }
            if (event.type === 'job_finished') {
              return {
                ...prev,
                [jobId]: {
                  ...existing,
                  status: (event.status as Job['status']) ?? existing.status,
                  finished_at: new Date().toISOString(),
                  progress: event.status === 'succeeded' ? 100 : existing.progress,
                },
              }
            }
            if (event.type === 'job_started') {
              return { ...prev, [jobId]: { ...existing, status: 'running' } }
            }
            return prev
          })
        } catch {
          /* ignore malformed frame */
        }
      }
    }

    open()
    return () => {
      closed = true
      if (retry) window.clearTimeout(retry)
      source?.close()
      setSseConnected(false)
    }
  }, [paired])

  // -- poll active jobs as a fallback ------------------------------------
  useEffect(() => {
    const active = Object.values(jobMap).filter((job) =>
      ['queued', 'running'].includes(job.status),
    )
    if (active.length === 0) return
    const timer = window.setInterval(async () => {
      for (const job of active) {
        try {
          const { run } = await api.run(job.id)
          upsertJob(run)
        } catch {
          /* keep previous state */
        }
      }
    }, 2500)
    return () => window.clearInterval(timer)
  }, [jobMap, upsertJob])

  const jobs = useMemo(() => Object.values(jobMap).sort((a, b) => b.created_at.localeCompare(a.created_at)), [jobMap])

  const value: AppContextValue = {
    theme,
    setTheme,
    lang,
    setLang,
    paired,
    connecting,
    connect,
    disconnect,
    info,
    refreshInfo,
    profile,
    budget,
    refreshMachine,
    scanMachine,
    toasts,
    notify,
    dismissToast,
    jobs,
    startJob,
    cancelJob,
    sseConnected,
    lastError,
  }

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp(): AppContextValue {
  const context = useContext(AppContext)
  if (!context) throw new Error('useApp must be used inside AppProvider')
  return context
}
