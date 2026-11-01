import { useCallback, useEffect, useRef, useState } from 'react'

import { api } from '../lib/api'
import type { Job, JobEvent } from '../lib/types'

/** Poll a job until it reaches a terminal state. */
export function useJob(jobId: string | null) {
  const [job, setJob] = useState<Job | null>(null)
  const [events, setEvents] = useState<JobEvent[]>([])
  const [error, setError] = useState<string | null>(null)
  const timer = useRef<number | null>(null)

  const refresh = useCallback(async () => {
    if (!jobId) return
    try {
      const data = await api.run(jobId)
      setJob(data.run)
      setEvents(data.events)
      setError(null)
      return data.run.status
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return null
    }
  }, [jobId])

  useEffect(() => {
    if (!jobId) {
      setJob(null)
      setEvents([])
      return
    }
    let cancelled = false
    const tick = async () => {
      const status = await refresh()
      if (cancelled) return
      if (status && ['queued', 'running'].includes(status)) {
        timer.current = window.setTimeout(tick, 900)
      }
    }
    void tick()
    return () => {
      cancelled = true
      if (timer.current) window.clearTimeout(timer.current)
    }
  }, [jobId, refresh])

  return { job, events, error, refresh }
}

/** Small data-loading hook with retry support. */
export function useAsync<T>(loader: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [nonce, setNonce] = useState(0)
  const loaderRef = useRef(loader)
  loaderRef.current = loader

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    loaderRef
      .current()
      .then((value) => {
        if (!cancelled) {
          setData(value)
          setError(null)
        }
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce])

  const reload = useCallback(() => setNonce((n) => n + 1), [])
  return { data, loading, error, reload, setData }
}
