import type {
  AgentTask,
  BenchmarkRun,
  Budget,
  Candidate,
  Dataset,
  DatasetRecord,
  Fit,
  HFFile,
  HFModel,
  Job,
  JobEvent,
  MachineProfile,
  Model,
  TrainingEnvironment,
} from './types'

const TOKEN_KEY = 'foundry.token'
// Local development and the self-hosted build use a relative path. A deployed
// frontend points at the user's local service explicitly (never a hosted backend).
const BASE = (import.meta.env.VITE_FOUNDRY_API as string | undefined) ?? '/api/v1'

export class ApiError extends Error {
  status: number
  code: string
  detail: unknown

  constructor(status: number, code: string, message: string, detail?: unknown) {
    super(message)
    this.status = status
    this.code = code
    this.detail = detail
  }
}

type Init = {
  method?: string
  body?: unknown
  signal?: AbortSignal
}

class ApiClient {
  token: string | null = null

  constructor() {
    try {
      this.token = sessionStorage.getItem(TOKEN_KEY)
    } catch {
      this.token = null
    }
  }

  setToken(token: string | null) {
    this.token = token
    try {
      if (token) sessionStorage.setItem(TOKEN_KEY, token)
      else sessionStorage.removeItem(TOKEN_KEY)
    } catch {
      /* private mode: token stays in memory */
    }
  }

  private headers(extra: Record<string, string> = {}): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'X-Foundry-Client': '1',
      ...extra,
    }
    if (this.token) headers.Authorization = `Bearer ${this.token}`
    return headers
  }

  private async toError(response: Response): Promise<ApiError> {
    let code = `http_${response.status}`
    let message = response.statusText || 'Request failed'
    let detail: unknown
    try {
      const body = await response.json()
      if (body?.error) {
        code = body.error.code ?? code
        message = body.error.message ?? message
        detail = body.error
      } else if (body?.detail) {
        if (typeof body.detail === 'string') {
          message = body.detail
        } else {
          code = body.detail.code ?? code
          message = body.detail.message ?? message
          detail = body.detail
        }
      }
    } catch {
      /* non-JSON error body */
    }
    return new ApiError(response.status, code, message, detail)
  }

  async request<T>(path: string, init: Init = {}): Promise<T> {
    const response = await fetch(`${BASE}${path}`, {
      method: init.method ?? 'GET',
      headers: this.headers(),
      body: init.body === undefined ? undefined : JSON.stringify(init.body),
      signal: init.signal,
    })
    if (!response.ok) throw await this.toError(response)
    if (response.status === 204) return undefined as T
    const text = await response.text()
    return (text ? JSON.parse(text) : undefined) as T
  }

  async raw(path: string): Promise<{ blob: Blob; filename: string | null }> {
    const response = await fetch(`${BASE}${path}`, { headers: this.headers() })
    if (!response.ok) throw await this.toError(response)
    const disposition = response.headers.get('Content-Disposition') ?? ''
    const match = /filename="?([^"]+)"?/.exec(disposition)
    return { blob: await response.blob(), filename: match ? match[1] : null }
  }

  // -- system ------------------------------------------------------------
  health() {
    return this.request<{ status: string; version: string; model_loaded: boolean }>('/health')
  }

  async pair(label = 'web ui', code?: string): Promise<string> {
    const response = await fetch(`${BASE}/system/pair`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Foundry-Client': '1' },
      body: JSON.stringify({ label, code }),
    })
    if (!response.ok) throw await this.toError(response)
    const body = (await response.json()) as { token: string }
    this.setToken(body.token)
    return body.token
  }

  info() {
    return this.request<{
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
    }>('/system/info')
  }

  settings() {
    return this.request<{ effective: Record<string, unknown>; overrides: Record<string, unknown> }>('/settings')
  }

  updateSettings(patch: Record<string, unknown>) {
    return this.request<{ effective: Record<string, unknown>; overrides: Record<string, unknown> }>('/settings', {
      method: 'PUT',
      body: patch,
    })
  }

  activity(limit = 80) {
    return this.request<{ items: Array<{ id: number; ts: string; kind: string; message: string; data: Record<string, unknown> }> }>(
      `/activity?limit=${limit}`,
    )
  }

  revokeTokens() {
    return this.request<{ revoked: number }>('/system/revoke', { method: 'POST', body: {} })
  }

  eventsUrl(): string {
    return `${BASE}/events?token=${encodeURIComponent(this.token ?? '')}`
  }

  // -- machine -----------------------------------------------------------
  scanMachine() {
    return this.request<{ profile: MachineProfile }>('/machine/scan', { method: 'POST', body: {} })
  }

  machineProfile() {
    return this.request<{ profile: MachineProfile | null; budget: Budget | null; scanned: boolean }>('/machine/profile')
  }

  budget() {
    return this.request<Budget>('/machine/budget')
  }

  runtimes() {
    return this.request<{ runtimes: MachineProfile['runtimes']; binary_in_use: string | null }>('/machine/runtimes')
  }

  scanRuntimes(roots: string[]) {
    return this.request<{ candidates: Array<Record<string, unknown>>; roots: string[] }>('/machine/runtimes/scan', {
      method: 'POST',
      body: { roots },
    })
  }

  addRuntime(path: string) {
    return this.request<{ candidate: Record<string, unknown> }>('/machine/runtimes/add', {
      method: 'POST',
      body: { path },
    })
  }

  // -- models ------------------------------------------------------------
  models() {
    return this.request<{ models: Model[]; roots: string[] }>('/models')
  }

  scanModels(roots?: string[]) {
    return this.request<{ job_id: string }>('/models/scan', { method: 'POST', body: roots ? { roots } : {} })
  }

  model(id: string) {
    return this.request<{ model: Model; fits: Record<string, Fit>; budget: Budget }>(`/models/${id}`)
  }

  patchModel(id: string, patch: Record<string, unknown>) {
    return this.request<{ model: Model }>(`/models/${id}`, { method: 'PATCH', body: patch })
  }

  deleteModel(id: string) {
    return this.request<{ removed: string; file_deleted: boolean }>(`/models/${id}`, { method: 'DELETE' })
  }

  verifyModel(id: string) {
    return this.request<{ job_id: string }>(`/models/${id}/verify`, { method: 'POST', body: {} })
  }

  modelFit(id: string, context: number) {
    return this.request<{ fit: Fit; budget: Budget }>(`/models/${id}/fit?context=${context}`)
  }

  // -- inference ---------------------------------------------------------
  loadModel(payload: Record<string, unknown>) {
    return this.request<{ status: Record<string, unknown>; fit: Fit }>('/inference/load', { method: 'POST', body: payload })
  }

  unloadModel() {
    return this.request<{ status: Record<string, unknown> }>('/inference/unload', { method: 'POST', body: {} })
  }

  inferenceStatus() {
    return this.request<Record<string, unknown>>('/inference/status')
  }

  inferenceLogs(limit = 200) {
    return this.request<{ lines: string[] }>(`/inference/logs?limit=${limit}`)
  }

  cancelInference() {
    return this.request<{ cancelled: boolean }>('/inference/cancel', { method: 'POST', body: {} })
  }

  async chat(
    messages: Array<{ role: string; content: string }>,
    options: { temperature?: number; top_p?: number; max_tokens?: number },
    onEvent: (event: Record<string, unknown>) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    const response = await fetch(`${BASE}/inference/chat`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify({ messages, ...options }),
      signal,
    })
    if (!response.ok) throw await this.toError(response)
    if (!response.body) throw new ApiError(500, 'no_body', 'Streaming is not supported by this browser.')
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const chunks = buffer.split('\n\n')
      buffer = chunks.pop() ?? ''
      for (const chunk of chunks) {
        const line = chunk.split('\n').find((l) => l.startsWith('data:'))
        if (!line) continue
        try {
          onEvent(JSON.parse(line.slice(5).trim()))
        } catch {
          /* ignore malformed frame */
        }
      }
    }
  }

  // -- datasets ----------------------------------------------------------
  previewDataset(paths: string[], provider = 'auto') {
    return this.request<{
      inventory: { files: Array<Record<string, unknown>>; problems: Array<Record<string, unknown>>; total_bytes: number; file_count: number }
      samples: Array<Record<string, unknown>>
      errors: string[]
      stats: Record<string, number>
    }>('/datasets/preview', { method: 'POST', body: { paths, provider } })
  }

  importDataset(payload: Record<string, unknown>) {
    return this.request<{ job_id: string }>('/datasets/import', { method: 'POST', body: payload })
  }

  datasets() {
    return this.request<{ datasets: Dataset[] }>('/datasets')
  }

  dataset(id: string) {
    return this.request<{ dataset: Dataset }>(`/datasets/${id}`)
  }

  records(
    id: string,
    params: { offset?: number; limit?: number; split?: string; synthetic?: boolean; include_excluded?: boolean } = {},
  ) {
    const query = new URLSearchParams()
    if (params.offset !== undefined) query.set('offset', String(params.offset))
    if (params.limit !== undefined) query.set('limit', String(params.limit))
    if (params.split) query.set('split', params.split)
    if (params.synthetic !== undefined) query.set('synthetic', String(params.synthetic))
    if (params.include_excluded !== undefined) query.set('include_excluded', String(params.include_excluded))
    return this.request<{ total: number; offset: number; limit: number; items: DatasetRecord[] }>(
      `/datasets/${id}/records?${query.toString()}`,
    )
  }

  excludeRecord(datasetId: string, recordId: string, excluded: boolean, reason?: string) {
    return this.request<{ id: string; excluded: boolean }>(`/datasets/${datasetId}/records/${recordId}/exclude`, {
      method: 'POST',
      body: { excluded, reason },
    })
  }

  setRights(datasetId: string, rights_status: string) {
    return this.request<{ dataset_id: string; rights_status: string }>(`/datasets/${datasetId}/rights`, {
      method: 'POST',
      body: { rights_status },
    })
  }

  enrichDataset(datasetId: string, mode: 'deterministic' | 'model' | 'gateway', limit = 100) {
    return this.request<{ job_id: string }>(`/datasets/${datasetId}/enrich`, { method: 'POST', body: { mode, limit } })
  }

  freezeDataset(datasetId: string) {
    return this.request<{ manifest: Record<string, unknown> }>(`/datasets/${datasetId}/freeze`, { method: 'POST', body: {} })
  }

  annotations(datasetId: string) {
    return this.request<{ synthetic_rows: number; grounded: number; ungrounded: number; template_version: string }>(
      `/datasets/${datasetId}/annotations`,
    )
  }

  exportDatasetUrl(datasetId: string, options?: { format?: 'jsonl' | 'json' | 'csv'; limit?: number }) {
    const params = new URLSearchParams()
    if (options?.format) params.set('format', options.format)
    if (options?.limit) params.set('limit', String(options.limit))
    const query = params.toString()
    return this.raw(`/datasets/${datasetId}/export${query ? `?${query}` : ''}`)
  }

  deleteDataset(datasetId: string) {
    return this.request<{ deleted: string }>(`/datasets/${datasetId}`, { method: 'DELETE' })
  }

  browserRuntime() {
    return this.request<{ models: Array<Record<string, unknown>>; hint: string; ready: number; total: number; exists: boolean }>(
      '/browser/runtime',
    )
  }

  browserModelUrl(modelId: string, file: string) {
    return `${BASE}/browser/models/${encodeURIComponent(modelId)}/${file}`
  }

  // -- identity ----------------------------------------------------------
  identityDraft(prompt: string, name?: string, languages?: string[]) {
    return this.request<{ id: string; card: Record<string, unknown>; status: string }>('/identity/drafts', {
      method: 'POST',
      body: { prompt, name: name || undefined, languages },
    })
  }

  identityList() {
    return this.request<{ profiles: Array<{ id: string; name: string; status: string; dataset_id: string | null; created_at: string }> }>(
      '/identity',
    )
  }

  identityGet(id: string) {
    return this.request<{ profile: Record<string, unknown> & { card: Record<string, unknown> } }>(`/identity/${id}`)
  }

  identityPatch(id: string, patch: Record<string, unknown>) {
    return this.request<{ profile: Record<string, unknown> }>(`/identity/${id}`, { method: 'PATCH', body: patch })
  }

  identityApprove(id: string) {
    return this.request<{ profile: Record<string, unknown> }>(`/identity/${id}/approve`, { method: 'POST', body: {} })
  }

  identityDataset(id: string) {
    return this.request<{ dataset_id: string; record_count: number }>(`/identity/${id}/dataset`, { method: 'POST', body: {} })
  }

  identityEvaluate(id: string) {
    return this.request<{ score: number; passed: number; total: number; cases: Array<Record<string, unknown>> }>(
      `/identity/${id}/evaluate`,
      { method: 'POST', body: {} },
    )
  }

  // -- training ----------------------------------------------------------
  trainingEnvironment() {
    return this.request<TrainingEnvironment>('/training/environment')
  }

  trainingPreflight(payload: { dataset_id: string; base_model?: string | null; method?: string }) {
    return this.request<Record<string, unknown>>('/training/preflight', { method: 'POST', body: payload })
  }

  trainingSmoke() {
    return this.request<{ job_id: string }>('/training/smoke', { method: 'POST', body: {} })
  }

  trainingRecipes() {
    return this.request<{ recipes: Array<Record<string, unknown>> }>('/training/recipes')
  }

  trainingProject(payload: Record<string, unknown>) {
    return this.request<{ config: Record<string, unknown>; resources: Record<string, unknown> }>('/training/project', {
      method: 'POST',
      body: payload,
    })
  }

  trainingSft(payload: Record<string, unknown>) {
    return this.request<{ job_id: string; output_dir: string; resources: Record<string, unknown> }>('/training/sft', {
      method: 'POST',
      body: payload,
    })
  }

  trainingRuns() {
    return this.request<{ runs: Array<Record<string, unknown>> }>('/training/runs')
  }

  trainingMergeInt4(payload: { adapter_dir: string; store_dir?: string; base_model_dir?: string; output_dir?: string; dtype?: string }) {
    return this.request<{ job_id: string }>('/training/merge-int4', { method: 'POST', body: payload })
  }

  // -- benchmarks --------------------------------------------------------
  suites() {
    return this.request<{
      suites: Array<{
        id: string
        name: string
        version: string
        kind: string
        description: string
        cases: number | null
        requires: string[]
        available: boolean
        unavailable_reason?: string
        external?: boolean
        command?: string
        comparability: string
      }>
    }>('/benchmarks/suites')
  }

  startBenchmark(suite_id: string, judge = false) {
    return this.request<{ job_id: string }>('/benchmarks/runs', { method: 'POST', body: { suite_id, judge } })
  }

  benchmarkRuns() {
    return this.request<{ runs: BenchmarkRun[] }>('/benchmarks/runs')
  }

  benchmarkRun(id: string) {
    return this.request<{ run: BenchmarkRun }>(`/benchmarks/runs/${id}`)
  }

  benchmarkExport(id: string, format: 'json' | 'csv' | 'md') {
    return this.raw(`/benchmarks/runs/${id}/export?format=${format}`)
  }

  compareBenchmarks(a: string, b: string) {
    return this.request<Record<string, unknown>>('/benchmarks/compare', { method: 'POST', body: { a, b } })
  }

  // -- upgrades ----------------------------------------------------------
  upgradeTaxonomy() {
    return this.request<{ taxonomy: Record<string, string>; active: Record<string, unknown> | null }>('/upgrades/taxonomy')
  }

  analyzeRun(runId: string) {
    return this.request<Record<string, unknown>>(`/upgrades/analyze?run_id=${encodeURIComponent(runId)}`, {
      method: 'POST',
      body: {},
    })
  }

  candidates() {
    return this.request<{ candidates: Candidate[] }>('/upgrades/candidates')
  }

  createCandidate(payload: { parent_run_id: string; dataset_id: string; kind?: string; note?: string }) {
    return this.request<{ candidate: Candidate }>('/upgrades/candidates', { method: 'POST', body: payload })
  }

  evaluateCandidate(id: string) {
    return this.request<{ job_id: string }>(`/upgrades/candidates/${id}/evaluate`, { method: 'POST', body: {} })
  }

  promoteCandidate(id: string, approver: string, threshold = 0.05) {
    return this.request<Record<string, unknown>>(`/upgrades/candidates/${id}/promote`, {
      method: 'POST',
      body: { approver, threshold },
    })
  }

  rollbackCandidate(id: string) {
    return this.request<Record<string, unknown>>(`/upgrades/candidates/${id}/rollback`, { method: 'POST', body: {} })
  }

  lineage() {
    return this.request<{ lineage: Array<Record<string, unknown>> }>('/upgrades/lineage')
  }

  // -- agent -------------------------------------------------------------
  agentTools() {
    return this.request<{ tools: Array<{ name: string; description: string; mutating: boolean }> }>('/agent/tools')
  }

  agentTasks() {
    return this.request<{ tasks: AgentTask[] }>('/agent/tasks')
  }

  agentTask(id: string) {
    return this.request<{ task: AgentTask; job_id: string | null; events: JobEvent[] }>(`/agent/tasks/${id}`)
  }

  createAgentTask(goal: string, autonomy: string) {
    return this.request<{ task: AgentTask }>('/agent/tasks', { method: 'POST', body: { goal, autonomy } })
  }

  approveAgentTask(id: string) {
    return this.request<{ task: AgentTask }>(`/agent/tasks/${id}/approve`, { method: 'POST', body: {} })
  }

  cancelAgentTask(id: string) {
    return this.request<{ task: AgentTask }>(`/agent/tasks/${id}/cancel`, { method: 'POST', body: {} })
  }

  // -- hub ---------------------------------------------------------------
  hubSearch(q: string, task?: string, limit = 20, sort = 'downloads') {
    const query = new URLSearchParams({ q, limit: String(limit), sort })
    if (task) query.set('task', task)
    return this.request<{ query: string; results: HFModel[]; anonymous: boolean }>(`/hub/search?${query.toString()}`)
  }

  hubModel(repoId: string) {
    return this.request<{ model: HFModel; files: HFFile[]; suggested: string[]; total_bytes: number }>(
      `/hub/models/${repoId}`,
    )
  }

  hubToken() {
    return this.request<{ configured: boolean; masked: string | null; note: string }>('/hub/token')
  }

  setHubToken(token: string) {
    return this.request<{ status: { configured: boolean; masked: string | null }; identity: Record<string, unknown> }>(
      '/hub/token',
      { method: 'PUT', body: { token } },
    )
  }

  clearHubToken() {
    return this.request<{ configured: boolean; masked: string | null }>('/hub/token', { method: 'DELETE' })
  }

  testHubToken() {
    return this.request<Record<string, unknown>>('/hub/token/test', { method: 'POST', body: {} })
  }

  hubDownload(repoId: string, files: string[]) {
    return this.request<{ job_id: string; files: string[]; dest: string; anonymous: boolean }>('/hub/download', {
      method: 'POST',
      body: { repo_id: repoId, files },
    })
  }

  // -- gateway -----------------------------------------------------------
  gatewayStatus() {
    return this.request<{ configured: boolean; enabled: boolean; model: string; masked: string | null; note: string }>('/gateway')
  }

  setGateway(key: string, model?: string) {
    return this.request<Record<string, unknown>>('/gateway', { method: 'PUT', body: { key, model } })
  }

  enableGateway(enabled: boolean) {
    return this.request<Record<string, unknown>>('/gateway/enable', { method: 'POST', body: { enabled } })
  }

  clearGateway() {
    return this.request<Record<string, unknown>>('/gateway', { method: 'DELETE' })
  }

  gatewayModels() {
    return this.request<{ models: Array<{ id: string; name: string; owned_by: string }> }>('/gateway/models')
  }

  testGateway() {
    return this.request<Record<string, unknown>>('/gateway/test', { method: 'POST', body: {} })
  }

  // -- v3: gguf / moe / tuning / sources / pipeline ----------------------
  ggufTools() {
    return this.request<{ toolchain: Record<string, string | null>; ready: boolean }>('/gguf/tools')
  }

  ggufExport(modelId: string, dtype = 'f16') {
    return this.request<{ job_id: string; output: string }>('/gguf/export', { method: 'POST', body: { model_id: modelId, dtype } })
  }

  ggufQuantize(source: string, method: string) {
    return this.request<{ job_id: string; output: string }>('/gguf/quantize', { method: 'POST', body: { source, method } })
  }

  ggufDequantize(source: string, method = 'F16') {
    return this.request<{ job_id: string; output: string }>('/gguf/dequantize', { method: 'POST', body: { source, method } })
  }

  moeBuild(modelId: string, experts: number, topK: number) {
    return this.request<{ job_id: string; output: string }>('/moe/build', { method: 'POST', body: { model_id: modelId, experts, top_k: topK } })
  }

  tuneVulkan(modelId: string, ngl: number[], context: number) {
    return this.request<{ job_id: string }>('/tune/vulkan', { method: 'POST', body: { model_id: modelId, ngl, context } })
  }

  sources() {
    return this.request<{ sources: Array<Record<string, unknown>> }>('/sources')
  }

  importSources(sources: string[], rights: string) {
    return this.request<{ job_id: string }>('/sources/import', { method: 'POST', body: { sources, rights_status: rights } })
  }

  trainingAuto(payload: Record<string, unknown>) {
    return this.request<{ job_id: string }>('/training/auto', { method: 'POST', body: payload })
  }

  trainingPlaybook() {
    return this.request<{ entries: Array<Record<string, unknown>> }>('/training/playbook')
  }

  pipelineRun(stages: string[], options: Record<string, unknown>) {
    return this.request<{ job_id: string }>('/pipeline/run', { method: 'POST', body: { stages, options } })
  }

  cleanSources(sources: string[], rights = 'confirmed') {
    return this.request<{ job_id: string }>('/sources/clean', {
      method: 'POST',
      body: { sources, rights_status: rights, quality_filter: true, quarantine: true },
    })
  }

  trainingUnified(payload: Record<string, unknown>) {
    return this.request<{ job_id: string }>('/training/unified', { method: 'POST', body: payload })
  }

  tuneProfile(modelId: string) {
    return this.request<{ profile: Record<string, unknown> }>(`/tune/profile/${modelId}`)
  }

  // -- runs --------------------------------------------------------------
  runs(limit = 50, type?: string) {
    const query = new URLSearchParams({ limit: String(limit) })
    if (type) query.set('type', type)
    return this.request<{ runs: Job[] }>(`/runs?${query.toString()}`)
  }

  run(id: string) {
    return this.request<{ run: Job; events: JobEvent[] }>(`/runs/${id}`)
  }

  cancelRun(id: string) {
    return this.request<{ cancelled: boolean }>(`/runs/${id}/cancel`, { method: 'POST', body: {} })
  }
}

export const api = new ApiClient()
