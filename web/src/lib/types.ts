export type Json = Record<string, unknown>

export interface Job {
  id: string
  type: string
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted'
  progress: number
  message: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
  error: string | null
  input: Json
  output: Json
  cancel_requested: boolean
}

export interface JobEvent {
  seq: number
  ts: string
  level: string
  kind: string
  message: string
  data: Json
}

export interface MachineProfile {
  schema_version: string
  scanned_at: string
  duration_ms: number | null
  os: Json & { name?: string; version?: string; build?: string; arch?: string; hostname?: string }
  cpu: Json & { model?: string; physical_cores?: number; logical_cores?: number; max_clock_mhz?: number }
  memory: Json & { total_bytes?: number; available_bytes?: number; percent_used?: number }
  gpu: Json & {
    devices?: Array<Json & {
      name?: string
      vendor?: string
      dedicated_vram_bytes?: number
      shared_gpu_memory_bytes?: number
      unified_memory?: boolean
      driver_version?: string
    }>
    dedicated_vram_bytes?: number
    shared_gpu_memory_bytes?: number
    unified_memory?: boolean
  }
  accelerators: Record<string, Json & { status?: string; reason?: string; devices?: string[]; instance_version?: string }>
  disks: Array<{ mount: string; total_bytes: number; free_bytes: number; fstype: string }>
  toolchains: Record<string, Json>
  runtimes: Json & {
    llama_server?: { status?: string; candidates?: Array<Json & { path?: string; version?: string; accelerator?: string; status?: string }> }
    ollama?: Json
    lmstudio?: Json
  }
  tier: { tier: number; label: string; confidence: string; reasons: string[] }
  warnings: string[]
}

export interface Budget {
  measured_at: string
  total_ram_bytes: number
  available_ram_bytes: number
  os_reserve_bytes: number
  usable_ram_bytes: number
  unified_memory: boolean
  gpu_shared_bytes: number
  gpu_headroom_bytes: number
  usable_gpu_bytes: number | null
  safe_budget_bytes: number
  tier: number
  tier_label: string
  notes: string[]
}

export interface Fit {
  status: 'comfortable' | 'tight' | 'risky' | 'too_large'
  confidence: string
  estimated_bytes_low: number
  estimated_bytes_high: number
  weights_bytes: number
  kv_cache_bytes: number
  overhead_bytes: number
  context: number
  safe_budget_bytes: number
  headroom_bytes: number
  kv_bytes_per_token: number | null
  notes: string[]
}

export interface Model {
  id: string
  path: string
  name: string
  size_bytes: number
  sha256: string | null
  architecture: string | null
  quantization: string | null
  parameter_count: number | null
  context_max: number | null
  license: string | null
  source_url: string | null
  task: string | null
  verified_at: string | null
  added_at: string
  meta: {
    gguf?: Json
    assets?: { projectors?: string[]; folder_gguf_bytes?: number }
    has_projector?: boolean
  }
  exists: boolean
  sha256_verified: boolean
  default_fit?: Fit
}

export interface Dataset {
  id: string
  name: string
  provider: string
  status: string
  rights_status: string
  record_count: number
  manifest: Json & { dataset_hash?: string; holdout_hash?: string; splits?: Record<string, number> }
  stats: Json & {
    by_task?: Record<string, number>
    by_language?: Record<string, number>
    by_split?: Record<string, number>
    by_outcome?: Record<string, number>
    synthetic_ratio?: number
    duplicates_removed?: number
    near_duplicates_removed?: number
    redactions?: Record<string, number>
  }
  holdout_hash: string | null
  created_at: string
  frozen_at: string | null
}

export interface DatasetRecord {
  id: string
  record_id: string
  split: string
  synthetic: boolean
  excluded: boolean
  exclusion_reason: string | null
  rights_status: string
  content_hash: string
  payload: Json & {
    task?: { user_request?: string; task_type?: string }
    trace?: Array<{ kind: string; content: string; tool_name?: string | null }>
    teaching?: Json
    privacy?: Json
    metadata?: Json
    outcome?: Json
  }
}

export interface BenchmarkRun {
  id: string
  suite_id: string
  suite_version: string
  model_id: string | null
  model_hash: string | null
  status: string
  comparability: string
  config: Json
  results: Json & { aggregate?: { score?: number; passed?: number; total?: number; metrics?: Json; judge?: Json }; cases?: Json[] }
  created_at: string
  finished_at: string | null
  cases?: Array<{ case_id: string; status: string; score: number; detail: Json & { detail?: string; prompt?: string; response?: string } }>
}

export interface Candidate {
  id: string
  parent_run_id: string
  dataset_id: string
  status: string
  config: Json & { patches?: string[]; system_prompt?: string; holdout_hash?: string; source_labels?: string[] }
  metrics: Json & { baseline_score?: number; candidate_score?: number; delta?: number; regressions?: string[]; critical_regressions?: string[] }
  decision: Json & { promoted?: boolean; reasons?: string[]; approver?: string }
  created_at: string
  decided_at: string | null
}

export interface AgentTask {
  id: string
  goal: string
  autonomy: string
  status: string
  plan: Array<{ tool: string; args: Json; mutating: boolean; description: string; requires_approval: boolean }>
  result: Json & { summary?: string; steps?: Array<Json>; plan?: Json }
  created_at: string
  finished_at: string | null
}

export interface HFModel {
  id: string
  author: string | null
  downloads: number
  likes: number
  tags: string[]
  pipeline_tag: string | null
  gated: boolean | string
  private: boolean
  last_modified: string | null
  library_name: string | null
  file_count?: number
}

export interface HFFile {
  path: string
  size: number
  sha256: string | null
  is_lfs: boolean
}

export interface TrainingEnvironment {
  state: 'unsupported' | 'cpu-experimental' | 'gpu'
  reason: string
  device: string
  gpu_name: string | null
  packages: Record<string, string | null>
  importable?: Record<string, boolean>
  import_errors?: Record<string, string>
  missing_packages: string[]
  torch: Json
  disk: { total_bytes: number; free_bytes: number } | null
  can_smoke: boolean
  can_finetune: boolean
  can_finetune_verified: boolean
  install_hint: string | null
}
