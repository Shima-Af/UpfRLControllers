// TypeScript mirrors of the Pydantic models in dashboard/backend/app/schemas.py.
// Keep in sync when adding fields.

export type PolicyId =
  | "ppo"
  | "random"
  | "always-dpdk"
  | "always-usr"
  | "threshold";

export interface PolicyInfo {
  id: PolicyId;
  label: string;
  description: string;
  requires_model: boolean;
  model_loaded: boolean;
}

export interface ClusterInfo {
  cluster_idx: number;
  episode_length: number;
  horizon_count: number;
  load_mean_gbps: number;
  load_p95_gbps: number;
  load_max_gbps: number;
}

export interface StepRecord {
  t: number;
  action: number;
  selected_upf: string;
  actual_load_gbps: number;
  predicted_load_gbps: number;
  power_watts: number;
  sec_w_per_mbps: number;
  delay_us: number;
  predicted_loss: number;
  q_score: number;
  qos_penalty: number;
  switching_energy_wh: number;
  switch_penalty: number;
  cooldown_penalty: number;
  steps_since_switch: number;
  is_safe: boolean;
  energy_term: number;
  reward: number;
  cumulative_reward: number;
}

export interface RolloutSummary {
  policy: PolicyId;
  label: string;
  cluster_idx: number;
  horizon_idx: number;
  steps: number;
  total_reward: number;
  mean_reward: number;
  total_energy_wh: number;
  total_switch_wh: number;
  unsafe_rate: number;
  dpdk_rate: number;
  usr_rate: number;
  n_switches: number;
}

export interface RolloutResponse {
  summary: RolloutSummary;
  steps: StepRecord[];
}

export interface CompareRequest {
  cluster_idx: number;
  horizon_idx: number;
  policies: PolicyId[];
  threshold_gbps?: number;
  max_steps?: number | null;
  seed?: number;
}

export interface CompareResponse {
  cluster_idx: number;
  horizon_idx: number;
  rollouts: RolloutResponse[];
}
