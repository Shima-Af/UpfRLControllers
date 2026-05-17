// TypeScript mirrors of the Pydantic models in dashboard/backend/app/schemas.py.
// Keep in sync when adding fields.

export type PolicyId =
  | "ppo"
  | "random"
  | "always-dpdk"
  | "always-usr"
  | "threshold"
  | "hysteresis";

export type SplitId = "train" | "val" | "test";

export type MultiPolicyId =
  | "mappo"
  | "all-dpdk"
  | "threshold-derived"
  | "hysteresis-tuned"
  | "hysteresis-auto";

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
  hysteresis_band_mbps?: number;
  hysteresis_cooldown_steps?: number;
  max_steps?: number | null;
  seed?: number;
  split?: SplitId;
}

export interface CompareResponse {
  cluster_idx: number;
  horizon_idx: number;
  rollouts: RolloutResponse[];
}

// ---------------------------------------------------------------------------
// Multi-cluster (Phase 7) types
// ---------------------------------------------------------------------------

export interface MultiPolicyInfo {
  id: MultiPolicyId;
  label: string;
  description: string;
  requires_model: boolean;
  model_loaded: boolean;
}

export interface MultiClusterSummary {
  policy: MultiPolicyId;
  label: string;
  horizon_idx: number;
  split: SplitId;
  K: number;
  steps: number;
  total_reward_unweighted: number;
  total_energy_wh: number;
  agg_unsafe_rate: number;
  agg_usr_rate: number;
  agg_n_switches: number;
  per_cluster_total_reward: number[];
  per_cluster_energy_wh: number[];
  per_cluster_unsafe_rate: number[];
  per_cluster_usr_rate: number[];
  per_cluster_n_switches: number[];
}

export interface MultiClusterRolloutResponse {
  summary: MultiClusterSummary;
  actions: number[][];               // K x T (0 = DPDK, 1 = USR)
  per_cluster_reward: number[][];    // K x T
  per_cluster_load: number[][];      // K x T
  cumulative_reward: number[];       // T
}

export interface MultiRolloutRequest {
  policy: MultiPolicyId;
  horizon_idx?: number;
  split?: SplitId;
  seed?: number;
  max_steps?: number | null;
  threshold_gbps?: number;
  hysteresis_band_mbps?: number;
  hysteresis_cooldown_steps?: number;
}

export interface MultiCompareRequest {
  policies: MultiPolicyId[];
  horizon_idx?: number;
  split?: SplitId;
  seed?: number;
  max_steps?: number | null;
  threshold_gbps?: number;
  hysteresis_band_mbps?: number;
  hysteresis_cooldown_steps?: number;
}

export interface MultiCompareResponse {
  horizon_idx: number;
  split: SplitId;
  K: number;
  rollouts: MultiClusterRolloutResponse[];
}

export interface DerivedThresholds {
  decision_gbps: number;
  t_up_gbps_auto: number;
  t_down_gbps_auto: number;
  hysteresis_band_gbps_auto: number;
  forecast_mae_gbps: number;
  energy_breakeven_gbps: number;
  qos_limit_gbps: number;
  derived_from: string;
}
