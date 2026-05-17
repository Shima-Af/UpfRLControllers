"""Pydantic request/response models for the dashboard API.

Typed contracts shared between FastAPI handlers and the React frontend.
v2's `/training` and `/experiments` endpoints will add models here next
to the existing ones — additive, not breaking.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PolicyId = Literal[
    "ppo", "random", "always-dpdk", "always-usr", "threshold", "hysteresis"
]

SplitId = Literal["train", "val", "test"]

# Multi-cluster controller identifiers (Phase 7).
MultiPolicyId = Literal[
    "mappo",
    "all-dpdk",
    "threshold-derived",
    "hysteresis-tuned",
    "hysteresis-auto",
]


class PolicyInfo(BaseModel):
    """One entry returned by GET /policies."""

    id: PolicyId
    label: str
    description: str
    requires_model: bool = Field(
        default=False,
        description="True if the policy can only be run when a trained model is loaded.",
    )
    model_loaded: bool = Field(
        default=False,
        description="True if the dashboard currently has a checkpoint that satisfies this policy.",
    )


class ClusterInfo(BaseModel):
    """One entry returned by GET /clusters."""

    cluster_idx: int
    episode_length: int
    horizon_count: int
    load_mean_gbps: float
    load_p95_gbps: float
    load_max_gbps: float


class RolloutRequest(BaseModel):
    cluster_idx: int = 0
    horizon_idx: int = 0
    policy: PolicyId = "ppo"
    threshold_gbps: float = Field(
        default=0.05,
        description="Predicted-load threshold for the 'threshold' policy. Ignored otherwise.",
    )
    hysteresis_band_mbps: float = Field(
        default=20.0,
        description="Hysteresis band in Mbps. Ignored unless policy='hysteresis'.",
    )
    hysteresis_cooldown_steps: int = Field(
        default=1,
        description="Hysteresis cooldown in steps. Ignored unless policy='hysteresis'.",
    )
    max_steps: int | None = Field(
        default=None,
        description="Cap the rollout at this many steps. Default: full episode.",
    )
    seed: int = 42
    split: SplitId = "test"


class StepRecord(BaseModel):
    """One time-step of a rollout — what the charts plot."""

    t: int
    action: int
    selected_upf: str
    actual_load_gbps: float
    predicted_load_gbps: float
    power_watts: float
    sec_w_per_mbps: float = 0.0
    delay_us: float
    predicted_loss: float
    q_score: float
    qos_penalty: float
    switching_energy_wh: float
    switch_penalty: float = 0.0
    cooldown_penalty: float = 0.0
    steps_since_switch: int = 0
    is_safe: bool
    energy_term: float = 0.0
    reward: float
    cumulative_reward: float


class RolloutSummary(BaseModel):
    """Aggregate metrics for a rollout — what the summary card shows."""

    policy: PolicyId
    label: str
    cluster_idx: int
    horizon_idx: int
    steps: int
    total_reward: float
    mean_reward: float
    total_energy_wh: float
    total_switch_wh: float
    unsafe_rate: float
    dpdk_rate: float
    usr_rate: float
    n_switches: int


class RolloutResponse(BaseModel):
    summary: RolloutSummary
    steps: list[StepRecord]


class CompareRequest(BaseModel):
    cluster_idx: int = 0
    horizon_idx: int = 0
    policies: list[PolicyId]
    threshold_gbps: float = 0.05
    hysteresis_band_mbps: float = 20.0
    hysteresis_cooldown_steps: int = 1
    max_steps: int | None = None
    seed: int = 42
    split: SplitId = "test"


class CompareResponse(BaseModel):
    cluster_idx: int
    horizon_idx: int
    rollouts: list[RolloutResponse]


# ---------------------------------------------------------------------------
# Multi-cluster (Phase 7) request/response models
# ---------------------------------------------------------------------------


class MultiPolicyInfo(BaseModel):
    """One entry returned by GET /multi/policies."""

    id: MultiPolicyId
    label: str
    description: str
    requires_model: bool = False
    model_loaded: bool = False


class MultiClusterStepRecord(BaseModel):
    """One step for one cluster within a multi-cluster rollout.

    Smaller than ``StepRecord`` (no `info` dict expansion) — the
    multi-cluster view plots aggregate KPIs, action heatmap, and
    per-cluster reward; full per-step diagnostics belong in the
    single-site view.
    """

    t: int
    action: int  # 0 = DPDK, 1 = USR
    load_gbps: float
    power_watts: float
    delay_us: float
    q_score: float
    reward: float
    is_safe: bool


class MultiClusterSummary(BaseModel):
    """Per-policy aggregate over all K clusters and all steps."""

    policy: MultiPolicyId
    label: str
    horizon_idx: int
    split: SplitId
    K: int
    steps: int
    total_reward_unweighted: float
    total_energy_wh: float
    agg_unsafe_rate: float
    agg_usr_rate: float
    agg_n_switches: int
    per_cluster_total_reward: list[float]
    per_cluster_energy_wh: list[float]
    per_cluster_unsafe_rate: list[float]
    per_cluster_usr_rate: list[float]
    per_cluster_n_switches: list[int]


class MultiClusterRolloutResponse(BaseModel):
    summary: MultiClusterSummary
    # actions[k][t] = 0/1, shape K x T
    actions: list[list[int]]
    # per_cluster_reward[k][t]
    per_cluster_reward: list[list[float]]
    # per_cluster_load[k][t]
    per_cluster_load: list[list[float]]
    # cumulative aggregate reward over the episode
    cumulative_reward: list[float]


class MultiRolloutRequest(BaseModel):
    policy: MultiPolicyId = "mappo"
    horizon_idx: int = 0
    split: SplitId = "test"
    seed: int = 42
    max_steps: int | None = None
    # Tuning parameters for stateless / classical baselines (ignored
    # for the trained MAPPO controller).
    threshold_gbps: float = Field(
        default=0.081,
        description="Decision threshold for the 'threshold-derived' baseline.",
    )
    hysteresis_band_mbps: float = Field(
        default=20.0,
        description="Hysteresis band for the 'hysteresis-tuned' baseline.",
    )
    hysteresis_cooldown_steps: int = Field(default=1)


class MultiCompareRequest(BaseModel):
    policies: list[MultiPolicyId] = Field(default_factory=list)
    horizon_idx: int = 0
    split: SplitId = "test"
    seed: int = 42
    max_steps: int | None = None
    threshold_gbps: float = 0.081
    hysteresis_band_mbps: float = 20.0
    hysteresis_cooldown_steps: int = 1


class MultiCompareResponse(BaseModel):
    horizon_idx: int
    split: SplitId
    K: int
    rollouts: list[MultiClusterRolloutResponse]
