"""Pydantic request/response models for the dashboard API.

Typed contracts shared between FastAPI handlers and the React frontend.
v2's `/training` and `/experiments` endpoints will add models here next
to the existing ones — additive, not breaking.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PolicyId = Literal["ppo", "random", "always-dpdk", "always-usr", "threshold"]


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
    max_steps: int | None = Field(
        default=None,
        description="Cap the rollout at this many steps. Default: full episode.",
    )
    seed: int = 42


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
    max_steps: int | None = None
    seed: int = 42


class CompareResponse(BaseModel):
    cluster_idx: int
    horizon_idx: int
    rollouts: list[RolloutResponse]
