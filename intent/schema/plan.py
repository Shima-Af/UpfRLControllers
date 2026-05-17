"""Plan — the structured intermediate the LLM emits (Slice 0: hand-written).

The Plan is deliberately auditable: a closed `goal`, a list of hard
constraints over the metric vocabulary, a closed list of soft
preferences, and free-text context. The Plan → weights / Plan →
predicate mapping lives in `intent.compile`, kept deterministic so a
human reviewer can replay the LLM's translation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from intent.schema.metrics import MetricName

Goal = Literal["minimize_energy", "minimize_unsafe", "balance"]

# `off_peak`/`peak` are advisory only in Slice 0 — they're stored but
# don't yet drive a time-conditioned reward. Slice 2+ may wire them in.
Horizon = Literal["full_episode", "off_peak", "peak"]

SoftPreference = Literal[
    "prefer_usr_when_load_low",
    "prefer_dpdk_when_load_high",
    "low_flip_count",
    "high_flip_count_ok",
    "maintain_qos_headroom",
    "minimize_oscillation",
]

ComparisonOp = Literal["<", "<=", ">", ">=", "==", "!="]


class HardConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: MetricName
    op: ComparisonOp
    threshold: float


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: Goal
    hard: list[HardConstraint] = Field(default_factory=list)
    soft: list[SoftPreference] = Field(default_factory=list)
    horizon: Horizon = "full_episode"
    context_tag: str = ""
