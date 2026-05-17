"""Plan — the structured intermediate the LLM emits (Slice 0: hand-written).

Slice 0 keeps the Plan deliberately small: a closed `goal` (the
optimisation direction), a closed list of soft preferences, an
advisory `horizon` tag, and free-text context. The Plan → weights
mapping in `intent.compile.plan_to_weights` is the only consumer.

Acceptance-criterion fields (hard constraints / thresholds) were
considered for Slice 0 but pulled out — they belong with Slice 2's
LLM verifier, where the LLM can emit thresholds grounded in observed
performance instead of guessed-in-advance numbers. The Predicate DSL
is still defined in `intent/schema/predicate.py` for that future use.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Goal = Literal["minimize_energy", "minimize_unsafe", "balance"]

# `off_peak` / `peak` are advisory only in Slice 0 — they're stored but
# don't yet drive a time-conditioned reward. A later slice may wire
# them in.
Horizon = Literal["full_episode", "off_peak", "peak"]

SoftPreference = Literal[
    "prefer_usr_when_load_low",
    "prefer_dpdk_when_load_high",
    "low_flip_count",
    "high_flip_count_ok",
    "maintain_qos_headroom",
    "minimize_oscillation",
]


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: Goal
    soft: list[SoftPreference] = Field(default_factory=list)
    horizon: Horizon = "full_episode"
    context_tag: str = ""
