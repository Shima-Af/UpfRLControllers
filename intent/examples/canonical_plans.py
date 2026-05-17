"""Three hand-written canonical Plans used by Slice 0 to exercise the
loop without an LLM in front. Once Slice 1 lands, these double as the
gold answers for the intent-paraphrase parse-accuracy benchmark.
"""

from __future__ import annotations

from intent.schema.plan import HardConstraint, Plan

CANONICAL_PLANS: dict[str, Plan] = {
    "energy_greedy": Plan(
        goal="minimize_energy",
        hard=[
            HardConstraint(metric="unsafe_pct", op="<", threshold=2.0),
            HardConstraint(metric="energy_wh", op="<", threshold=1990.0),
        ],
        soft=["low_flip_count", "prefer_usr_when_load_low"],
        horizon="full_episode",
        context_tag="off-peak: aggressively save energy, accept mild QoS slack",
    ),
    "safety_first": Plan(
        goal="minimize_unsafe",
        hard=[
            HardConstraint(metric="unsafe_pct", op="<", threshold=0.4),
            HardConstraint(metric="mean_q", op=">", threshold=0.95),
        ],
        soft=["maintain_qos_headroom"],
        horizon="full_episode",
        context_tag="peak load: never breach QoS",
    ),
    "balanced": Plan(
        goal="balance",
        hard=[
            HardConstraint(metric="unsafe_pct", op="<", threshold=1.0),
        ],
        soft=[],
        horizon="full_episode",
        context_tag="paper-aligned default operating point",
    ),
}
