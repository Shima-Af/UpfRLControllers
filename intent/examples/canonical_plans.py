"""Three hand-written canonical Plans used by Slice 0 to exercise the
loop without an LLM in front. Once Slice 1 lands, these double as the
gold answers for the intent-paraphrase parse-accuracy benchmark.

Slice 0 Plans intentionally have no acceptance criteria — only a
goal and soft preferences. The fine-tune effect is judged by
inspecting the metrics in the CLI output (energy_wh, unsafe_pct,
flips, usr_rate, ...), not by a predicate verifier. Thresholds /
predicates come back in Slice 2 once the LLM is in the loop and can
emit them grounded in observed numbers.
"""

from __future__ import annotations

from intent.schema.plan import Plan

CANONICAL_PLANS: dict[str, Plan] = {
    "energy_greedy": Plan(
        goal="minimize_energy",
        soft=["low_flip_count", "prefer_usr_when_load_low"],
        horizon="full_episode",
        context_tag="off-peak: aggressively save energy, accept mild QoS slack",
    ),
    "safety_first": Plan(
        goal="minimize_unsafe",
        soft=["maintain_qos_headroom"],
        horizon="full_episode",
        context_tag="peak load: never breach QoS",
    ),
    "balanced": Plan(
        goal="balance",
        soft=[],
        horizon="full_episode",
        context_tag="paper-aligned default operating point",
    ),
}
