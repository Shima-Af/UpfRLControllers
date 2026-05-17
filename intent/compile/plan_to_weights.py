"""Deterministic Plan -> reward_weights compiler.

The mapping is a table, not a model: each `goal` sets a baseline
shift relative to the paper defaults; each soft preference applies a
named tweak. Override order is `BASE -> goal delta -> soft deltas (in
list order)` — later wins. This is intentionally simple so the
LLM-emitted Plan can be replayed by a human reading this file.
"""

from __future__ import annotations

from intent.schema.plan import Goal, Plan, SoftPreference

# Paper-aligned defaults — match `configs/scenario_rl.yaml::reward_weights`.
# Keep `c_dpdk`/`c_usr` at 0.0: switching cost is folded into SEC via the
# twin's `sw_energy_wh` (see env docstring). They're still emitted so the
# scenario_cfg override path is complete.
BASE: dict[str, float | int] = {
    "alpha":           100.0,
    "lambda_qos":      30.0,
    "tau":             0.90,
    "c_dpdk":          0.0,
    "c_usr":           0.0,
    "cooldown_period": 4,
    "cooldown_cost":   0.5,
}

GOAL_DELTAS: dict[Goal, dict[str, float | int]] = {
    "minimize_energy": {"alpha": 200.0, "lambda_qos": 20.0, "tau": 0.85},
    "minimize_unsafe": {"alpha":  50.0, "lambda_qos": 60.0, "tau": 0.95},
    "balance":         {},
}

SOFT_DELTAS: dict[SoftPreference, dict[str, float | int]] = {
    "low_flip_count":            {"cooldown_cost": 1.0},
    "high_flip_count_ok":        {"cooldown_cost": 0.1},
    "minimize_oscillation":      {"cooldown_cost": 1.0, "cooldown_period": 6},
    "maintain_qos_headroom":     {"lambda_qos": 60.0, "tau": 0.95},
    # The two `prefer_*` flags are advisory for now — they would map to a
    # load-conditioned reward shape, which Slice 0 doesn't implement.
    "prefer_usr_when_load_low":  {},
    "prefer_dpdk_when_load_high": {},
}


def compile_weights(plan: Plan) -> dict[str, float | int]:
    """Return a `reward_weights` dict suitable for overriding
    `scenario_cfg["reward_weights"]` before env construction."""
    w: dict[str, float | int] = dict(BASE)
    w.update(GOAL_DELTAS[plan.goal])
    for s in plan.soft:
        w.update(SOFT_DELTAS[s])
    return w
