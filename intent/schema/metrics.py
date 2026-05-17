"""Metric vocabulary for intent predicates.

Every name here must map to a value the twin replay produces. The
`normalize_metrics` helper rewrites `rollout_episode`'s output into
this vocabulary (with `_pct` rescaling and friendlier aliases).
"""

from __future__ import annotations

from typing import Literal

MetricName = Literal[
    "total_reward",
    "mean_reward",
    "energy_wh",          # alias for rollout_episode.total_energy_wh
    "switch_wh",          # alias for rollout_episode.total_switch_wh
    "mean_sec",
    "mean_q",
    "qos_violation_pct",  # rollout_episode.qos_violation_rate * 100
    "unsafe_pct",         # rollout_episode.unsafe_rate * 100
    "dpdk_rate",
    "usr_rate",
    "flips",              # alias for rollout_episode.n_switches
    "steps",
]

METRIC_VOCAB: tuple[str, ...] = (
    "total_reward",
    "mean_reward",
    "energy_wh",
    "switch_wh",
    "mean_sec",
    "mean_q",
    "qos_violation_pct",
    "unsafe_pct",
    "dpdk_rate",
    "usr_rate",
    "flips",
    "steps",
)


def normalize_metrics(raw: dict[str, float]) -> dict[str, float]:
    """Map `src.trainers.ppo_single_site.rollout_episode` output to the
    canonical predicate vocabulary."""
    return {
        "total_reward": float(raw["total_reward"]),
        "mean_reward": float(raw["mean_reward"]),
        "energy_wh": float(raw["total_energy_wh"]),
        "switch_wh": float(raw["total_switch_wh"]),
        "mean_sec": float(raw["mean_sec"]),
        "mean_q": float(raw["mean_q"]),
        "qos_violation_pct": float(raw["qos_violation_rate"]) * 100.0,
        "unsafe_pct": float(raw["unsafe_rate"]) * 100.0,
        "dpdk_rate": float(raw["dpdk_rate"]),
        "usr_rate": float(raw["usr_rate"]),
        "flips": float(raw["n_switches"]),
        "steps": float(raw["steps"]),
    }
