"""Replay a trained PPO checkpoint on the digital twin and return a
metrics dict in the canonical predicate vocabulary."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO

from intent.schema.metrics import normalize_metrics
from src.trainers.ppo_single_site import ppo_policy, rollout_episode


def replay(
    checkpoint_path: str | Path,
    *,
    cluster_idx: int = 0,
    split: str = "val",
    seed: int = 42,
) -> dict[str, float]:
    """One full-episode rollout with the deterministic PPO policy.

    `split` defaults to "val" because this helper is for the inner
    intent loop (cheap, repeatable). The test slice is reserved for
    final reported numbers, same convention as Phase 2.
    """
    model = PPO.load(str(checkpoint_path))
    pol = ppo_policy(model, deterministic=True)
    raw = rollout_episode(
        cluster_idx=cluster_idx,
        horizon_idx=0,
        policy=pol,
        seed=seed,
        split=split,
    )
    return normalize_metrics(raw)
