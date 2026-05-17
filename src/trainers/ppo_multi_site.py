"""Stable-Baselines3 PPO trainer for MultiSiteUPFEnv (Phase 3).

Same skeleton as the single-site trainer: feed-forward MLP policy
(SB3 handles MultiDiscrete natively, factored per-component logits),
EvalCallback against the validation slice, best-model checkpointing.

Hyperparameters default to the Phase-2 paper values
(``ent_coef=0.15``, ``learning_rate=1e-4``, ``n_steps=1024``,
``gamma=0.995``, ``gae_lambda=0.9``). Two changes were considered but
not adopted in the default: a wider net (the obs is 10× larger) and
a higher entropy floor (joint-action exploration is harder). Both
left to the CLI for sweep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.multi_site_upf_env import MultiSiteUPFEnv


def _make_env(
    horizon_idx: int,
    seed: int,
    split: str,
    cluster_indices: list[int] | None,
) -> Callable:
    def _thunk() -> Monitor:
        env = MultiSiteUPFEnv(
            cluster_indices=cluster_indices,
            horizon_idx=horizon_idx,
            split=split,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _thunk


def train_ppo_multi_site(
    horizon_idx: int = 0,
    total_timesteps: int = 200_000,
    seed: int = 42,
    out_dir: str | Path | None = None,
    *,
    cluster_indices: list[int] | None = None,
    train_split: str = "train",
    eval_split: str = "val",
    learning_rate: float = 1e-4,
    n_steps: int = 1024,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.995,
    gae_lambda: float = 0.9,
    clip_range: float = 0.2,
    ent_coef: float = 0.15,
    vf_coef: float = 0.5,
    verbose: int = 1,
    progress_bar: bool = True,
    tensorboard_log: str | Path | None = None,
    eval_freq: int = 4096,
) -> PPO:
    """Train centralised PPO on MultiSiteUPFEnv.

    Returns the best-of-val model (selected by EvalCallback). The
    final-step checkpoint is saved alongside as
    ``ppo_multi_site_final.zip``.

    Split semantics mirror Phase 2: ``train_split`` drives the rollout
    buffer; ``eval_split`` drives best-model selection; the test
    slice is reserved for the headline evaluation in
    ``scripts/evaluate_multi_site_test.py``.
    """
    env = DummyVecEnv(
        [_make_env(horizon_idx, seed, train_split, cluster_indices)]
    )
    eval_env = DummyVecEnv(
        [_make_env(horizon_idx, seed + 1, eval_split, cluster_indices)]
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=verbose,
        seed=seed,
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
    )

    callbacks = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        callbacks = EvalCallback(
            eval_env,
            best_model_save_path=str(out_dir),
            log_path=str(out_dir),
            eval_freq=eval_freq,
            n_eval_episodes=1,
            deterministic=True,
            verbose=0,
        )

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=progress_bar,
        callback=callbacks,
    )

    if out_dir is not None:
        model.save(out_dir / "ppo_multi_site_final")
        best_path = out_dir / "best_model.zip"
        if best_path.exists():
            model = PPO.load(best_path)
            model.save(out_dir / "ppo_multi_site")
        else:
            model.save(out_dir / "ppo_multi_site")

    return model


# ----------------------------------------------------------------------
# Policy adapters and rollout helper
# ----------------------------------------------------------------------

MultiPolicyFn = Callable[[np.ndarray], np.ndarray]


def ppo_multi_policy(model: PPO, deterministic: bool = True) -> MultiPolicyFn:
    def _fn(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.int64).reshape(-1)

    return _fn


def constant_multi_policy(action_id: int, K: int) -> MultiPolicyFn:
    """All clusters take the same constant action every step."""
    arr = np.full(K, int(action_id), dtype=np.int64)

    def _fn(_obs: np.ndarray) -> np.ndarray:
        return arr.copy()

    return _fn


def random_multi_policy(seed: int, K: int) -> MultiPolicyFn:
    rng = np.random.default_rng(seed)

    def _fn(_obs: np.ndarray) -> np.ndarray:
        return rng.integers(0, 2, size=K).astype(np.int64)

    return _fn


def predicted_load_threshold_multi_policy(
    threshold_gbps: float, K: int, obs_dim_per_cluster: int = 14,
    forecast_idx_within_cluster: int = 9,
) -> MultiPolicyFn:
    """Per-cluster threshold rule applied independently to each cluster.

    Reads each cluster's forecast slot from the concatenated obs;
    ``forecast_idx_within_cluster`` defaults to 9 — the index of the
    1-step forecast within the single-site obs schema (current load
    + 8 history + forecast).
    """

    def _fn(obs: np.ndarray) -> np.ndarray:
        out = np.zeros(K, dtype=np.int64)
        for k in range(K):
            base = k * obs_dim_per_cluster
            forecast = float(obs[base + forecast_idx_within_cluster])
            out[k] = 1 if forecast < threshold_gbps else 0
        return out

    return _fn


def rollout_multi_episode(
    policy: MultiPolicyFn,
    *,
    horizon_idx: int = 0,
    seed: int = 0,
    split: str = "test",
    cluster_indices: list[int] | None = None,
    max_steps: int | None = None,
) -> dict:
    """Run one full episode of the multi-site env under ``policy``.

    Returns aggregate KPIs plus per-cluster breakdowns.
    """
    env = MultiSiteUPFEnv(
        cluster_indices=cluster_indices,
        horizon_idx=horizon_idx,
        split=split,
    )
    obs, _info = env.reset(seed=seed)
    K = env.K
    step_h = env.step_h
    tau = env.tau

    total_weighted_reward = 0.0
    total_unweighted_reward = 0.0
    per_cluster_total_reward = np.zeros(K, dtype=np.float64)
    per_cluster_energy_wh = np.zeros(K, dtype=np.float64)
    per_cluster_unsafe = np.zeros(K, dtype=np.int64)
    per_cluster_qos_violation = np.zeros(K, dtype=np.int64)
    per_cluster_dpdk = np.zeros(K, dtype=np.int64)
    per_cluster_usr = np.zeros(K, dtype=np.int64)
    per_cluster_switches = np.zeros(K, dtype=np.int64)
    per_cluster_last_action: list[int | None] = [None] * K
    steps = 0

    while True:
        action = policy(obs)
        obs, r, term, trunc, info = env.step(action)
        total_weighted_reward += r
        total_unweighted_reward += info["sum_reward_unweighted"]
        for k, ck in enumerate(info["per_cluster"]):
            per_cluster_total_reward[k] += info["per_cluster_reward"][k]
            per_cluster_energy_wh[k] += float(ck["power_watts"]) * step_h
            if not ck["is_safe"]:
                per_cluster_unsafe[k] += 1
            if ck["q_score"] < tau:
                per_cluster_qos_violation[k] += 1
            if ck["selected_upf"] == "DPDK":
                per_cluster_dpdk[k] += 1
            else:
                per_cluster_usr[k] += 1
            a_k = int(action[k])
            if (
                per_cluster_last_action[k] is not None
                and per_cluster_last_action[k] != a_k
            ):
                per_cluster_switches[k] += 1
            per_cluster_last_action[k] = a_k
        steps += 1
        if term or trunc:
            break
        if max_steps is not None and steps >= max_steps:
            break

    n = max(1, steps)
    return {
        "K": K,
        "steps": steps,
        "split": split,
        "total_weighted_reward": total_weighted_reward,
        "total_unweighted_reward": total_unweighted_reward,
        "mean_weighted_reward": total_weighted_reward / n,
        "total_energy_wh": float(per_cluster_energy_wh.sum()),
        "per_cluster_total_reward": per_cluster_total_reward.tolist(),
        "per_cluster_energy_wh": per_cluster_energy_wh.tolist(),
        "per_cluster_unsafe_rate": (per_cluster_unsafe / n).tolist(),
        "per_cluster_qos_violation_rate": (per_cluster_qos_violation / n).tolist(),
        "per_cluster_dpdk_rate": (per_cluster_dpdk / n).tolist(),
        "per_cluster_usr_rate": (per_cluster_usr / n).tolist(),
        "per_cluster_n_switches": per_cluster_switches.tolist(),
        "agg_unsafe_rate": float(per_cluster_unsafe.sum() / (K * n)),
        "agg_usr_rate": float(per_cluster_usr.sum() / (K * n)),
        "agg_n_switches": int(per_cluster_switches.sum()),
    }
