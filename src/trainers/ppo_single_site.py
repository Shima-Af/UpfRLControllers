"""Stable-Baselines3 PPO trainer for SingleSiteUPFEnv.

Phase 2 sanity check: train a feed-forward PPO policy on one cluster and
compare it against three baselines (random, always-DPDK, always-USR).
Intentionally minimal — no LSTM, no VecNormalize, no curriculum. Once
this shows learning signal we can build the proper training stack on top.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.single_site_upf_env import SingleSiteUPFEnv


def _make_env(cluster_idx: int, horizon_idx: int, seed: int) -> Callable:
    def _thunk() -> Monitor:
        env = SingleSiteUPFEnv(cluster_idx=cluster_idx, horizon_idx=horizon_idx)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _thunk


def train_ppo_single_site(
    cluster_idx: int = 0,
    horizon_idx: int = 0,
    total_timesteps: int = 50_000,
    seed: int = 42,
    out_dir: str | Path | None = None,
    *,
    learning_rate: float = 1e-4,
    n_steps: int = 1024,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.995,
    gae_lambda: float = 0.9,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    verbose: int = 1,
    progress_bar: bool = True,
    tensorboard_log: str | Path | None = None,
) -> PPO:
    """Train PPO on one cluster of SingleSiteUPFEnv. Returns the model."""
    env = DummyVecEnv([_make_env(cluster_idx, horizon_idx, seed)])

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

    model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save(out_dir / "ppo_single_site")

    return model


# ----------------------------------------------------------------------
# Policy adapters for evaluation
# ----------------------------------------------------------------------

PolicyFn = Callable[[np.ndarray], int]


def ppo_policy(model: PPO, deterministic: bool = True) -> PolicyFn:
    def _fn(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=deterministic)
        return int(np.asarray(action).item())

    return _fn


def constant_policy(action: int) -> PolicyFn:
    def _fn(_obs: np.ndarray) -> int:
        return int(action)

    return _fn


def random_policy(seed: int = 0) -> PolicyFn:
    rng = np.random.default_rng(seed)

    def _fn(_obs: np.ndarray) -> int:
        return int(rng.integers(0, 2))

    return _fn


def predicted_load_threshold_policy(threshold_gbps: float) -> PolicyFn:
    """USR (1) when ``predicted_load_gbps`` < threshold, else DPDK (0).

    Reads ``obs[1]``, which the env defines as the forecaster's load
    estimate for the current decision step. Mirrors the structure of the
    threshold baselines used in prior in-house work.
    """

    def _fn(obs: np.ndarray) -> int:
        predicted = float(obs[1])
        return 1 if predicted < threshold_gbps else 0

    return _fn


# ----------------------------------------------------------------------
# Single-episode rollout used to compare policies
# ----------------------------------------------------------------------


def rollout_episode(
    cluster_idx: int,
    horizon_idx: int,
    policy: PolicyFn,
    seed: int = 0,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Run one full episode under ``policy`` and report summary metrics.

    ``max_steps`` truncates the rollout early — useful when the digital
    twin's per-step cost makes a full 1009-step episode expensive.
    """
    env = SingleSiteUPFEnv(cluster_idx=cluster_idx, horizon_idx=horizon_idx)
    obs, _info = env.reset(seed=seed)

    total_reward = 0.0
    total_energy_wh = 0.0
    total_switch_wh = 0.0
    n_unsafe = 0
    n_dpdk = 0
    n_usr = 0
    n_switches = 0
    last_action: int | None = None
    steps = 0

    step_h = env._step_h  # noqa: SLF001 — read-only access, internal by design

    while True:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_energy_wh += float(info["power_watts"]) * step_h
        total_switch_wh += float(info["switching_energy_wh"])
        if not info["is_safe"]:
            n_unsafe += 1
        if info["selected_upf"] == "DPDK":
            n_dpdk += 1
        else:
            n_usr += 1
        if last_action is not None and last_action != action:
            n_switches += 1
        last_action = action
        steps += 1
        if terminated or truncated:
            break
        if max_steps is not None and steps >= max_steps:
            break

    return {
        "total_reward": total_reward,
        "mean_reward": total_reward / max(1, steps),
        "total_energy_wh": total_energy_wh,
        "total_switch_wh": total_switch_wh,
        "unsafe_rate": n_unsafe / max(1, steps),
        "dpdk_rate": n_dpdk / max(1, steps),
        "usr_rate": n_usr / max(1, steps),
        "n_switches": n_switches,
        "steps": steps,
    }
