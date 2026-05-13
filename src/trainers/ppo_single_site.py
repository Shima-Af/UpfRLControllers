"""Stable-Baselines3 PPO trainer for SingleSiteUPFEnv.

Trains a feed-forward PPO controller aligned with the paper (Section
4.6). Adds best-model checkpointing via SB3's ``EvalCallback`` so a
late-stage policy collapse during training does not corrupt the final
deployed policy — the paper documents this collapse and we use the
same defence.

Intentionally minimal otherwise: no LSTM, no VecNormalize, no
curriculum, no per-cluster K=10 sweep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
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
    ent_coef: float = 0.15,
    vf_coef: float = 0.5,
    verbose: int = 1,
    progress_bar: bool = True,
    tensorboard_log: str | Path | None = None,
    eval_freq: int = 4096,
) -> PPO:
    """Train PPO on one cluster of SingleSiteUPFEnv.

    Returns the best model encountered during training (selected by
    deterministic eval-env episode return). The final-step checkpoint
    is also saved as ``ppo_single_site_final.zip`` for completeness.

    Defaults follow the paper (Section 6.2 / Table 3): ent_coef=0.15,
    n_steps=1024, gae_lambda=0.9, gamma=0.995. ``learning_rate=1e-4``
    is the paper's choice; the CLI exposes it for sweeps.
    """
    env = DummyVecEnv([_make_env(cluster_idx, horizon_idx, seed)])
    eval_env = DummyVecEnv([_make_env(cluster_idx, horizon_idx, seed + 1)])

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
        # Persist the final-step model alongside the best checkpoint
        # so we can compare "final" vs "best" externally if needed.
        model.save(out_dir / "ppo_single_site_final")
        best_path = out_dir / "best_model.zip"
        if best_path.exists():
            model = PPO.load(best_path)
            # Convenience copy at a stable filename for the dashboard.
            model.save(out_dir / "ppo_single_site")
        else:
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

    ``max_steps`` truncates the rollout early.
    """
    env = SingleSiteUPFEnv(cluster_idx=cluster_idx, horizon_idx=horizon_idx)
    obs, _info = env.reset(seed=seed)

    total_reward = 0.0
    total_energy_wh = 0.0
    total_switch_wh = 0.0
    sec_sum = 0.0
    q_sum = 0.0
    n_unsafe = 0
    n_qos_violation = 0  # paper-aligned: Q < tau
    n_dpdk = 0
    n_usr = 0
    n_switches = 0
    last_action: int | None = None
    steps = 0
    tau = env._tau  # noqa: SLF001 — for the violation-rate metric

    step_h = env._step_h  # noqa: SLF001

    while True:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_energy_wh += float(info["power_watts"]) * step_h
        total_switch_wh += float(info["switching_energy_wh"])
        sec_sum += float(info["sec_w_per_mbps"])
        q_sum += float(info["q_score"])
        if not info["is_safe"]:
            n_unsafe += 1
        if info["q_score"] < tau:
            n_qos_violation += 1
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

    n = max(1, steps)
    return {
        "total_reward": total_reward,
        "mean_reward": total_reward / n,
        "total_energy_wh": total_energy_wh,
        "total_switch_wh": total_switch_wh,
        "mean_sec": sec_sum / n,
        "mean_q": q_sum / n,
        "qos_violation_rate": n_qos_violation / n,
        "unsafe_rate": n_unsafe / n,
        "dpdk_rate": n_dpdk / n,
        "usr_rate": n_usr / n,
        "n_switches": n_switches,
        "steps": steps,
    }
