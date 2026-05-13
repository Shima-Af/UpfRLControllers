"""Policy registry + rollout helper, decoupled from FastAPI handlers.

Adapts the trainer-side ``PolicyFn`` adapters to the dashboard's needs:
a uniform ``run_rollout(policy_id, ...) -> RolloutResponse`` that the
HTTP layer just serializes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO

from src.envs.single_site_upf_env import SingleSiteUPFEnv
from src.trainers.ppo_single_site import (
    constant_policy,
    ppo_policy,
    predicted_load_threshold_policy,
    random_policy,
)

from .schemas import (
    PolicyId,
    PolicyInfo,
    RolloutResponse,
    RolloutSummary,
    StepRecord,
)


# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------

POLICY_LABELS: dict[PolicyId, str] = {
    "ppo": "PPO (trained)",
    "random": "Random",
    "always-dpdk": "Always DPDK",
    "always-usr": "Always USR",
    "threshold": "Threshold (USR<x)",
}

POLICY_DESCRIPTIONS: dict[PolicyId, str] = {
    "ppo": "Trained PPO checkpoint loaded from disk; deterministic argmax.",
    "random": "Uniform random over {DPDK, USR}, seeded.",
    "always-dpdk": "Always DPDK — the safe baseline.",
    "always-usr": "Always USR — the energy-greedy baseline.",
    "threshold": "USR if predicted_load < threshold_gbps, else DPDK.",
}


def _latest_ppo_checkpoint(repo_root: Path) -> Path | None:
    exp_dir = repo_root / "experiments"
    if not exp_dir.is_dir():
        return None
    candidates = sorted(
        exp_dir.glob("ppo_single_site_*/ppo_single_site.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


class PolicyRegistry:
    """Single source of truth for which policies are available.

    The PPO model is loaded lazily on first use and cached. Other
    policies are stateless and constructed per-call.
    """

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self._ppo_model: PPO | None = None
        self._ppo_path = _latest_ppo_checkpoint(repo_root)

    def list_policies(self) -> list[PolicyInfo]:
        return [
            PolicyInfo(
                id=pid,
                label=POLICY_LABELS[pid],
                description=POLICY_DESCRIPTIONS[pid],
                requires_model=(pid == "ppo"),
                model_loaded=(pid != "ppo") or (self._ppo_path is not None),
            )
            for pid in POLICY_LABELS
        ]

    def get_policy(
        self,
        policy_id: PolicyId,
        *,
        seed: int,
        threshold_gbps: float,
    ) -> Callable[[np.ndarray], int]:
        if policy_id == "ppo":
            if self._ppo_path is None:
                raise FileNotFoundError(
                    "No PPO checkpoint found under experiments/. "
                    "Train one first with scripts/train_ppo_single_site.py."
                )
            if self._ppo_model is None:
                self._ppo_model = PPO.load(self._ppo_path)
            return ppo_policy(self._ppo_model, deterministic=True)
        if policy_id == "random":
            return random_policy(seed=seed)
        if policy_id == "always-dpdk":
            return constant_policy(0)
        if policy_id == "always-usr":
            return constant_policy(1)
        if policy_id == "threshold":
            return predicted_load_threshold_policy(threshold_gbps)
        raise ValueError(f"Unknown policy_id: {policy_id}")


# ---------------------------------------------------------------------------
# Rollout helper that returns the dashboard's response shape directly
# ---------------------------------------------------------------------------


def run_rollout(
    registry: PolicyRegistry,
    *,
    policy_id: PolicyId,
    cluster_idx: int,
    horizon_idx: int,
    seed: int,
    threshold_gbps: float,
    max_steps: int | None,
) -> RolloutResponse:
    """Execute one episode and return the structured response.

    The env is constructed fresh per request so cluster_idx/horizon_idx
    changes work without state. Cost is dominated by the env's
    one-time surrogate batch precompute (~1.3s); the per-step lookup
    that follows is microseconds.
    """
    env = SingleSiteUPFEnv(cluster_idx=cluster_idx, horizon_idx=horizon_idx)
    obs, _info = env.reset(seed=seed)
    policy = registry.get_policy(
        policy_id, seed=seed, threshold_gbps=threshold_gbps
    )

    step_h = env._step_h  # noqa: SLF001 — read-only accessor
    records: list[StepRecord] = []
    cum_reward = 0.0
    n_dpdk = n_usr = n_unsafe = n_switches = 0
    last_action: int | None = None
    total_energy_wh = 0.0
    total_switch_wh = 0.0

    t = 0
    while True:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward
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

        records.append(
            StepRecord(
                t=int(info["timestep"]),
                action=int(action),
                selected_upf=str(info["selected_upf"]),
                actual_load_gbps=float(info["actual_load_gbps"]),
                predicted_load_gbps=float(info["predicted_load_gbps"]),
                power_watts=float(info["power_watts"]),
                sec_w_per_mbps=float(info.get("sec_w_per_mbps", 0.0)),
                delay_us=float(info["delay_us"]),
                predicted_loss=float(info["predicted_loss"]),
                q_score=float(info.get("q_score", info.get("performance", 1.0))),
                qos_penalty=float(info["qos_penalty"]),
                switching_energy_wh=float(info["switching_energy_wh"]),
                switch_penalty=float(info.get("switch_penalty", 0.0)),
                cooldown_penalty=float(info.get("cooldown_penalty", 0.0)),
                steps_since_switch=int(info.get("steps_since_switch", 0)),
                is_safe=bool(info["is_safe"]),
                energy_term=float(info.get("energy_term", 0.0)),
                reward=float(reward),
                cumulative_reward=float(cum_reward),
            )
        )
        t += 1
        if terminated or truncated:
            break
        if max_steps is not None and t >= max_steps:
            break

    n_steps = max(1, len(records))
    summary = RolloutSummary(
        policy=policy_id,
        label=POLICY_LABELS[policy_id],
        cluster_idx=cluster_idx,
        horizon_idx=horizon_idx,
        steps=len(records),
        total_reward=cum_reward,
        mean_reward=cum_reward / n_steps,
        total_energy_wh=total_energy_wh,
        total_switch_wh=total_switch_wh,
        unsafe_rate=n_unsafe / n_steps,
        dpdk_rate=n_dpdk / n_steps,
        usr_rate=n_usr / n_steps,
        n_switches=n_switches,
    )
    return RolloutResponse(summary=summary, steps=records)
