"""Multi-cluster policy registry + rollout helper.

Mirror of ``policies.py`` for the Phase-7 multi-cluster view of the
dashboard. Wraps:

  - MAPPO from ``src/trainers/mappo.py`` (best-of-val checkpoint).
  - The classical baselines ``always-DPDK``, ``threshold (derived)``,
    ``hysteresis (tuned)``, ``hysteresis (auto)`` — same operating
    points as the Phase 7 evaluator.

The MAPPO checkpoint is loaded lazily on first request and cached.
The classical baselines are stateless or per-request stateful (the
hysteresis instances reset themselves on rollout start).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from upf_digital_twin import DigitalTwin

from src.baselines.hysteresis import MultiAgentHysteresis
from src.baselines.threshold_derivation import (
    derive_thresholds,
    load_forecast_mae_gbps,
)
from src.envs.multi_agent_upf_env import MultiAgentUPFEnv
from src.trainers.mappo import Actor
from src.utils.config import load_yaml

from .schemas import (
    MultiClusterRolloutResponse,
    MultiClusterSummary,
    MultiPolicyId,
    MultiPolicyInfo,
    SplitId,
)

# ---------------------------------------------------------------------------
# Static metadata
# ---------------------------------------------------------------------------

MULTI_POLICY_LABELS: dict[MultiPolicyId, str] = {
    "mappo": "MAPPO (trained)",
    "all-dpdk": "Always DPDK",
    "threshold-derived": "Threshold (derived)",
    "hysteresis-tuned": "Hysteresis (band=20 Mbps)",
    "hysteresis-auto": "Hysteresis (auto band)",
}

MULTI_POLICY_DESCRIPTIONS: dict[MultiPolicyId, str] = {
    "mappo": (
        "Phase 6 MAPPO — centralised training, decentralised execution. "
        "Shared actor (one network, all agents), centralised critic on "
        "the joint state."
    ),
    "all-dpdk": "Every cluster on DPDK at every step. The safe energy ceiling.",
    "threshold-derived": (
        "Per-cluster stateless threshold at the twin-derived decision "
        "point (min(energy_breakeven, qos_limit) − safety margin)."
    ),
    "hysteresis-tuned": (
        "Per-cluster hysteresis with a manually-tuned narrow band "
        "(default 20 Mbps), cooldown=1. Best classical controller."
    ),
    "hysteresis-auto": (
        "Per-cluster hysteresis with the paper-faithful auto band "
        "(2 × forecast MAE). Tends to degenerate to always-DPDK when "
        "the forecaster is noisy."
    ),
}


def _latest_mappo_checkpoint(repo_root: Path) -> Path | None:
    exp = repo_root / "experiments"
    if not exp.is_dir():
        return None
    cands = sorted(
        exp.glob("mappo_seed*_*"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for d in cands:
        for name in ("mappo_best.pt", "mappo_final.pt"):
            cand = d / name
            if cand.exists():
                return cand
    return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class MultiPolicyRegistry:
    """Lazy-loading registry for the multi-cluster controllers.

    Heavy artefacts (PyTorch checkpoint, twin-derived threshold spec)
    are computed on first use and cached for the process lifetime.
    """

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self._mappo_path = _latest_mappo_checkpoint(repo_root)
        self._mappo_actor: Actor | None = None
        self._derived_thresholds: dict | None = None

    def list_policies(self) -> list[MultiPolicyInfo]:
        return [
            MultiPolicyInfo(
                id=pid,
                label=MULTI_POLICY_LABELS[pid],
                description=MULTI_POLICY_DESCRIPTIONS[pid],
                requires_model=(pid == "mappo"),
                model_loaded=(pid != "mappo") or (self._mappo_path is not None),
            )
            for pid in MULTI_POLICY_LABELS
        ]

    def _load_actor(self) -> Actor:
        if self._mappo_actor is not None:
            return self._mappo_actor
        if self._mappo_path is None:
            raise FileNotFoundError(
                "No MAPPO checkpoint found under experiments/. "
                "Train one first with scripts/train_mappo.py."
            )
        ckpt = torch.load(self._mappo_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        a = Actor(
            ckpt["obs_dim"], ckpt["n_actions"], cfg.get("actor_hidden", 64)
        )
        a.load_state_dict(ckpt["actor"])
        a.eval()
        self._mappo_actor = a
        return a

    def derived_thresholds(self) -> dict:
        """Compute (and cache) the twin-derived threshold operating points."""
        if self._derived_thresholds is not None:
            return self._derived_thresholds
        scenario = load_yaml("configs/scenario_rl.yaml")
        paths = load_yaml("configs/digital_twin_paths.yaml")
        twin = DigitalTwin(
            scenario_cfg=scenario, paths_cfg=paths, project_root=self.repo_root
        )
        tf = paths.get("traffic_forecaster", {})
        summary_path = self.repo_root / tf.get(
            "forecast_eval_summary",
            "data/external/traffic_forecaster/forecast_eval_summary.json",
        )
        alpha = float(
            tf.get("alpha")
            or scenario.get("traffic", {}).get("alpha")
            or 1.0
        )
        mae = load_forecast_mae_gbps(
            summary_path, alpha_gbps_per_norm=alpha, K=10
        )
        spec = derive_thresholds(
            twin, safety_margin_mbps=10.0, forecast_mae_gbps=mae
        )
        self._derived_thresholds = {
            "decision_gbps": spec.decision_gbps,
            "t_up_gbps_auto": spec.t_up_gbps,
            "t_down_gbps_auto": spec.t_down_gbps,
            "hysteresis_band_gbps_auto": spec.hysteresis_band_gbps,
            "forecast_mae_gbps": spec.forecast_mae_gbps,
            "energy_breakeven_gbps": spec.energy_breakeven_gbps,
            "qos_limit_gbps": spec.qos_limit_gbps,
            "derived_from": spec.derived_from,
        }
        return self._derived_thresholds

    def get_policy(
        self,
        policy_id: MultiPolicyId,
        *,
        agents: list[str],
        seed: int,
        threshold_gbps: float,
        hysteresis_band_mbps: float,
        hysteresis_cooldown_steps: int,
    ) -> Callable[[dict[str, np.ndarray]], dict[str, int]]:
        if policy_id == "mappo":
            actor = self._load_actor()

            @torch.no_grad()
            def _mappo(obs_dict):
                obs = np.stack(
                    [obs_dict[a] for a in agents]
                ).astype(np.float32)
                action, _ = actor.act(torch.from_numpy(obs), deterministic=True)
                return {a: int(action[k].item()) for k, a in enumerate(agents)}
            return _mappo

        if policy_id == "all-dpdk":
            def _dpdk(obs_dict):
                return {a: 0 for a in obs_dict.keys()}
            return _dpdk

        if policy_id == "threshold-derived":
            decision = threshold_gbps  # already in Gbps
            def _threshold(obs_dict):
                out = {}
                for agent, obs in obs_dict.items():
                    forecast = float(obs[9])
                    out[agent] = 1 if forecast < decision else 0
                return out
            return _threshold

        if policy_id == "hysteresis-tuned":
            t_up = threshold_gbps
            t_down = max(0.0, threshold_gbps - hysteresis_band_mbps / 1000.0)
            return MultiAgentHysteresis(
                agents=agents,
                t_up_gbps=t_up, t_down_gbps=t_down,
                cooldown_steps=hysteresis_cooldown_steps,
            )

        if policy_id == "hysteresis-auto":
            d = self.derived_thresholds()
            return MultiAgentHysteresis(
                agents=agents,
                t_up_gbps=d["t_up_gbps_auto"],
                t_down_gbps=d["t_down_gbps_auto"],
                cooldown_steps=hysteresis_cooldown_steps,
            )

        raise ValueError(f"Unknown multi-policy id: {policy_id}")


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def run_multi_rollout(
    registry: MultiPolicyRegistry,
    *,
    policy_id: MultiPolicyId,
    horizon_idx: int,
    split: SplitId,
    seed: int,
    max_steps: int | None,
    threshold_gbps: float,
    hysteresis_band_mbps: float,
    hysteresis_cooldown_steps: int,
) -> MultiClusterRolloutResponse:
    """Run one full episode of MultiAgentUPFEnv and serialise the response."""
    env = MultiAgentUPFEnv(horizon_idx=horizon_idx, split=split)
    obs, _info = env.reset(seed=seed)
    agents = list(env.possible_agents)
    K = env.K
    step_h = env.step_h

    policy = registry.get_policy(
        policy_id,
        agents=agents,
        seed=seed,
        threshold_gbps=threshold_gbps,
        hysteresis_band_mbps=hysteresis_band_mbps,
        hysteresis_cooldown_steps=hysteresis_cooldown_steps,
    )
    if hasattr(policy, "reset"):
        policy.reset()

    actions: list[list[int]] = [[] for _ in range(K)]
    per_c_reward: list[list[float]] = [[] for _ in range(K)]
    per_c_load: list[list[float]] = [[] for _ in range(K)]
    per_c_total_r = np.zeros(K, dtype=np.float64)
    per_c_energy = np.zeros(K, dtype=np.float64)
    per_c_unsafe = np.zeros(K, dtype=np.int64)
    per_c_dpdk = np.zeros(K, dtype=np.int64)
    per_c_usr = np.zeros(K, dtype=np.int64)
    per_c_switches = np.zeros(K, dtype=np.int64)
    last_act: list[int | None] = [None] * K
    cumulative: list[float] = []
    cum_r = 0.0
    t = 0

    while env.agents:
        act = policy(obs)
        obs, r, _term, _trunc, info = env.step(act)
        for k, a in enumerate(agents):
            ai = int(act[a])
            actions[k].append(ai)
            per_c_reward[k].append(float(r[a]))
            per_c_total_r[k] += r[a]
            ck = info[a]
            per_c_load[k].append(float(ck["actual_load_gbps"]))
            per_c_energy[k] += float(ck["power_watts"]) * step_h
            if not ck["is_safe"]:
                per_c_unsafe[k] += 1
            if ck["selected_upf"] == "DPDK":
                per_c_dpdk[k] += 1
            else:
                per_c_usr[k] += 1
            if last_act[k] is not None and last_act[k] != ai:
                per_c_switches[k] += 1
            last_act[k] = ai
        cum_r += float(sum(r.values()))
        cumulative.append(cum_r)
        t += 1
        if max_steps is not None and t >= max_steps:
            break

    n = max(1, t)
    summary = MultiClusterSummary(
        policy=policy_id,
        label=MULTI_POLICY_LABELS[policy_id],
        horizon_idx=horizon_idx,
        split=split,
        K=K,
        steps=t,
        total_reward_unweighted=float(per_c_total_r.sum()),
        total_energy_wh=float(per_c_energy.sum()),
        agg_unsafe_rate=float(per_c_unsafe.sum() / (K * n)),
        agg_usr_rate=float(per_c_usr.sum() / (K * n)),
        agg_n_switches=int(per_c_switches.sum()),
        per_cluster_total_reward=per_c_total_r.tolist(),
        per_cluster_energy_wh=per_c_energy.tolist(),
        per_cluster_unsafe_rate=(per_c_unsafe / n).tolist(),
        per_cluster_usr_rate=(per_c_usr / n).tolist(),
        per_cluster_n_switches=per_c_switches.tolist(),
    )
    return MultiClusterRolloutResponse(
        summary=summary,
        actions=actions,
        per_cluster_reward=per_c_reward,
        per_cluster_load=per_c_load,
        cumulative_reward=cumulative,
    )
