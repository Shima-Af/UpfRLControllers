"""Multi-site Gymnasium environment — K UPF clusters in parallel.

Phase 3 setup (paper-aligned, centralised single-agent PPO):

  Observation: concatenation of the K per-cluster observations produced
               by ``SingleSiteUPFEnv``. Shape = (K * obs_dim_per_cluster,).
  Action:      ``MultiDiscrete([2] * K)`` — one binary decision per
               cluster (0 = DPDK, 1 = USR). Joint factored policy.
  Reward:      load-weighted sum of per-cluster rewards,
               minus an optional fleet-power-pool soft penalty
               (see ``configs/scenario_rl.yaml`` -> ``pool``):
                   P_total(t) = sum_k power_watts_steady_k(t)
                   overrun(t) = max(0, P_total(t) - power_cap_w)
                   L_pool(t)  = lambda_pool * overrun(t)
                   reward(t)  = sum_k w_k(t) * r_k(t) - L_pool(t)
               When ``power_cap_w`` is null the pool is disabled and
               behaviour matches prior runs exactly.

Per-cluster bookkeeping (cooldown, switching costs, history window,
prev_Q, prev_SEC) is delegated entirely to the underlying
``SingleSiteUPFEnv`` instances — the multi-site wrapper only handles
observation concatenation, the joint action, reward aggregation, and
the shared power-pool coupling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.single_site_upf_env import SingleSiteUPFEnv
from src.utils.config import load_yaml, project_root

_EPS_LOAD = 1e-6  # Gbps floor for the load-share denominator.


class MultiSiteUPFEnv(gym.Env):
    """K-cluster joint controller over independent SingleSite envs.

    The number of clusters ``K`` is inferred from the forecaster's
    target tensor shape (e.g. K=10 for the current Netflix run).
    Pass ``cluster_indices`` to control which subset participates
    (useful for smaller debug runs).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        cluster_indices: list[int] | None = None,
        horizon_idx: int = 0,
        scenario_cfg: dict | None = None,
        paths_cfg: dict | None = None,
        project_root_dir: str | Path | None = None,
        split: str = "test",
    ) -> None:
        super().__init__()

        self._root = Path(project_root_dir) if project_root_dir else project_root()
        self.scenario_cfg = scenario_cfg if scenario_cfg is not None else load_yaml(
            "configs/scenario_rl.yaml"
        )
        self.paths_cfg = paths_cfg if paths_cfg is not None else load_yaml(
            "configs/digital_twin_paths.yaml"
        )

        # Infer K from the target tensor shape if not given.
        if cluster_indices is None:
            tf = self.paths_cfg.get("traffic_forecaster", {})
            targ_rel = tf.get(
                f"targets_{split}",
                f"data/external/traffic_forecaster/targets_{split}.npy",
            )
            targets = np.load(self._root / targ_rel)
            K = int(targets.shape[2])
            cluster_indices = list(range(K))
        self.cluster_indices = list(cluster_indices)
        self.K = len(self.cluster_indices)
        self.horizon_idx = int(horizon_idx)
        self.split = split

        # Construct one SingleSite env per cluster — they own all the
        # per-cluster reward / cooldown / observation logic.
        self._envs: list[SingleSiteUPFEnv] = [
            SingleSiteUPFEnv(
                cluster_idx=k,
                horizon_idx=self.horizon_idx,
                scenario_cfg=self.scenario_cfg,
                paths_cfg=self.paths_cfg,
                project_root_dir=self._root,
                split=self.split,
            )
            for k in self.cluster_indices
        ]
        # All sub-envs share the same episode length and obs dim.
        self._N = self._envs[0]._N  # noqa: SLF001
        obs_dim_per = int(self._envs[0].observation_space.shape[0])
        self._obs_dim_per = obs_dim_per

        # Sanity: all sub-envs must agree on length and obs dim.
        for e in self._envs[1:]:
            if e._N != self._N:  # noqa: SLF001
                raise ValueError(
                    "Sub-env episode lengths differ — split/horizon mismatch."
                )
            if int(e.observation_space.shape[0]) != obs_dim_per:
                raise ValueError("Sub-env observation dims differ.")

        # --- Spaces ---
        self.action_space = spaces.MultiDiscrete([2] * self.K)
        # Stack per-cluster bounds.
        low = np.concatenate([e.observation_space.low for e in self._envs])
        high = np.concatenate([e.observation_space.high for e in self._envs])
        self.observation_space = spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32,
        )

        # Shared fleet-power pool (instant soft penalty). Disabled when null.
        pool_cfg = self.scenario_cfg.get("pool", {})
        cap = pool_cfg.get("power_cap_w", None)
        self._pool_cap_w: float | None = float(cap) if cap is not None else None
        self._lambda_pool = float(pool_cfg.get("lambda_pool", 0.0))

        # Integrated energy budget (telecom-realistic kWh-style). Disabled
        # when null. Tracks cumulative steady-state energy per episode.
        budget_cfg = self.scenario_cfg.get("budget", {})
        b = budget_cfg.get("budget_wh", None)
        self._budget_wh: float | None = float(b) if b is not None else None
        self._lambda_budget = float(budget_cfg.get("lambda_budget", 0.0))
        self._cumulative_energy_wh: float = 0.0

        # Episode state — tracked by sub-envs; we only keep t for the
        # truncation flag.
        self._t = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        obs_chunks: list[np.ndarray] = []
        for k_idx, env in enumerate(self._envs):
            sub_seed = None if seed is None else int(seed) + k_idx
            obs, _info = env.reset(seed=sub_seed)
            obs_chunks.append(obs)
        self._t = 0
        self._cumulative_energy_wh = 0.0
        info = {
            "cluster_indices": list(self.cluster_indices),
            "K": self.K,
            "episode_length": self._N,
            "split": self.split,
            "obs_dim_per_cluster": self._obs_dim_per,
        }
        return np.concatenate(obs_chunks).astype(np.float32), info

    def step(
        self, action: np.ndarray | list[int]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if action.shape[0] != self.K:
            raise ValueError(
                f"action must have length K={self.K}, got {action.shape[0]}"
            )

        obs_chunks: list[np.ndarray] = []
        per_cluster_rewards = np.zeros(self.K, dtype=np.float64)
        per_cluster_loads = np.zeros(self.K, dtype=np.float64)
        per_cluster_info: list[dict[str, Any]] = []
        terminated_any = False
        truncated_any = False

        for k_idx, env in enumerate(self._envs):
            obs_k, r_k, term_k, trunc_k, info_k = env.step(int(action[k_idx]))
            obs_chunks.append(obs_k)
            per_cluster_rewards[k_idx] = r_k
            per_cluster_loads[k_idx] = float(info_k["actual_load_gbps"])
            per_cluster_info.append(info_k)
            terminated_any = terminated_any or term_k
            truncated_any = truncated_any or trunc_k

        # Per-step load-weighted sum reward.
        total_load = per_cluster_loads.sum()
        if total_load < _EPS_LOAD:
            # All-idle step — fall back to a uniform mean so the gradient
            # doesn't vanish entirely. Rare in practice.
            weights = np.full(self.K, 1.0 / self.K, dtype=np.float64)
        else:
            weights = per_cluster_loads / total_load
        weighted_reward = float(np.dot(weights, per_cluster_rewards))

        # Shared fleet-power pool — soft penalty on steady-state overrun.
        p_total_w = float(
            sum(ic["power_watts_steady"] for ic in per_cluster_info)
        )
        pool_overrun_w = 0.0
        pool_penalty = 0.0
        if self._pool_cap_w is not None:
            pool_overrun_w = max(0.0, p_total_w - self._pool_cap_w)
            pool_penalty = self._lambda_pool * pool_overrun_w
            weighted_reward -= pool_penalty

        # Integrated energy budget — penalty grows with running excess.
        step_energy_wh = p_total_w * self.step_h
        self._cumulative_energy_wh += step_energy_wh
        budget_excess_wh = 0.0
        budget_penalty = 0.0
        if self._budget_wh is not None:
            target_energy_wh = self._budget_wh * (self._t + 1) / max(1, self._N)
            budget_excess_wh = max(0.0, self._cumulative_energy_wh - target_energy_wh)
            budget_penalty = self._lambda_budget * budget_excess_wh
            weighted_reward -= budget_penalty

        self._t += 1
        info: dict[str, Any] = {
            "timestep": int(self._t),
            "per_cluster_reward": per_cluster_rewards.tolist(),
            "per_cluster_load_gbps": per_cluster_loads.tolist(),
            "per_cluster_weight": weights.tolist(),
            "per_cluster": per_cluster_info,
            "total_load_gbps": float(total_load),
            "sum_reward_unweighted": float(per_cluster_rewards.sum()),
            "pool_power_w": p_total_w,
            "pool_cap_w": self._pool_cap_w,
            "pool_overrun_w": pool_overrun_w,
            "pool_penalty": pool_penalty,
            "step_energy_wh": step_energy_wh,
            "cumulative_energy_wh": self._cumulative_energy_wh,
            "budget_wh": self._budget_wh,
            "budget_excess_wh": budget_excess_wh,
            "budget_penalty": budget_penalty,
        }
        obs = np.concatenate(obs_chunks).astype(np.float32)
        return obs, weighted_reward, terminated_any, truncated_any, info

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def step_h(self) -> float:
        """Step duration in hours (shared across sub-envs)."""
        return float(self._envs[0]._step_h)  # noqa: SLF001

    @property
    def tau(self) -> float:
        return float(self._envs[0]._tau)  # noqa: SLF001
