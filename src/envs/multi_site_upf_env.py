"""Multi-site Gymnasium environment — K independent UPF clusters in parallel.

Phase 3 setup (paper-aligned, centralised single-agent PPO):

  Observation: concatenation of the K per-cluster observations produced
               by ``SingleSiteUPFEnv``. Shape = (K * obs_dim_per_cluster,).
  Action:      ``MultiDiscrete([2] * K)`` — one binary decision per
               cluster (0 = DPDK, 1 = USR). Joint factored policy.
  Reward:      load-weighted sum of per-cluster rewards.
               At each step ``t``, weights are
                   w_k(t) = load_k(t) / max(sum_j load_j(t), eps)
               so an idle cluster contributes ~0 to the gradient and a
               peaking cluster dominates. Matches the paper's per-step
               SEC framing.

Per-cluster bookkeeping (cooldown, switching costs, history window,
prev_Q, prev_SEC) is delegated entirely to the underlying
``SingleSiteUPFEnv`` instances — the multi-site wrapper only handles
observation concatenation, the joint action, and reward aggregation.
This keeps the per-cluster reward math defined in exactly one place
(``SingleSiteUPFEnv.step``) and avoids drift between Phase 2 and
Phase 3.
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

        self._t += 1
        info: dict[str, Any] = {
            "timestep": int(self._t),
            "per_cluster_reward": per_cluster_rewards.tolist(),
            "per_cluster_load_gbps": per_cluster_loads.tolist(),
            "per_cluster_weight": weights.tolist(),
            "per_cluster": per_cluster_info,
            "total_load_gbps": float(total_load),
            "sum_reward_unweighted": float(per_cluster_rewards.sum()),
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
