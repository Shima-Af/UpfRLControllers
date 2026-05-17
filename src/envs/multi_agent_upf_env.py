"""Phase 5 — PettingZoo Parallel multi-agent UPF environment.

Each of the K=10 clusters becomes one agent (``"cluster_0"`` ..
``"cluster_9"``). All agents act simultaneously every step (Parallel
API). Per-cluster reward, cooldown, switching cost, and observation
are delegated to the underlying ``SingleSiteUPFEnv`` instances — same
pattern as the Phase-3 ``MultiSiteUPFEnv`` wrapper, just exposed via
the multi-agent dict-based API instead of a flat joint Box/MultiDiscrete.

This sets up the API surface for Phase-6 MAPPO (centralised training,
decentralised execution): the trainer can collect per-agent
trajectories from this env, while a centralised critic still has
access to the joint state via ``state()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from src.envs.single_site_upf_env import SingleSiteUPFEnv
from src.utils.config import load_yaml, project_root

_EPS_LOAD = 1e-6


class MultiAgentUPFEnv(ParallelEnv):
    """K-agent PettingZoo Parallel env wrapping K SingleSite envs.

    API contract (PettingZoo Parallel, v1.26+):
        agents: list of agent IDs alive this step
        possible_agents: full roster (immutable)
        observation_space(agent) / action_space(agent): per-agent
        reset(seed=, options=) -> (obs_dict, info_dict)
        step(actions_dict) -> (obs_dict, rewards_dict, terminations_dict,
                               truncations_dict, infos_dict)
        state() -> global state for the centralised critic

    Per-agent reward is the underlying ``SingleSiteUPFEnv`` reward.
    The wrapper does not apply the load-weighting that ``MultiSiteUPFEnv``
    used — the multi-agent setup expects each agent to optimise its
    own reward; the centralised critic in MAPPO learns the value of
    the joint state.
    """

    metadata = {"name": "upf_multi_agent_v1"}

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

        if cluster_indices is None:
            tf = self.paths_cfg.get("traffic_forecaster", {})
            targ_rel = tf.get(
                f"targets_{split}",
                f"data/external/traffic_forecaster/targets_{split}.npy",
            )
            targets = np.load(self._root / targ_rel)
            cluster_indices = list(range(int(targets.shape[2])))

        self.cluster_indices = list(cluster_indices)
        self.K = len(self.cluster_indices)
        self.horizon_idx = int(horizon_idx)
        self.split = split

        # One sub-env per cluster — they own all per-cluster math.
        self._envs: dict[str, SingleSiteUPFEnv] = {
            self._agent_id(k): SingleSiteUPFEnv(
                cluster_idx=k,
                horizon_idx=self.horizon_idx,
                scenario_cfg=self.scenario_cfg,
                paths_cfg=self.paths_cfg,
                project_root_dir=self._root,
                split=self.split,
            )
            for k in self.cluster_indices
        }

        self.possible_agents: list[str] = list(self._envs.keys())
        self.agents: list[str] = list(self.possible_agents)

        # All sub-envs share episode length and obs dim.
        first = next(iter(self._envs.values()))
        self._N = int(first._N)  # noqa: SLF001
        self._obs_dim_per = int(first.observation_space.shape[0])

        for e in self._envs.values():
            if int(e._N) != self._N:  # noqa: SLF001
                raise ValueError("Sub-env episode lengths differ.")
            if int(e.observation_space.shape[0]) != self._obs_dim_per:
                raise ValueError("Sub-env obs dims differ.")

        self._t = 0

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Space:
        return self._envs[agent].observation_space

    def action_space(self, agent: str) -> spaces.Space:
        return self._envs[agent].action_space

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        obs: dict[str, np.ndarray] = {}
        infos: dict[str, dict] = {}
        for k_idx, agent in enumerate(self.possible_agents):
            sub_seed = None if seed is None else int(seed) + k_idx
            o, i = self._envs[agent].reset(seed=sub_seed)
            obs[agent] = o
            infos[agent] = i
        self.agents = list(self.possible_agents)
        self._t = 0
        return obs, infos

    def step(
        self, actions: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        if set(actions.keys()) != set(self.agents):
            missing = set(self.agents) - set(actions.keys())
            extra = set(actions.keys()) - set(self.agents)
            raise ValueError(
                f"actions dict mismatch: missing={missing} extra={extra}"
            )

        obs: dict[str, np.ndarray] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict] = {}

        any_truncated = False
        for agent in self.agents:
            o, r, term, trunc, info = self._envs[agent].step(int(actions[agent]))
            obs[agent] = o
            rewards[agent] = float(r)
            terminations[agent] = bool(term)
            truncations[agent] = bool(trunc)
            infos[agent] = info
            any_truncated = any_truncated or trunc

        self._t += 1
        if any_truncated:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Helpers used by MAPPO (centralised critic global-state access)
    # ------------------------------------------------------------------

    def state(self) -> np.ndarray:
        """Concatenated per-agent observation — the global state.

        Used by the centralised critic in CTDE training.
        """
        if not self.agents:
            # End-of-episode terminal state — return zeros to keep the
            # critic forward pass well-defined.
            return np.zeros(self.K * self._obs_dim_per, dtype=np.float32)
        chunks = [self._envs[a]._obs() for a in self.possible_agents]  # noqa: SLF001
        return np.concatenate(chunks).astype(np.float32)

    def state_space(self) -> spaces.Space:
        low = np.concatenate(
            [self._envs[a].observation_space.low for a in self.possible_agents]
        )
        high = np.concatenate(
            [self._envs[a].observation_space.high for a in self.possible_agents]
        )
        return spaces.Box(low=low.astype(np.float32),
                          high=high.astype(np.float32),
                          dtype=np.float32)

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass

    @property
    def step_h(self) -> float:
        return float(next(iter(self._envs.values()))._step_h)  # noqa: SLF001

    @property
    def tau(self) -> float:
        return float(next(iter(self._envs.values()))._tau)  # noqa: SLF001

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _agent_id(cluster_idx: int) -> str:
        return f"cluster_{cluster_idx}"
