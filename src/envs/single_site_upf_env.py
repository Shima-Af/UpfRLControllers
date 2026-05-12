"""Single-site Gymnasium environment around the UPF digital twin.

One episode = one cluster's full time series at a fixed forecast horizon.
At each step the agent picks a UPF type (0=DPDK, 1=USR); the digital twin
evaluates power / QoS for the realized load and applies switching-cost
accounting via ``DigitalTwin.compute_step``.

No training code lives here — this is just the environment for Phase 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.utils.config import load_yaml, project_root


_ACTION_TO_UPF: dict[int, str] = {0: "DPDK", 1: "USR"}
_UPF_TO_ACTION: dict[str, int] = {"DPDK": 0, "USR": 1}


class SingleSiteUPFEnv(gym.Env):
    """Discrete-action env wrapping one cluster of the UPF digital twin.

    Observation (Box, shape (6,), float32):
        [actual_load_gbps, predicted_load_gbps, prev_action,
         prev_power_watts, prev_safety_flag, timestep_progress]

    Action (Discrete(2)):
        0 -> DPDK, 1 -> USR

    Reward:
        -(energy_weight * energy_wh
          + qos_weight * unsafe_penalty
          + switching_weight * switching_energy_wh)
        where unsafe_penalty = 0 if composite.is_safe else 1.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cluster_idx: int = 0,
        horizon_idx: int = 0,
        scenario_cfg: dict | None = None,
        paths_cfg: dict | None = None,
        project_root_dir: str | Path | None = None,
    ) -> None:
        super().__init__()

        self.cluster_idx = int(cluster_idx)
        self.horizon_idx = int(horizon_idx)

        self._root = Path(project_root_dir) if project_root_dir else project_root()
        self.scenario_cfg = scenario_cfg if scenario_cfg is not None else load_yaml(
            "configs/scenario_rl.yaml"
        )
        self.paths_cfg = paths_cfg if paths_cfg is not None else load_yaml(
            "configs/digital_twin_paths.yaml"
        )

        # --- Traffic artifacts (predictions / targets) ---
        tf = self.paths_cfg.get("traffic_forecaster", {})
        pred_rel = tf.get("predictions", "data/external/traffic_forecaster/predictions_test.npy")
        targ_rel = tf.get("targets", "data/external/traffic_forecaster/targets_test.npy")
        pred_path = self._root / pred_rel
        targ_path = self._root / targ_rel
        if not pred_path.exists():
            raise FileNotFoundError(f"Traffic predictions not found: {pred_path}")
        if not targ_path.exists():
            raise FileNotFoundError(f"Traffic targets not found: {targ_path}")

        predictions_norm = np.load(pred_path)
        targets_norm = np.load(targ_path)
        if predictions_norm.shape != targets_norm.shape:
            raise ValueError(
                f"predictions shape {predictions_norm.shape} != "
                f"targets shape {targets_norm.shape}"
            )
        if predictions_norm.ndim != 3:
            raise ValueError(
                f"Expected 3-D arrays (N, H, K), got {predictions_norm.shape}"
            )

        N, H, K = predictions_norm.shape
        if not (0 <= self.cluster_idx < K):
            raise ValueError(f"cluster_idx {self.cluster_idx} out of range [0, {K})")
        if not (0 <= self.horizon_idx < H):
            raise ValueError(f"horizon_idx {self.horizon_idx} out of range [0, {H})")

        # Denormalize to Gbps. Prefer alpha from paths_cfg (set during artifact
        # publishing); fall back to scenario_rl.yaml placeholder.
        alpha = (
            tf.get("alpha")
            or self.scenario_cfg.get("traffic", {}).get("alpha")
            or 1.0
        )
        self._alpha = float(alpha)
        self._actual_gbps = (
            targets_norm[:, self.horizon_idx, self.cluster_idx] * self._alpha
        ).astype(np.float64)
        self._predicted_gbps = (
            predictions_norm[:, self.horizon_idx, self.cluster_idx] * self._alpha
        ).astype(np.float64)
        self._N = int(N)

        # --- Digital twin (imported lazily so the module imports even if the
        # dependency is missing on a fresh checkout) ---
        from upf_digital_twin import DigitalTwin  # noqa: WPS433 — lazy on purpose

        self._twin = DigitalTwin(
            scenario_cfg=self.scenario_cfg,
            paths_cfg=self.paths_cfg,
            project_root=self._root,
        )
        self._step_h = float(self._twin.step_h)

        # --- Reward weights ---
        rw = self.scenario_cfg.get("reward_weights", {})
        self._w_energy = float(rw.get("energy_weight", 1.0))
        self._w_qos = float(rw.get("qos_weight", 1.0))
        self._w_switch = float(rw.get("switching_weight", 0.1))

        # --- Initial action ---
        ia = self.scenario_cfg.get("initial_action", "DPDK")
        if not isinstance(ia, str) or ia not in ("DPDK", "USR"):
            ia = "DPDK"
        self._initial_action = ia

        # --- Spaces ---
        self.action_space = spaces.Discrete(2)
        # Generous finite bounds; load and power are non-negative in practice.
        high = np.array([1e3, 1e3, 1.0, 1e3, 1.0, 1.0], dtype=np.float32)
        low = np.zeros(6, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # --- Episode state ---
        self._current_action: str = self._initial_action
        self._t: int = 0
        self._prev_power: float = 0.0
        self._prev_safety: float = 1.0

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
        self._current_action = self._initial_action
        self._t = 0
        self._prev_power = 0.0
        self._prev_safety = 1.0
        info = {
            "cluster_idx": self.cluster_idx,
            "horizon_idx": self.horizon_idx,
            "episode_length": self._N,
            "initial_action": self._initial_action,
        }
        return self._obs(), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._t >= self._N:
            raise RuntimeError("step() called after episode ended; call reset() first.")

        requested = _ACTION_TO_UPF[int(action)]
        load = float(self._actual_gbps[self._t])
        predicted = float(self._predicted_gbps[self._t])

        old_r = self._twin.evaluate_action(self._current_action, load)
        new_r = (
            self._twin.evaluate_action(requested, load)
            if requested != self._current_action
            else old_r
        )
        composite, sw_energy, new_current, pending = self._twin.compute_step(
            self._current_action, requested, old_r, new_r,
        )
        realised = self._current_action if pending else requested
        if not pending:
            self._current_action = new_current

        energy_wh = float(composite.power_watts) * self._step_h
        unsafe_penalty = 0.0 if bool(composite.is_safe) else 1.0
        reward = -(
            self._w_energy * energy_wh
            + self._w_qos * unsafe_penalty
            + self._w_switch * float(sw_energy)
        )

        info: dict[str, Any] = {
            "selected_upf": realised,
            "requested_upf": requested,
            "actual_load_gbps": load,
            "predicted_load_gbps": predicted,
            "power_watts": float(composite.power_watts),
            "switching_energy_wh": float(sw_energy),
            "is_safe": bool(composite.is_safe),
            "timestep": int(self._t),
            "cluster_idx": int(self.cluster_idx),
        }

        # Bookkeeping for the *next* observation.
        self._prev_power = float(composite.power_watts)
        self._prev_safety = 1.0 if bool(composite.is_safe) else 0.0
        self._t += 1

        terminated = False
        truncated = self._t >= self._N
        obs = self._terminal_obs() if truncated else self._obs()

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        actual = float(self._actual_gbps[self._t])
        predicted = float(self._predicted_gbps[self._t])
        prev_action_flag = float(_UPF_TO_ACTION[self._current_action])
        progress = float(self._t) / max(1, self._N - 1)
        return np.array(
            [
                actual,
                predicted,
                prev_action_flag,
                self._prev_power,
                self._prev_safety,
                progress,
            ],
            dtype=np.float32,
        )

    def _terminal_obs(self) -> np.ndarray:
        prev_action_flag = float(_UPF_TO_ACTION[self._current_action])
        return np.array(
            [0.0, 0.0, prev_action_flag, self._prev_power, self._prev_safety, 1.0],
            dtype=np.float32,
        )
