"""Single-site Gymnasium environment around the UPF digital twin.

Reward and observation are aligned with COMCOM-S-26-00430 (Section 4.5
and 4.3), adapted to a binary action space (DPDK vs USR; horizontal
scaling is a future extension) and to a soft cooldown penalty (the
paper uses a hard mask).

One episode = one cluster's full time series at a fixed forecast
horizon. At each step the agent picks a UPF type (0=DPDK, 1=USR); the
digital twin evaluates power and QoS for the realised load and applies
switching-cost accounting via ``DigitalTwin.compute_step``.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.utils.config import load_yaml, project_root

_ACTION_TO_UPF: dict[int, str] = {0: "DPDK", 1: "USR"}
_UPF_TO_ACTION: dict[str, int] = {"DPDK": 0, "USR": 1}
_EPS = 1e-3  # numerical floor for load when computing SEC (Mbps)


class SingleSiteUPFEnv(gym.Env):
    """Discrete-action env wrapping one cluster of the UPF digital twin.

    Action (Discrete(2)):
        0 -> DPDK, 1 -> USR

    Observation (Box, float32). Shape is (history_window + 6,):
        index 0                              : current actual load (Gbps)
        indices 1 .. history_window          : past actual loads, oldest first
        index history_window+1               : 1-step forecast (Gbps)
        index history_window+2               : prev realised action (0=DPDK, 1=USR)
        index history_window+3               : previous Q score in [0, 1]
        index history_window+4               : previous SEC (W/Mbps)
        index history_window+5               : cooldown progress in [0, 1]

    Reward (paper-aligned, Section 4.5):
        SEC_t  = P_t / max(load_mbps, eps)
        delay_excess = max(0, delay_us - delay_budget_us)
        loss_excess  = max(0, predicted_loss - max_loss_pkts_per_interval)
        delay_score  = max(0, 1 - delay_excess / delay_budget_us)
        loss_score   = max(0, 1 - loss_excess  / max_loss_pkts_per_interval)
        Q_t          = min(delay_score, loss_score)
        L_QoS        = lambda_qos * max(0, tau - Q_t)
        L_SW         = c_dpdk if realised switched->DPDK else
                       c_usr  if realised switched->USR else 0
        L_CD         = soft cooldown surcharge (our extension)
        reward       = -(alpha * SEC_t + L_QoS + L_SW + L_CD)
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
        pred_rel = tf.get(
            "predictions", "data/external/traffic_forecaster/predictions_test.npy"
        )
        targ_rel = tf.get(
            "targets", "data/external/traffic_forecaster/targets_test.npy"
        )
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

        alpha_load = (
            tf.get("alpha")
            or self.scenario_cfg.get("traffic", {}).get("alpha")
            or 1.0
        )
        self._load_alpha = float(alpha_load)
        self._actual_gbps = (
            targets_norm[:, self.horizon_idx, self.cluster_idx] * self._load_alpha
        ).astype(np.float64)
        self._predicted_gbps = (
            predictions_norm[:, self.horizon_idx, self.cluster_idx] * self._load_alpha
        ).astype(np.float64)
        self._N = int(N)

        # --- Digital twin + lazy precompute of per-step surrogate outputs ---
        from upf_digital_twin import DigitalTwin  # noqa: WPS433
        from upf_digital_twin.twin.upf_profile import UPFResult  # noqa: WPS433

        self._UPFResult = UPFResult
        self._twin = DigitalTwin(
            scenario_cfg=self.scenario_cfg,
            paths_cfg=self.paths_cfg,
            project_root=self._root,
        )
        self._step_h = float(self._twin.step_h)
        self._dpdk_batch = self._twin.evaluate_batch("DPDK", self._actual_gbps)
        self._usr_batch = self._twin.evaluate_batch("USR", self._actual_gbps)

        # --- Reward weights ---
        rw = self.scenario_cfg.get("reward_weights", {})
        self._alpha = float(rw.get("alpha", 100.0))
        self._lambda_qos = float(rw.get("lambda_qos", 30.0))
        self._tau = float(rw.get("tau", 0.90))
        self._c_dpdk = float(rw.get("c_dpdk", 0.03))
        self._c_usr = float(rw.get("c_usr", 0.012))
        self._cooldown_period = int(rw.get("cooldown_period", 0))
        self._cooldown_cost = float(rw.get("cooldown_cost", 0.0))
        if self._cooldown_period < 0:
            raise ValueError("cooldown_period must be >= 0")

        # QoS budgets used to normalise the per-step QoS score.
        qos_cfg = self.scenario_cfg.get("upf", {}).get("qos_budget", {})
        self._delay_budget_us = float(qos_cfg.get("delay_budget_us", 200.0))
        self._max_loss_pkts = float(qos_cfg.get("max_loss_pkts_per_interval", 5.0))
        if self._delay_budget_us <= 0 or self._max_loss_pkts <= 0:
            raise ValueError(
                "delay_budget_us and max_loss_pkts_per_interval must be > 0; got "
                f"{self._delay_budget_us=}, {self._max_loss_pkts=}"
            )

        # --- Observation parameters ---
        obs_cfg = self.scenario_cfg.get("observation", {})
        self._history_window = int(obs_cfg.get("history_window", 8))
        if self._history_window < 0:
            raise ValueError("history_window must be >= 0")

        # --- Initial action ---
        ia = self.scenario_cfg.get("initial_action", "DPDK")
        if not isinstance(ia, str) or ia not in ("DPDK", "USR"):
            ia = "DPDK"
        self._initial_action = ia

        # --- Spaces ---
        self.action_space = spaces.Discrete(2)
        obs_dim = self._history_window + 6
        # Generous bounds; SEC can spike at near-zero load but we clip in obs.
        high = np.array(
            [1e3] * (self._history_window + 1)  # current + history loads (Gbps)
            + [1e3]                              # forecast (Gbps)
            + [1.0]                              # prev action
            + [1.0]                              # prev Q
            + [1e3]                              # prev SEC (W/Mbps), generous
            + [1.0],                             # cooldown progress
            dtype=np.float32,
        )
        low = np.zeros(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # --- Episode state (set in reset) ---
        self._current_action: str = self._initial_action
        self._t: int = 0
        self._prev_q: float = 1.0
        self._prev_sec: float = 0.0
        self._steps_since_switch: int = max(1, self._cooldown_period)
        self._history: deque[float] = deque(
            [0.0] * self._history_window, maxlen=max(1, self._history_window)
        )

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
        self._prev_q = 1.0
        self._prev_sec = 0.0
        self._steps_since_switch = max(1, self._cooldown_period)
        if self._history_window > 0:
            self._history = deque(
                [0.0] * self._history_window, maxlen=self._history_window
            )
        else:
            self._history = deque(maxlen=1)
        info = {
            "cluster_idx": self.cluster_idx,
            "horizon_idx": self.horizon_idx,
            "episode_length": self._N,
            "initial_action": self._initial_action,
            "history_window": self._history_window,
        }
        return self._obs(), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._t >= self._N:
            raise RuntimeError("step() called after episode ended; call reset() first.")

        requested = _ACTION_TO_UPF[int(action)]
        load_gbps = float(self._actual_gbps[self._t])
        predicted = float(self._predicted_gbps[self._t])

        old_r = self._result_at(self._current_action, self._t)
        new_r = (
            self._result_at(requested, self._t)
            if requested != self._current_action
            else old_r
        )
        composite, sw_energy_wh, new_current, pending = self._twin.compute_step(
            self._current_action, requested, old_r, new_r,
        )
        realised = self._current_action if pending else requested
        prev_realised = self._current_action
        if not pending:
            self._current_action = new_current

        # --- Specific energy consumption (paper Eq. 5) ---
        load_mbps = max(load_gbps * 1000.0, _EPS)
        power_w = float(composite.power_watts)
        sec = power_w / load_mbps  # W/Mbps
        energy_term = self._alpha * sec

        # --- Continuous QoS score Q in [0, 1] ---
        delay_us = float(composite.delay_us)
        loss_pkts = float(composite.predicted_loss)
        delay_excess = max(0.0, delay_us - self._delay_budget_us)
        loss_excess = max(0.0, loss_pkts - self._max_loss_pkts)
        delay_score = max(0.0, 1.0 - delay_excess / self._delay_budget_us)
        loss_score = max(0.0, 1.0 - loss_excess / self._max_loss_pkts)
        q_score = min(delay_score, loss_score)
        qos_term = self._lambda_qos * max(0.0, self._tau - q_score)

        # --- Asymmetric switching cost (paper Eq. 7) ---
        type_changed = realised != prev_realised
        if type_changed:
            switch_term = self._c_dpdk if realised == "DPDK" else self._c_usr
        else:
            switch_term = 0.0

        # --- Soft cooldown (our extension) ---
        if self._cooldown_period > 0 and type_changed:
            steps_left = max(
                0, self._cooldown_period - self._steps_since_switch
            )
            cooldown_term = (
                self._cooldown_cost * steps_left / self._cooldown_period
            )
        else:
            cooldown_term = 0.0

        reward = -(energy_term + qos_term + switch_term + cooldown_term)

        info: dict[str, Any] = {
            "selected_upf": realised,
            "requested_upf": requested,
            "actual_load_gbps": load_gbps,
            "predicted_load_gbps": predicted,
            "power_watts": power_w,
            "sec_w_per_mbps": sec,
            "delay_us": delay_us,
            "predicted_loss": loss_pkts,
            "q_score": q_score,
            "qos_penalty": qos_term,
            "switching_energy_wh": float(sw_energy_wh),
            "switch_penalty": switch_term,
            "cooldown_penalty": cooldown_term,
            "steps_since_switch": int(self._steps_since_switch),
            "is_safe": bool(composite.is_safe),
            "energy_term": energy_term,
            "timestep": int(self._t),
            "cluster_idx": int(self.cluster_idx),
        }

        # --- Bookkeeping for next observation ---
        self._prev_q = q_score
        self._prev_sec = sec
        if type_changed:
            self._steps_since_switch = 0
        else:
            self._steps_since_switch += 1
        if self._history_window > 0:
            self._history.append(load_gbps)
        self._t += 1

        terminated = False
        truncated = self._t >= self._N
        obs = self._terminal_obs() if truncated else self._obs()
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _result_at(self, action: str, t: int) -> Any:
        b = self._dpdk_batch if action == "DPDK" else self._usr_batch
        return self._UPFResult(
            upf_type=action,
            load_gbps=float(self._actual_gbps[t]),
            power_watts=float(b["power_watts"][t]),
            delay_us=float(b["delay_us"][t]),
            throughput_gbps=float(b["throughput_gbps"][t]),
            predicted_loss=float(b["predicted_loss"][t]),
            is_safe=bool(b["is_safe"][t]),
            is_efficient=False,
        )

    def _cooldown_progress(self) -> float:
        if self._cooldown_period <= 0:
            return 1.0
        return min(1.0, self._steps_since_switch / self._cooldown_period)

    def _obs(self) -> np.ndarray:
        current = float(self._actual_gbps[self._t])
        forecast = float(self._predicted_gbps[self._t])
        prev_action_flag = float(_UPF_TO_ACTION[self._current_action])
        # History buffer: oldest at left, most recent at right.
        history = list(self._history) if self._history_window > 0 else []
        # Clip very large SEC for numerical safety in the observation.
        sec_obs = min(self._prev_sec, 1e3)
        feats = np.array(
            [current, *history, forecast, prev_action_flag,
             self._prev_q, sec_obs, self._cooldown_progress()],
            dtype=np.float32,
        )
        return feats

    def _terminal_obs(self) -> np.ndarray:
        prev_action_flag = float(_UPF_TO_ACTION[self._current_action])
        history = list(self._history) if self._history_window > 0 else []
        sec_obs = min(self._prev_sec, 1e3)
        feats = np.array(
            [0.0, *history, 0.0, prev_action_flag,
             self._prev_q, sec_obs, self._cooldown_progress()],
            dtype=np.float32,
        )
        return feats
