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

    Observation (Box, shape (7,), float32):
        [actual_load_gbps, predicted_load_gbps, prev_action,
         prev_power_watts, prev_safety_flag, timestep_progress,
         cooldown_progress]

        ``cooldown_progress`` = min(1, steps_since_last_switch / cooldown_period)
        — 0.0 immediately after a switch, 1.0 once the agent is "free"
        to switch again without paying the soft-cooldown surcharge.

    Action (Discrete(2)):
        0 -> DPDK, 1 -> USR

    Reward:
        -(energy_weight * energy_wh
          + qos_lambda * max(0, performance_threshold - performance)
          + switching_weight * switching_energy_wh)

        ``performance`` is a continuous [0, 1] score derived from the
        digital twin's continuous outputs. It is 1.0 while delay and loss
        are inside budget, then declines linearly with how far they
        exceed it:

            delay_excess = max(0, delay_us - delay_budget_us)
            delay_score  = max(0, 1 - delay_excess / delay_budget_us)
            loss_excess  = max(0, predicted_loss - max_loss_pkts_per_interval)
            loss_score   = max(0, 1 - loss_excess / max_loss_pkts_per_interval)
            performance  = min(delay_score, loss_score)

        This mirrors the graded QoS shape used by prior in-house models —
        a small dip below threshold incurs a small penalty, not a cliff,
        and being inside budget incurs no QoS penalty at all. The twin's
        binary ``is_safe`` flag remains available in ``info`` for logging.
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
        from upf_digital_twin.twin.upf_profile import UPFResult  # noqa: WPS433

        self._UPFResult = UPFResult  # stash for _result_at()

        self._twin = DigitalTwin(
            scenario_cfg=self.scenario_cfg,
            paths_cfg=self.paths_cfg,
            project_root=self._root,
        )
        self._step_h = float(self._twin.step_h)

        # --- Pre-compute surrogate predictions for the entire episode ---
        # The surrogate models (sklearn pipelines) are the dominant cost
        # of env.step() — ~95 ms per evaluate_action() call. Since the
        # load trajectory is fixed at __init__, we can batch all 1009
        # steps' worth of predictions once and look them up in O(1)
        # during step(). compute_step() still runs per-step for its
        # stateful switching-cost accounting.
        self._dpdk_batch = self._twin.evaluate_batch(
            "DPDK", self._actual_gbps
        )
        self._usr_batch = self._twin.evaluate_batch(
            "USR", self._actual_gbps
        )

        # --- Reward weights ---
        rw = self.scenario_cfg.get("reward_weights", {})
        self._w_energy = float(rw.get("energy_weight", 1.0))
        self._qos_lambda = float(rw.get("qos_lambda", 5.0))
        self._perf_threshold = float(rw.get("performance_threshold", 0.90))
        self._w_switch = float(rw.get("switching_weight", 0.1))
        self._type_switch_cost = float(rw.get("type_switch_cost", 0.0))
        self._cooldown_period = int(rw.get("cooldown_period", 0))
        self._cooldown_cost = float(rw.get("cooldown_cost", 0.0))
        if self._cooldown_period < 0:
            raise ValueError("cooldown_period must be >= 0")

        # --- QoS budgets, used to normalize delay/loss into [0, 1] ---
        qos_cfg = self.scenario_cfg.get("upf", {}).get("qos_budget", {})
        self._delay_budget_us = float(qos_cfg.get("delay_budget_us", 200.0))
        self._max_loss_pkts = float(qos_cfg.get("max_loss_pkts_per_interval", 5.0))
        if self._delay_budget_us <= 0 or self._max_loss_pkts <= 0:
            raise ValueError(
                "delay_budget_us and max_loss_pkts_per_interval must be > 0 "
                "to normalize QoS scores; got "
                f"{self._delay_budget_us=}, {self._max_loss_pkts=}"
            )

        # --- Initial action ---
        ia = self.scenario_cfg.get("initial_action", "DPDK")
        if not isinstance(ia, str) or ia not in ("DPDK", "USR"):
            ia = "DPDK"
        self._initial_action = ia

        # --- Spaces ---
        self.action_space = spaces.Discrete(2)
        # Generous finite bounds; load and power are non-negative in practice.
        high = np.array([1e3, 1e3, 1.0, 1e3, 1.0, 1.0, 1.0], dtype=np.float32)
        low = np.zeros(7, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # --- Episode state ---
        self._current_action: str = self._initial_action
        self._t: int = 0
        self._prev_power: float = 0.0
        self._prev_safety: float = 1.0
        # Cooldown bookkeeping: how many steps have elapsed since the last
        # observed UPF type change. Starts large so the first step is free.
        self._steps_since_switch: int = max(1, self._cooldown_period)

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
        self._steps_since_switch = max(1, self._cooldown_period)
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

        old_r = self._result_at(self._current_action, self._t)
        new_r = (
            self._result_at(requested, self._t)
            if requested != self._current_action
            else old_r
        )
        composite, sw_energy, new_current, pending = self._twin.compute_step(
            self._current_action, requested, old_r, new_r,
        )
        realised = self._current_action if pending else requested
        prev_realised = self._current_action
        if not pending:
            self._current_action = new_current

        # Did the *realised* UPF type change on this step? Use the realised
        # action so a pending activation (which the twin masks internally)
        # does not count as a switch.
        type_changed = realised != prev_realised

        # Soft cooldown: penalty when switching while still within the
        # cooldown window, scaled linearly so it tapers to zero as we
        # approach the end of the window.
        if self._cooldown_period > 0 and type_changed:
            steps_left = max(
                0, self._cooldown_period - self._steps_since_switch
            )
            cooldown_pen = (
                self._cooldown_cost * steps_left / self._cooldown_period
            )
        else:
            cooldown_pen = 0.0
        type_switch_pen = self._type_switch_cost if type_changed else 0.0

        energy_wh = float(composite.power_watts) * self._step_h

        # Graded QoS score: continuous performance ∈ [0, 1] derived from
        # the twin's continuous delay/loss predictions. Score is 1.0
        # while we're inside budget (any headroom is fine), then declines
        # linearly with how far we've exceeded budget. min() picks the
        # limiting metric.
        delay_us = float(composite.delay_us)
        loss_pkts = float(composite.predicted_loss)
        delay_excess = max(0.0, delay_us - self._delay_budget_us)
        loss_excess = max(0.0, loss_pkts - self._max_loss_pkts)
        delay_score = max(0.0, 1.0 - delay_excess / self._delay_budget_us)
        loss_score = max(0.0, 1.0 - loss_excess / self._max_loss_pkts)
        performance = min(delay_score, loss_score)
        qos_penalty = self._qos_lambda * max(
            0.0, self._perf_threshold - performance
        )

        reward = -(
            self._w_energy * energy_wh
            + qos_penalty
            + self._w_switch * float(sw_energy)
            + type_switch_pen
            + cooldown_pen
        )

        info: dict[str, Any] = {
            "selected_upf": realised,
            "requested_upf": requested,
            "actual_load_gbps": load,
            "predicted_load_gbps": predicted,
            "power_watts": float(composite.power_watts),
            "delay_us": delay_us,
            "predicted_loss": loss_pkts,
            "performance": performance,
            "qos_penalty": qos_penalty,
            "switching_energy_wh": float(sw_energy),
            "type_switch_penalty": type_switch_pen,
            "cooldown_penalty": cooldown_pen,
            "steps_since_switch": int(self._steps_since_switch),
            "is_safe": bool(composite.is_safe),
            "timestep": int(self._t),
            "cluster_idx": int(self.cluster_idx),
        }

        # Bookkeeping for the *next* observation.
        self._prev_power = float(composite.power_watts)
        self._prev_safety = 1.0 if bool(composite.is_safe) else 0.0
        # Cooldown counter: reset on a realised switch, otherwise +1.
        if type_changed:
            self._steps_since_switch = 0
        else:
            self._steps_since_switch += 1
        self._t += 1

        terminated = False
        truncated = self._t >= self._N
        obs = self._terminal_obs() if truncated else self._obs()

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _result_at(self, action: str, t: int) -> Any:
        """Reconstruct a UPFResult from the precomputed batch arrays.

        Equivalent to ``self._twin.evaluate_action(action, load[t])`` but
        without re-running the sklearn surrogate at every step.
        """
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
                self._cooldown_progress(),
            ],
            dtype=np.float32,
        )

    def _terminal_obs(self) -> np.ndarray:
        prev_action_flag = float(_UPF_TO_ACTION[self._current_action])
        return np.array(
            [
                0.0,
                0.0,
                prev_action_flag,
                self._prev_power,
                self._prev_safety,
                1.0,
                self._cooldown_progress(),
            ],
            dtype=np.float32,
        )
