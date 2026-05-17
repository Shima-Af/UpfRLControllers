"""Hysteresis threshold policy — stateful, two-threshold, with cooldown.

Direct port of the policy from UpfDigitalTwin (UPF_NDT). One instance
holds state for ONE cluster (last action + cooldown counter); the
``MultiAgentHysteresis`` wrapper composes K instances for the
multi-agent UPF env.

Switching rule:
    USR  -> DPDK  when predicted_load_gbps >= t_up_gbps
    DPDK -> USR   when predicted_load_gbps <= t_down_gbps
After any switch the controller is locked in the new mode for
``cooldown_steps`` decision steps.

Initial action: chosen by midpoint of the band on the first call.
"""

from __future__ import annotations

import numpy as np


class HysteresisPolicy:
    """Per-cluster stateful hysteresis controller (binary DPDK / USR)."""

    DPDK = 0
    USR = 1

    def __init__(
        self,
        t_up_gbps: float,
        t_down_gbps: float,
        cooldown_steps: int = 1,
    ) -> None:
        if t_down_gbps > t_up_gbps:
            raise ValueError(
                f"t_down_gbps ({t_down_gbps}) must be <= t_up_gbps ({t_up_gbps})"
            )
        self.t_up_gbps = float(t_up_gbps)
        self.t_down_gbps = float(t_down_gbps)
        self.cooldown_steps = int(cooldown_steps)
        self._last_action: int | None = None
        self._cooldown_left = 0

    def reset(self) -> None:
        self._last_action = None
        self._cooldown_left = 0

    def act(self, predicted_load_gbps: float) -> int:
        load = float(predicted_load_gbps)

        # Initial step: pick by midpoint of band.
        if self._last_action is None:
            decision_pt = 0.5 * (self.t_up_gbps + self.t_down_gbps)
            self._last_action = self.USR if load < decision_pt else self.DPDK
            return self._last_action

        # In cooldown: must keep last action regardless of thresholds.
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return self._last_action

        # Try to switch.
        new_action = self._last_action
        if self._last_action == self.USR and load >= self.t_up_gbps:
            new_action = self.DPDK
        elif self._last_action == self.DPDK and load <= self.t_down_gbps:
            new_action = self.USR

        if new_action != self._last_action:
            self._cooldown_left = self.cooldown_steps
        self._last_action = new_action
        return new_action


class MultiAgentHysteresis:
    """K independent ``HysteresisPolicy`` instances, one per cluster.

    Acts as a PettingZoo-style policy: obs_dict -> action_dict. Reads
    the 1-step forecast from each agent's observation (index 9 in the
    Phase-2 obs schema: ``[current, 8 history, forecast, prev_action,
    prev_Q, prev_SEC, cooldown_progress]``).
    """

    FORECAST_IDX = 9

    def __init__(
        self,
        agents: list[str],
        t_up_gbps: float,
        t_down_gbps: float,
        cooldown_steps: int = 1,
    ) -> None:
        self.agents = list(agents)
        self._controllers: dict[str, HysteresisPolicy] = {
            a: HysteresisPolicy(t_up_gbps, t_down_gbps, cooldown_steps)
            for a in self.agents
        }

    def reset(self) -> None:
        for c in self._controllers.values():
            c.reset()

    def __call__(self, obs_dict: dict[str, np.ndarray]) -> dict[str, int]:
        out: dict[str, int] = {}
        for agent, obs in obs_dict.items():
            forecast = float(obs[self.FORECAST_IDX])
            out[agent] = self._controllers[agent].act(forecast)
        return out
