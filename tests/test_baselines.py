"""Tests for stateless baseline controllers.

These tests do not need the digital twin — they exercise pure
decision logic in src/baselines/hysteresis.py.
"""

from __future__ import annotations

import pytest


def test_hysteresis_validates_band():
    """t_down > t_up is nonsensical and must raise."""
    from src.baselines.hysteresis import HysteresisPolicy

    with pytest.raises(ValueError):
        HysteresisPolicy(t_up_gbps=0.05, t_down_gbps=0.10)


def test_hysteresis_no_flap_within_band():
    """Loads inside the band must not flip the action once initialised."""
    from src.baselines.hysteresis import HysteresisPolicy

    h = HysteresisPolicy(t_up_gbps=0.10, t_down_gbps=0.05, cooldown_steps=1)
    # First call (midpoint = 0.075): load 0.08 -> DPDK (load >= midpoint).
    first = h.act(0.08)
    # Loads strictly inside the band: no switch.
    for f in (0.06, 0.07, 0.08, 0.09):
        assert h.act(f) == first, f"hysteresis flipped at load={f}"


def test_hysteresis_flips_when_load_crosses_lower_threshold():
    from src.baselines.hysteresis import HysteresisPolicy

    h = HysteresisPolicy(t_up_gbps=0.10, t_down_gbps=0.05, cooldown_steps=0)
    h.act(0.20)  # initialise high -> DPDK
    a = h.act(0.01)  # below t_down -> should switch to USR
    assert a == HysteresisPolicy.USR


def test_hysteresis_cooldown_locks_action():
    """During cooldown, the action must not change even if load crosses."""
    from src.baselines.hysteresis import HysteresisPolicy

    h = HysteresisPolicy(t_up_gbps=0.10, t_down_gbps=0.05, cooldown_steps=2)
    h.act(0.20)  # DPDK
    a1 = h.act(0.01)  # switch to USR, cooldown=2
    assert a1 == HysteresisPolicy.USR
    a2 = h.act(0.50)  # would normally flip to DPDK, but cooldown
    assert a2 == HysteresisPolicy.USR
