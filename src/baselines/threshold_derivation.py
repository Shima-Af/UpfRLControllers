"""Derive controller thresholds from the digital twin's surrogate models.

Ports `derive_thresholds` from UpfDigitalTwin (UPF_NDT) to this repo
so the Phase-7 baselines use the same physics-grounded operating
points as the paper's controller demos, instead of an arbitrary
hardcoded `0.05 Gbps`.

The derivation answers two questions, identically to the upstream
implementation:

  T_qos_limit:  highest USR load at which QoS is still preserved
                (no predicted packet loss, delay within budget).
  T_breakeven:  load at which USR power equals DPDK power.
                Below this USR is more efficient, above DPDK is.

The decision threshold is `min(T_qos_limit, T_breakeven) - safety_margin`.

Anti-flapping (hysteresis band) is derived from forecast accuracy:
  `hysteresis_band = 2 * forecast_MAE`
guaranteeing a single noisy forecast cannot trigger a flip.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ThresholdSpec:
    """All operating points derived from the surrogate models."""

    # Physics-derived
    energy_breakeven_gbps: float
    qos_limit_gbps: float
    delay_limit_gbps: float

    # Decision points used by controllers
    decision_gbps: float       # min(breakeven, qos) - safety_margin
    hysteresis_band_gbps: float  # 2 * forecast_MAE
    t_up_gbps: float           # USR -> DPDK switch
    t_down_gbps: float         # DPDK -> USR switch (= t_up - band)

    # Provenance
    safety_margin_mbps: float
    forecast_mae_gbps: float | None
    derived_from: str


def derive_thresholds(
    twin,
    *,
    safety_margin_mbps: float = 10.0,
    forecast_mae_gbps: float | None = None,
    sweep_max_gbps: float = 0.5,
    sweep_resolution_mbps: float = 1.0,
) -> ThresholdSpec:
    """Compute decision thresholds from the digital twin's surrogate.

    Args:
        twin: ``upf_digital_twin.DigitalTwin`` instance.
        safety_margin_mbps: subtract this from the limiting threshold to
            keep a buffer from the QoS / break-even cliff.
        forecast_mae_gbps: forecast error magnitude in Gbps. If provided,
            ``hysteresis_band = 2 * MAE``. If ``None``, band = 0 (degrades
            to a single-threshold policy).
        sweep_max_gbps: upper bound of the load sweep used for derivation.
        sweep_resolution_mbps: granularity of the sweep.

    Returns:
        ``ThresholdSpec`` with derived operating points and provenance.
    """
    profile = twin._profile  # noqa: SLF001

    n_pts = int(sweep_max_gbps * 1000 / sweep_resolution_mbps)
    loads = np.linspace(sweep_resolution_mbps / 1000, sweep_max_gbps, n_pts)

    dpdk = profile.evaluate_batch("DPDK", loads)
    usr = profile.evaluate_batch(
        "USR", loads, alt_power_watts=dpdk["power_watts"]
    )

    # Energy break-even: first load where USR power >= DPDK power.
    usr_above_dpdk = usr["power_watts"] >= dpdk["power_watts"]
    if usr_above_dpdk.any():
        breakeven = float(loads[int(np.argmax(usr_above_dpdk))])
    else:
        breakeven = float(sweep_max_gbps)

    # QoS limit: last load where USR is_safe.
    usr_safe = usr["is_safe"]
    if usr_safe.all():
        qos_limit = float(sweep_max_gbps)
    elif not usr_safe.any():
        qos_limit = 0.0
    else:
        first_unsafe = int(np.argmax(~usr_safe))
        qos_limit = float(loads[first_unsafe - 1]) if first_unsafe > 0 else 0.0

    # Delay-only limit (informational).
    delay_us = usr["delay_us"]
    delay_budget = profile._delay_budget_us  # noqa: SLF001
    delay_ok = delay_us <= delay_budget
    if delay_ok.all():
        delay_limit = float(sweep_max_gbps)
    elif not delay_ok.any():
        delay_limit = 0.0
    else:
        first_bad = int(np.argmax(~delay_ok))
        delay_limit = float(loads[first_bad - 1]) if first_bad > 0 else 0.0

    # Decision threshold.
    safety_margin_gbps = safety_margin_mbps / 1000.0
    limiting = min(breakeven, qos_limit)
    decision = max(0.0, limiting - safety_margin_gbps)

    # Hysteresis band.
    band = 2 * forecast_mae_gbps if forecast_mae_gbps is not None else 0.0
    t_up = decision
    t_down = max(0.0, decision - band)

    derived_from = (
        f"min(breakeven={breakeven * 1000:.1f} Mbps, "
        f"qos={qos_limit * 1000:.1f} Mbps) "
        f"- {safety_margin_mbps:.0f} Mbps margin"
    )
    return ThresholdSpec(
        energy_breakeven_gbps=breakeven,
        qos_limit_gbps=qos_limit,
        delay_limit_gbps=delay_limit,
        decision_gbps=decision,
        hysteresis_band_gbps=band,
        t_up_gbps=t_up,
        t_down_gbps=t_down,
        safety_margin_mbps=safety_margin_mbps,
        forecast_mae_gbps=forecast_mae_gbps,
        derived_from=derived_from,
    )


def load_forecast_mae_gbps(
    forecast_eval_summary_path: Path,
    *,
    alpha_gbps_per_norm: float,
    K: int,
) -> float | None:
    """Load the K-cluster MAE from the forecaster's eval summary JSON.

    The summary stores MAE in normalised units; multiply by ``alpha`` to
    convert to Gbps.
    """
    if not forecast_eval_summary_path.exists():
        return None
    summary = json.loads(forecast_eval_summary_path.read_text())
    for row in summary.get("results", []):
        if int(row.get("K", -1)) == int(K):
            mae_norm = float(row.get("test_mae", 0.0))
            return mae_norm * alpha_gbps_per_norm
    return None
