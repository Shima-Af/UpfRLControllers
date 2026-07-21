"""Executable form of CONTRACTS.md.

Every claim in CONTRACTS.md about a cross-repo artifact's shape, dtype, units,
or identity is asserted here. If this file and CONTRACTS.md disagree, this file
wins and the doc is stale.

The four defects found during the 2026-07 consolidation (`selected_k` 8 vs 10,
`service` video vs Netflix, `alpha` 0.12 vs 1.0, orphaned `switching_costs.yaml`)
would each have been caught by one of these tests before reaching a result.

Skips cleanly when ``data/external/`` is unpopulated, per the repo convention.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

# --- Contract constants (mirror CONTRACTS.md §1 and §4) ---------------------
K = 10
H = 4
N_TRAIN, N_VAL, N_TEST = 5073, 1009, 1009
SERVICE = "Netflix"
N_SITES = 965

CANON = {
    "alpha": 1.0,
    "delay_budget_us": 200.0,
    "max_loss_pkts_per_interval": 5.0,
    "prewarm_enabled": False,
    "prewarm_standby_w": 0.0,
    "time_step_minutes": 15,
}

SPLITS = {"train": N_TRAIN, "val": N_VAL, "test": N_TEST}


@pytest.fixture(scope="module")
def ext(repo_root):
    return repo_root / "data" / "external"


@pytest.fixture(scope="module")
def scenario(repo_root):
    return yaml.safe_load((repo_root / "configs" / "scenario_rl.yaml").read_text())


def _need(has_twin_artifacts):
    if not has_twin_artifacts:
        pytest.skip("data/external/ not populated — run `dvc pull`")


# --- §1 Forecaster contract -------------------------------------------------


@pytest.mark.parametrize("split,n", SPLITS.items())
@pytest.mark.parametrize("kind", ["predictions", "targets"])
def test_forecast_array_shape_and_dtype(ext, has_twin_artifacts, kind, split, n):
    """Axis order is (N, H, K) — never (N, K, H) — and dtype is float32."""
    _need(has_twin_artifacts)
    p = ext / "traffic_forecaster" / f"{kind}_{split}.npy"
    if not p.exists():
        pytest.skip(f"{p.name} absent")
    a = np.load(p)
    assert a.shape == (n, H, K), f"{p.name}: expected {(n, H, K)}, got {a.shape}"
    assert a.dtype == np.float32


def test_predictions_and_targets_align(ext, has_twin_artifacts):
    _need(has_twin_artifacts)
    for split in SPLITS:
        pr = ext / "traffic_forecaster" / f"predictions_{split}.npy"
        tg = ext / "traffic_forecaster" / f"targets_{split}.npy"
        if not (pr.exists() and tg.exists()):
            continue
        assert np.load(pr).shape == np.load(tg).shape, f"{split}: shape mismatch"


def test_traffic_is_dimensionless_not_unit_normalised(ext, has_twin_artifacts):
    """Values are SUMMED per-BS dl_norm, so they exceed 1.0 by design.

    Guards the misreading that these are [0,1]-normalised, which is what makes
    `alpha` look like a recoverable scaler when it is a declared assumption.
    """
    _need(has_twin_artifacts)
    t = np.load(ext / "traffic_forecaster" / "targets_test.npy")
    assert t.max() > 1.0, "expected summed dl_norm > 1.0; is this unit-normalised?"
    assert t.min() >= 0.0, "ground-truth load must be non-negative"


def test_predictions_may_be_slightly_negative(ext, has_twin_artifacts):
    """Documents a real quirk: forecasts can dip below zero.

    Consumers converting to Gbps must tolerate or clip. Fails loudly if the
    magnitude ever grows beyond a rounding-scale artifact.
    """
    _need(has_twin_artifacts)
    p = ext / "traffic_forecaster" / "predictions_test.npy"
    if not p.exists():
        pytest.skip("predictions_test absent")
    assert np.load(p).min() > -0.1, "negative forecast is no longer negligible"


def test_cluster_metadata_consistent(ext, has_twin_artifacts):
    _need(has_twin_artifacts)
    cs = np.load(ext / "traffic_forecaster" / "cluster_series.npy")
    assert cs.shape[0] == K, f"cluster_series has {cs.shape[0]} rows, expected K={K}"

    ca = pd.read_parquet(ext / "traffic_forecaster" / "cluster_assignments.parquet")
    assert list(ca.columns) == ["site_id", "cluster_id"]
    assert len(ca) == N_SITES
    assert ca.cluster_id.nunique() == K

    bs = ext / "traffic_forecaster" / "bs_locations.parquet"
    if bs.exists():
        assert len(pd.read_parquet(bs)) == N_SITES

    with open(ext / "traffic_forecaster" / "cluster_bs_map.json") as fh:
        assert len(json.load(fh)) == K


def test_forecast_summary_service_matches_scenario(ext, scenario, has_twin_artifacts):
    """The `service` mismatch bug: scenario said 'video', data says 'Netflix'.

    UpfDigitalTwin's traffic_loader raises on this; the RL env dodged it only
    by loading arrays itself.
    """
    _need(has_twin_artifacts)
    p = ext / "traffic_forecaster" / "forecast_eval_summary.json"
    if not p.exists():
        pytest.skip("forecast_eval_summary absent")
    with open(p) as fh:
        summary = json.load(fh)
    assert summary["service"].lower() == SERVICE.lower()
    assert scenario["traffic"]["service"].lower() == summary["service"].lower()


def test_scenario_selected_k_matches_arrays(ext, scenario, has_twin_artifacts):
    """The `selected_k` bug: scenario said 8, arrays are K=10."""
    _need(has_twin_artifacts)
    t = ext / "traffic_forecaster" / "targets_test.npy"
    if not t.exists():
        pytest.skip("targets_test absent")
    assert int(scenario["traffic"]["selected_k"]) == np.load(t).shape[2] == K


# --- §2 Profiling contract --------------------------------------------------


def test_profiling_manifest_covers_required_lite_models(ext, has_twin_artifacts):
    """The twin loads `lite` exclusively — NetMob load supplies throughput only."""
    _need(has_twin_artifacts)
    with open(ext / "profiling_twin" / "models" / "manifest.json") as fh:
        man = json.load(fh)
    for variant in ("dpdk", "usr_full", "usr_safe"):
        assert f"{variant}__layer2__power_watts__lite" in man, variant
        assert f"{variant}__layer1__throughput_gbps__lite" in man, variant


def test_manifest_paths_need_separator_normalisation(ext, has_twin_artifacts):
    """Manifest paths are Windows-style; consumers must normalise.

    Pinned so a future POSIX-path rewrite upstream is a visible change.
    """
    _need(has_twin_artifacts)
    with open(ext / "profiling_twin" / "models" / "manifest.json") as fh:
        man = json.load(fh)
    paths = [e["path"] for e in man.values() if isinstance(e, dict) and "path" in e]
    assert paths and all("\\" in p for p in paths), "separator convention changed"


def test_switching_costs_durations_present(ext, has_twin_artifacts):
    """Durations are the only portable quantity — absolute Wh are not.

    Measured on a different host than the profiling campaign (see the file's
    own provenance header).
    """
    _need(has_twin_artifacts)
    p = ext / "profiling_twin" / "switching_costs.yaml"
    if not p.exists():
        pytest.skip("switching_costs.yaml absent")
    d = yaml.safe_load(p.read_text())["activation_duration_s"]
    assert d["dpdk"] == 24.0 and d["usr"] == 3.3


def test_operating_ranges_present(ext, has_twin_artifacts):
    """`operating_ranges` is not derivable from anything else in any repo.

    It is also the source of the historical alpha=0.12 anchor
    (danger_threshold_gbps), so its loss would be unrecoverable.
    """
    _need(has_twin_artifacts)
    p = ext / "profiling_twin" / "params.yaml"
    if not p.exists():
        pytest.skip("params.yaml absent")
    r = yaml.safe_load(p.read_text())["operating_ranges"]
    assert r["usr"]["danger_threshold_gbps"] == 0.70
    assert r["usr"]["safe_capacity_gbps"] == 0.69
    assert r["dpdk"]["power_watts_idle"] == 0.82


# --- §4 Scenario canon ------------------------------------------------------


def test_scenario_canon_values(scenario):
    """alpha=1.0 is a DECLARED assumption, not a measurement.

    NetMob is dimensionless and the forecaster sums per-BS dl_norm, so no
    scaler exists to recover. Changing this changes the operating regime, not
    just the units.
    """
    assert scenario["traffic"]["alpha"] == CANON["alpha"]
    qos = scenario["upf"]["qos_budget"]
    assert qos["delay_budget_us"] == CANON["delay_budget_us"]
    assert qos["max_loss_pkts_per_interval"] == CANON["max_loss_pkts_per_interval"]
    pre = scenario["upf_switching"]["prewarm"]
    assert pre["enabled"] is CANON["prewarm_enabled"]
    assert pre["standby_power_watts"] == CANON["prewarm_standby_w"]
    assert scenario["traffic"]["time_step_minutes"] == CANON["time_step_minutes"]


def test_max_loss_nonzero_for_graded_score(scenario):
    """The graded QoS score divides by this; 0.0 would raise."""
    assert scenario["upf"]["qos_budget"]["max_loss_pkts_per_interval"] > 0


def test_twin_pin_consistent_across_dependency_files(repo_root):
    """pyproject.toml and requirements.txt pinned different versions once."""
    pins = set()
    for name in ("pyproject.toml", "requirements.txt"):
        for line in (repo_root / name).read_text().splitlines():
            if "UpfDigitalTwin.git@" in line:
                pins.add(line.split("UpfDigitalTwin.git@")[1].strip(' ",\''))
    assert len(pins) == 1, f"twin pin drifted across dependency files: {pins}"
