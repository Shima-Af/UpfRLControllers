"""Shared paths, constants and data loading for the held-out twin validation.

Nothing in this directory writes to the sibling repositories; they are read
only. Every input is hash-checked against the DVC lock of the profiling
campaign at tag thesis-v1 (commit a73870f), which is the revision the twin
(UPF_NDT v0.4.0, commit 95de456) and the controller (models.dvc) are pinned to.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]                                   # UpfRLControllers
PROFILING = Path(os.environ.get(
    "UPF_PROFILING_REPO", REPO_ROOT.parent / "UpfProfilingCampaign" / "UpfProfilingCampaign"))
PINNED_MODELS = REPO_ROOT / "data" / "external" / "profiling_twin" / "models"
SCENARIO = REPO_ROOT / "configs" / "scenario_rl.yaml"
OUT = HERE

# md5 values recorded in UpfProfilingCampaign dvc.lock at a73870f (thesis-v1)
EXPECTED_MD5 = {
    "data/interim/merged.csv":     "3d6d4c6ec7b7e3f9e18d2482563aa023",
    "data/processed/features.csv": "04497319167520a102d66985238ad005",
    "src/features.py":             "7bd9edc0143b47ec52a8b325ea42a3b8",
    "src/train.py":                "95a7ec21ea37f02aa3f1ee837f56e9e4",
    "src/merge.py":                "ab49fd74f3a4d492182c45436369ef52",
    "src/ingest.py":               "4d24c1f9d2c5c37dec10da396da70607",
}
EXPECTED_MODELS_DIR_MD5 = "a92ad2caa7f058b85e582261594b17b4"   # models.dvc + dvc.lock

# Column names (UpfProfilingCampaign params.yaml, train section)
DL_TX = "gtpu_kbitss_dn__kbits_tx_s"            # offered downlink load, kbit/s
UL_TX = "gtpu_kbitss_ngran__gtpu_kbits_tx_s"    # offered uplink (N3) load, kbit/s
LITE_FEATURES = [DL_TX, UL_TX]
T_THR = "throughput_gbps"
T_CPU = "cpu_pct"
T_LOSS = "gtpu_packets_dn__packets_lost_delta"
T_DELAY = "downlink_one_way_delay_distribution__weighted_mean_delay_us"
T_POWER = "power_watts"
L1_TARGETS = [T_THR, T_CPU, T_LOSS, T_DELAY]    # order == twin's L2 column_stack

# Twin variant map (upf_digital_twin/twin/upf_profile.py::_VARIANT_MAP)
UPFS = {"DPDK": ("dpdk", 1), "USR": ("usr_full", 0)}   # name -> (manifest variant, is_dpdk)

# Design grid of the profiling campaign (offered DL load, Gbps). Verified in
# load_runs(): every run maps to one level, 5 runs per level, one per sweep.
DESIGN_LEVELS_GBPS = np.array([0.0, 1e-4, 1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 0.01, 0.02,
                               0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0,
                               2.0, 3.0, 4.0, 5.0])

# Candidate model spaces, copied verbatim from UpfProfilingCampaign src/train.py
# (build_search) and params.yaml (train.models, search_n_iter, random_state).
RANDOM_STATE = 42
SEARCH_N_ITER = 20
SPACES = {
    "ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    "random_forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [6, 10, 15, None],
        "model__min_samples_leaf": [2, 5, 10],
        "model__max_features": [0.5, 0.7, 1.0],
    },
    "gradient_boosting": {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth": [3, 5, 7],
        "model__subsample": [0.7, 0.85, 1.0],
    },
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dvc_dir_md5(root: Path) -> tuple[str, int]:
    """DVC 3 directory hash: md5 of the sorted JSON list of {md5, relpath}."""
    entries = []
    for dp, _, files in os.walk(root):
        for f in files:
            p = Path(dp) / f
            entries.append({"md5": md5(p), "relpath": p.relative_to(root).as_posix()})
    entries.sort(key=lambda e: e["relpath"])
    blob = json.dumps(entries, sort_keys=True).encode()
    return hashlib.md5(blob).hexdigest(), len(entries)


def load_scenario() -> dict:
    import yaml
    with open(SCENARIO) as fh:
        return yaml.safe_load(fh)


def qos_budget() -> tuple[float, float]:
    q = load_scenario()["upf"]["qos_budget"]
    return float(q["delay_budget_us"]), float(q["max_loss_pkts_per_interval"])


def _profiling_features_module():
    spec = importlib.util.spec_from_file_location("profiling_features", PROFILING / "src" / "features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rebuild_features_with_run() -> pd.DataFrame:
    """Re-run the profiling featurize stage on merged.csv, keeping run_dir.

    features.py drops run_dir in cleanup(); we replicate cleanup() except for
    that drop, then assert the result equals the DVC-locked features.csv
    column for column. The returned frame is row-aligned with features.csv.
    """
    fx = _profiling_features_module()
    df = pd.read_csv(PROFILING / "data" / "interim" / "merged.csv")
    df = fx.drop_uninformative_columns(df)
    df = fx.add_unit_conversions(df)
    df = fx.compute_targets(df)
    df = fx.add_rolling_features(df, 30)          # params.yaml features.rolling_window_sec
    df = fx.add_derived_features(df)
    df = fx.encode_variant(df)
    run_dir = df["run_dir"].copy()
    ts = df["Timestamp epoch ms"].copy()
    clean, _ = fx.cleanup(df, "sec_total")        # params.yaml features.target
    clean = clean.copy()
    clean.insert(len(clean.columns), "run_dir", run_dir.loc[clean.index].values)
    clean.insert(len(clean.columns), "timestamp_ms", ts.loc[clean.index].values)
    return clean.reset_index(drop=True)


def load_runs(verify: bool = True) -> pd.DataFrame:
    """features.csv + verified run_dir, nominal load level and sweep index."""
    feats = pd.read_csv(PROFILING / "data" / "processed" / "features.csv")
    rebuilt = rebuild_features_with_run()
    if verify:
        assert len(rebuilt) == len(feats), (len(rebuilt), len(feats))
        for c in feats.columns:
            a, b = feats[c].to_numpy(), rebuilt[c].to_numpy()
            if np.issubdtype(a.dtype, np.number):
                # atol absorbs ~1e-15 round-off in pandas' rolling std (not a model input)
                ok = np.allclose(a, b.astype(float), rtol=1e-12, atol=1e-12, equal_nan=True)
            else:
                ok = (pd.Series(a).astype(str) == pd.Series(b).astype(str)).all()
            assert ok, f"rebuilt column {c} differs from features.csv"
    df = feats.copy()
    df["run_dir"] = rebuilt["run_dir"].values
    df["timestamp_ms"] = rebuilt["timestamp_ms"].values
    df["upf"] = np.where(df["is_dpdk"] == 1, "DPDK", "USR")
    df["load_gbps"] = df[DL_TX] / 1e6

    runs = (df.groupby("run_dir")
              .agg(upf=("upf", "first"), n=("load_gbps", "size"),
                   load_gbps=("load_gbps", "mean"), t0=("timestamp_ms", "min"))
              .reset_index())
    lv = np.log10(runs["load_gbps"].to_numpy()[:, None] + 1e-5)
    grid = np.log10(DESIGN_LEVELS_GBPS[None, :] + 1e-5)
    runs["level_gbps"] = DESIGN_LEVELS_GBPS[np.abs(lv - grid).argmin(1)]
    runs["sweep"] = runs.groupby(["upf", "level_gbps"])["t0"].rank(method="first").astype(int) - 1
    if verify:
        per = runs.groupby(["upf", "level_gbps"]).size()
        assert (per == 5).all(), per[per != 5]
        # sweeps must be chronological passes: every run of sweep s starts
        # after every run of sweep s-1 within the same UPF campaign
        for upf, g in runs.groupby("upf"):
            span = g.groupby("sweep")["t0"].agg(["min", "max"]).sort_index()
            assert (span["min"].to_numpy()[1:] > span["max"].to_numpy()[:-1]).all(), upf
        rel = np.abs(runs["load_gbps"] - runs["level_gbps"]) / runs["level_gbps"].clip(lower=1e-5)
        runs["rel_dev_from_level"] = rel
    return df.merge(runs[["run_dir", "level_gbps", "sweep"]], on="run_dir", how="left"), runs


def near_boundary_band() -> tuple[float, float]:
    """Pre-declared reporting band: within a factor of 3 of the twin-derived
    USR QoS limit lambda_qos = 149 Mbps used in the manuscript."""
    lam = 0.149
    return lam / 3.0, lam * 3.0
