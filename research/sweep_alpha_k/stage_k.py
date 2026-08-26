#!/usr/bin/env python3
"""Stage per-K forecaster artifacts so a K sweep reads the right arrays.

Without this, MultiAgentUPFEnv derives K from targets.shape[2] and always
loads the K=10 arrays, so every cell of a K sweep would silently be a K=10
run wearing a different label.

Creates data/external/traffic_forecaster_K<K>/ containing symlinks to the
Netflix K<K> predictions/targets, a forecast_eval_summary.json carrying that
K's test MAE, and symlinks to the K-independent auxiliary files.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = Path("/home/ubuntu/UPF_Forecasting/UpfTrafficForecaster/results/cluster_first/Netflix")
BASE = REPO / "data" / "external" / "traffic_forecaster"
AUX = ["cluster_series.npy", "cluster_assignments.parquet",
       "cluster_bs_map.json", "bs_locations.parquet"]


def stage(K: int) -> Path:
    src = SRC / f"K{K}"
    if not src.is_dir():
        raise SystemExit(f"no source artifacts for K={K} at {src}")
    dst = BASE.parent / f"traffic_forecaster_K{K}"
    dst.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        for kind in ("predictions", "targets"):
            s = src / f"{kind}_{split}.npy"
            d = dst / f"{kind}_{split}.npy"
            if d.is_symlink() or d.exists(): d.unlink()
            d.symlink_to(s)

    mae = None
    ev = src / "forecast_eval.json"
    if ev.exists():
        try:
            mae = json.loads(ev.read_text())["forecast"]["model"]["mae"]
        except Exception:
            mae = None
    (dst / "forecast_eval_summary.json").write_text(json.dumps(
        {"service": "Netflix", "best_k": K,
         "results": [{"K": K, "test_mae": mae, "service": "Netflix"}]}, indent=2))

    for name in AUX:                       # K-independent; reuse the base copies
        s = BASE / name
        if not s.exists(): continue
        d = dst / name
        if d.is_symlink() or d.exists(): d.unlink()
        d.symlink_to(s)

    import numpy as np
    t = np.load(dst / "targets_test.npy")
    print(f"  K={K:>3}  targets{str(t.shape):>16}  mae={mae}  -> {dst.relative_to(REPO)}")
    if t.shape[2] != K:
        raise SystemExit(f"FATAL: staged K={K} but array has {t.shape[2]} clusters")
    return dst


if __name__ == "__main__":
    ks = [int(x) for x in (sys.argv[1:] or ["4", "6", "10", "20", "50", "100"])]
    print("staging per-K forecaster artifacts:")
    for k in ks: stage(k)
    print("done")
