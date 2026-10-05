"""Step 1 — provenance checks and reconstruction of the ORIGINAL split.

Outputs
  provenance.json                 input hashes, versions, dataset/run counts
  run_metadata.csv                one row per profiling run (220)
  original_split_reproduction.csv pinned pickles re-scored on the reconstructed
                                  train.py split vs the metrics in manifest.json

The original split is train_test_split(test_size=0.2, random_state=42) applied
row-wise per variant (UpfProfilingCampaign src/train.py::main). It is
reconstructed here ONLY to check that the manuscript numbers come from it and
to quantify its leakage; it is not used as held-out evidence.
"""
from __future__ import annotations

import json
import platform
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import common as C

warnings.filterwarnings("ignore")


def load_pinned(variant: str, layer: str, target: str, mv: str):
    suffix = "__lite" if mv == "lite" else ""
    return joblib.load(C.PINNED_MODELS / layer / f"{variant}__{target}{suffix}.pkl")


def score(y, p):
    return dict(r2=float(r2_score(y, p)), mae=float(mean_absolute_error(y, p)),
                rmse=float(np.sqrt(mean_squared_error(y, p))))


def main():
    prov = {"python": sys.version.split()[0], "platform": platform.platform(),
            "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}

    # ── input hashes ────────────────────────────────────────────────────────
    hashes = {}
    for rel, exp in C.EXPECTED_MD5.items():
        got = C.md5(C.PROFILING / rel)
        hashes[rel] = {"md5": got, "expected_thesis_v1": exp, "match": got == exp}
    dir_md5, nfiles = C.dvc_dir_md5(C.PINNED_MODELS)
    hashes["pinned models dir (controller data/external/profiling_twin/models)"] = {
        "md5": dir_md5 + ".dir", "nfiles": nfiles,
        "expected_thesis_v1": C.EXPECTED_MODELS_DIR_MD5 + ".dir",
        "match": dir_md5 == C.EXPECTED_MODELS_DIR_MD5}
    hashes["configs/scenario_rl.yaml"] = {"md5": C.md5(C.SCENARIO)}
    prov["input_hashes"] = hashes
    assert all(v.get("match", True) for v in hashes.values()), hashes

    # ── dataset with verified run grouping ──────────────────────────────────
    df, runs = C.load_runs(verify=True)
    runs.to_csv(C.OUT / "run_metadata.csv", index=False)
    prov["dataset"] = {
        "rows": int(len(df)),
        "rows_by_upf": df["upf"].value_counts().to_dict(),
        "runs": int(runs["run_dir"].nunique()),
        "runs_by_upf": runs["upf"].value_counts().to_dict(),
        "samples_per_run_min_median_max": [int(runs.n.min()), float(runs.n.median()), int(runs.n.max())],
        "load_levels_per_upf": int(runs.groupby("upf")["level_gbps"].nunique().iloc[0]),
        "runs_per_level": 5, "sweeps_per_upf": 5,
        "reporting_interval_s": 3.0,
        "max_rel_dev_run_load_from_level": float(runs.loc[runs.level_gbps >= 0.01, "rel_dev_from_level"].max()),
        "offered_dl_load_range_gbps": {u: [float(g.load_gbps.min()), float(g.load_gbps.max())]
                                       for u, g in df.groupby("upf")},
        "campaign_time_span": {u: [str(pd.to_datetime(g.timestamp_ms.min(), unit="ms")),
                                   str(pd.to_datetime(g.timestamp_ms.max(), unit="ms"))]
                               for u, g in df.groupby("upf")},
        "ul_over_dl_offered_median": {u: float(np.nanmedian(g[C.UL_TX] / g[C.DL_TX].replace(0, np.nan)))
                                      for u, g in df.groupby("upf")},
        "run_dir_rebuilt_from_merged_csv_and_verified_equal_to_features_csv": True,
    }
    # untracked helper file left by an earlier session: check it agrees
    fwr = C.PROFILING / "data" / "processed" / "features_with_run.csv"
    if fwr.exists():
        other = pd.read_csv(fwr, usecols=["run_dir"])
        prov["dataset"]["untracked_features_with_run_csv_run_dir_agrees"] = bool((other["run_dir"].values == df["run_dir"].values).all())

    # ── reconstruct original split and re-score pinned pickles ──────────────
    manifest = json.loads((C.PINNED_MODELS / "manifest.json").read_text())
    feats_full = manifest["usr_full__layer1__throughput_gbps__full"]["features"]
    base = pd.read_csv(C.PROFILING / "data" / "processed" / "features.csv")
    rows = []
    leakage = {}
    for upf, (variant, is_dpdk) in C.UPFS.items():
        sub = base[base["is_dpdk"] == is_dpdk].copy()
        tr, te = train_test_split(sub, test_size=0.2, random_state=C.RANDOM_STATE)
        run_of = df["run_dir"]
        te_runs, tr_runs = run_of.loc[te.index], run_of.loc[tr.index]
        sib = tr_runs.value_counts()
        leakage[upf] = {
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "test_runs": int(te_runs.nunique()), "train_runs": int(tr_runs.nunique()),
            "test_rows_whose_run_is_in_train_pct": float(100 * te_runs.isin(set(tr_runs)).mean()),
            "median_train_rows_from_same_run_per_test_row": float(te_runs.map(sib).median()),
            "test_load_levels_present_in_train_pct": 100.0,
        }
        for mv, feats in (("lite", C.LITE_FEATURES), ("full", feats_full)):
            l1_te_pred = {}
            for t in C.L1_TARGETS:
                m = load_pinned(variant, "layer1", t, mv)
                l1_te_pred[t] = m.predict(te[feats])
                ok = te[t].notna().values
                got = score(te[t].values[ok], l1_te_pred[t][ok])
                exp = manifest[f"{variant}__layer1__{t}__{mv}"]
                rows.append(dict(upf=upf, variant=variant, feature_set=mv, layer="layer1", target=t,
                                 model=exp["best_params"]["model_type"], n_test=int(ok.sum()),
                                 **{f"{k}_reproduced": v for k, v in got.items()},
                                 **{f"{k}_manifest": exp[k] for k in ("r2", "mae", "rmse")}))
            X = te[feats].copy()
            for t in C.L1_TARGETS:
                X[f"l1_pred__{t}"] = l1_te_pred[t]
            m2 = load_pinned(variant, "layer2", C.T_POWER, mv)
            p = m2.predict(X.fillna(0))
            got = score(te[C.T_POWER].values, p)
            exp = manifest[f"{variant}__layer2__{C.T_POWER}__{mv}"]
            rows.append(dict(upf=upf, variant=variant, feature_set=mv, layer="layer2", target=C.T_POWER,
                             model=exp["best_params"]["model_type"], n_test=int(len(te)),
                             **{f"{k}_reproduced": v for k, v in got.items()},
                             **{f"{k}_manifest": exp[k] for k in ("r2", "mae", "rmse")}))
    rep = pd.DataFrame(rows)
    for k in ("r2", "mae", "rmse"):
        rep[f"{k}_abs_diff"] = (rep[f"{k}_reproduced"] - rep[f"{k}_manifest"]).abs()
    rep.to_csv(C.OUT / "original_split_reproduction.csv", index=False)
    prov["original_split"] = {
        "definition": "sklearn train_test_split(test_size=0.2, random_state=42) per variant, row level (src/train.py)",
        "validation_split": "none (model family chosen by best TEST R2; hyper-parameters by ungrouped 5-fold KFold on train rows)",
        "leakage": leakage,
        "max_abs_diff_vs_manifest": {k: float(rep[f"{k}_abs_diff"].max()) for k in ("r2", "mae", "rmse")},
    }

    (C.OUT / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    print(json.dumps(prov["original_split"], indent=2))
    print(rep[["upf", "feature_set", "layer", "target", "model", "r2_reproduced", "r2_manifest", "mae_reproduced", "rmse_reproduced", "r2_abs_diff"]].to_string())


if __name__ == "__main__":
    main()
