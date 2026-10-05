"""Step 2 — genuinely held-out surrogate predictions (nested, run-grouped CV).

Why refit at all: the pinned twin models were trained on a row-level random
80 % of EVERY one of the 220 profiling runs (step 1 shows 100 % of the original
test rows share a run with training rows). No profiling data is held out from
them, so their held-out error can only be estimated by re-running the same
modelling procedure with whole runs withheld.

Replicated from UpfProfilingCampaign src/train.py (thesis-v1): lite inputs
(offered DL and UL kbit/s), variants dpdk / usr_full (the two the twin loads),
candidate families + search spaces + R2 scoring + random_state, StandardScaler
pipelines, two-layer stacking with out-of-fold Layer-1 predictions as Layer-2
inputs, Layer-2 power clipped at 0 as in the twin.

Changed, to remove leakage:
  * outer folds withhold whole runs (never rows);
  * family AND hyper-parameters are selected by CV inside the outer-training
    runs only (the original chose the family by TEST R2 and tuned with
    ungrouped KFold);
  * Layer-2 stacking features are out-of-fold over inner GROUPED folds, using
    the configuration selected on the outer-training runs;
  * rows with a missing Layer-1 target still receive an OOF prediction (the
    original wrote 0 for the 21 such rows).

Protocols (outer / inner grouping)
  LOSO  leave-one-sweep-out: 5 outer folds = the 5 chronological sweeps; each
        held-out sweep contains one run at each of the 22 load levels. Inner
        folds = the 4 remaining sweeps.
  LOLO  leave-load-level-out: outer fold = rank(level) mod 5, so all 5
        replicate runs of a held-out level are withheld and neighbouring levels
        are never withheld together. Inner fold = rank among training levels
        mod 4. Held-out levels 0 and 5 Gbps are extrapolation.

Configurations
  nested     selection inside training runs (primary)
  deployed   family + hyper-parameters fixed to the pinned manifest (what the
             twin runs); refit on training runs only; selection was done
             upstream with all runs, so this is a sensitivity bracket
  lookup     non-ML reference: piecewise-linear interpolation (in load) of the
             per-level means of the training runs

Test inputs are twin-style (UL := DL, as upf_profile.py::_lite_X). Predictions
from profiling-style inputs (measured UL) are stored alongside (suffix _prof).

Outputs: heldout_predictions.csv.gz, heldout_model_selection.json
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, PredefinedSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import common as C

warnings.filterwarnings("ignore")
N_JOBS = -1
MEASURED = [C.T_THR, C.T_CPU, C.T_LOSS, C.T_DELAY, C.T_POWER]
SHORT = {C.T_THR: "thr", C.T_CPU: "cpu", C.T_LOSS: "loss", C.T_DELAY: "delay", C.T_POWER: "power"}


def make_pipe(family: str, params: dict | None = None) -> Pipeline:
    est = {"ridge": Ridge(),
           "random_forest": RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=1),
           "gradient_boosting": GradientBoostingRegressor(random_state=C.RANDOM_STATE)}[family]
    pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
    if params:
        pipe.set_params(**{f"model__{k}": v for k, v in params.items()})
    return pipe


def fit_select(X, y, inner_ids, config, manifest_key, manifest):
    """Return (fitted estimator, unfitted template, selection record)."""
    if np.unique(y).size == 1:                       # DPDK loss: identically 0
        tpl = DummyRegressor(strategy="constant", constant=float(y[0]))
        return clone(tpl).fit(X, y), tpl, {"model_type": "constant", "constant": float(y[0])}
    if config == "deployed":
        bp = dict(manifest[manifest_key]["best_params"])
        family = bp.pop("model_type")
        tpl = make_pipe(family, bp)
        return clone(tpl).fit(X, y), tpl, {"model_type": family, "params": bp, "selected_by": "pinned manifest"}
    cv = PredefinedSplit(inner_ids)
    scores, fitted = {}, {}
    for family, space in C.SPACES.items():
        if family == "ridge":
            s = GridSearchCV(make_pipe(family), space, cv=cv, scoring="r2", refit=True, n_jobs=N_JOBS)
        else:
            s = RandomizedSearchCV(make_pipe(family), space, n_iter=C.SEARCH_N_ITER, cv=cv, scoring="r2",
                                   refit=True, random_state=C.RANDOM_STATE, n_jobs=N_JOBS)
        s.fit(X, y)
        scores[family] = float(s.best_score_) if np.isfinite(s.best_score_) else -np.inf
        fitted[family] = (s.best_estimator_, {k.replace("model__", ""): v for k, v in s.best_params_.items()})
    winner = max(scores, key=scores.get)
    est, params = fitted[winner]
    return est, make_pipe(winner, params), {"model_type": winner, "params": params,
                                            "inner_cv_r2": scores, "selected_by": "inner grouped CV R2"}


def inner_oof(tpl, X, y, notna, inner_ids):
    oof = np.zeros(len(X))
    for j in np.unique(inner_ids):
        fit_rows = (inner_ids != j) & notna
        oof[inner_ids == j] = clone(tpl).fit(X[fit_rows], y[fit_rows]).predict(X[inner_ids == j])
    return oof


def lookup_predict(tr: pd.DataFrame, te_load: np.ndarray, target: str) -> np.ndarray:
    lv = tr.groupby("level_gbps")[target].mean().dropna().sort_index()
    return np.interp(te_load, lv.index.to_numpy(), lv.to_numpy())


def fold_ids(sub: pd.DataFrame, protocol: str):
    """Outer fold per row, and a function giving inner fold ids for a training frame."""
    if protocol == "LOSO":
        outer = sub["sweep"].to_numpy()

        def inner(tr):
            return pd.Series(tr["sweep"]).rank(method="dense").astype(int).to_numpy() - 1
    else:
        levels = np.sort(sub["level_gbps"].unique())
        rank = {lv: i for i, lv in enumerate(levels)}
        outer = sub["level_gbps"].map(rank).to_numpy() % 5

        def inner(tr):
            tl = np.sort(tr["level_gbps"].unique())
            r = {lv: i for i, lv in enumerate(tl)}
            return tr["level_gbps"].map(r).to_numpy() % 4
    return outer, inner


def run(protocol: str, config: str, df: pd.DataFrame, manifest: dict, selections: list) -> pd.DataFrame:
    out = []
    for upf, (variant, _) in C.UPFS.items():
        sub = df[df["upf"] == upf].reset_index(drop=True)
        outer, inner_fn = fold_ids(sub, protocol)
        for k in np.unique(outer):
            t0 = time.time()
            tr, te = sub[outer != k], sub[outer == k]
            assert not set(tr["run_dir"]) & set(te["run_dir"]), "held-out run leaked into training"
            if protocol == "LOLO":
                assert not set(tr["level_gbps"]) & set(te["level_gbps"]), "held-out level leaked into training"
            inner_ids = inner_fn(tr)
            X_tr = tr[C.LITE_FEATURES].to_numpy()
            X_te_twin = np.column_stack([te[C.DL_TX].to_numpy()] * 2)
            X_te_prof = te[C.LITE_FEATURES].to_numpy()
            rec = te[["row_id", "run_dir", "sweep", "level_gbps", "load_gbps", *MEASURED]].copy()
            rec.insert(0, "outer_fold", int(k))
            rec.insert(0, "upf", upf)
            rec.insert(0, "config", config)
            rec.insert(0, "protocol", protocol)
            rec["level_in_training"] = te["level_gbps"].isin(set(tr["level_gbps"])).to_numpy()
            rec["load_within_training_range"] = ((te["load_gbps"] >= tr["load_gbps"].min())
                                                 & (te["load_gbps"] <= tr["load_gbps"].max())).to_numpy()

            if config == "lookup":
                for t in MEASURED:
                    p = lookup_predict(tr, te["load_gbps"].to_numpy(), t)
                    rec[f"pred_{SHORT[t]}"] = p
                    rec[f"pred_{SHORT[t]}_prof"] = p
                out.append(rec)
                continue

            l1_twin, l1_prof, oof = [], [], []
            for t in C.L1_TARGETS:
                y = tr[t].to_numpy()
                notna = ~np.isnan(y)
                key = f"{variant}__layer1__{t}__lite"
                est, tpl, sel = fit_select(X_tr[notna], y[notna], inner_ids[notna], config, key, manifest)
                oof.append(inner_oof(tpl, X_tr, np.nan_to_num(y), notna, inner_ids))
                l1_twin.append(est.predict(X_te_twin))
                l1_prof.append(est.predict(X_te_prof))
                selections.append(dict(protocol=protocol, config=config, upf=upf, outer_fold=int(k), layer="layer1",
                                       target=t, n_train_rows=int(notna.sum()), n_train_runs=int(tr["run_dir"].nunique()),
                                       **sel))
                rec[f"pred_{SHORT[t]}"] = l1_twin[-1]
                rec[f"pred_{SHORT[t]}_prof"] = l1_prof[-1]

            X2_tr = np.column_stack([X_tr, *oof])
            y2 = tr[C.T_POWER].to_numpy()
            key = f"{variant}__layer2__{C.T_POWER}__lite"
            est, _, sel = fit_select(X2_tr, y2, inner_ids, config, key, manifest)
            rec["pred_power"] = np.maximum(0.0, est.predict(np.column_stack([X_te_twin, *l1_twin])))
            rec["pred_power_prof"] = np.maximum(0.0, est.predict(np.column_stack([X_te_prof, *l1_prof])))
            selections.append(dict(protocol=protocol, config=config, upf=upf, outer_fold=int(k), layer="layer2",
                                   target=C.T_POWER, n_train_rows=int(len(tr)), n_train_runs=int(tr["run_dir"].nunique()),
                                   **sel))
            out.append(rec)
            print(f"  {protocol:4s} {config:8s} {upf:4s} fold {k}: train {tr.run_dir.nunique()} runs / "
                  f"test {te.run_dir.nunique()} runs, {time.time() - t0:5.1f}s", flush=True)
    return pd.concat(out, ignore_index=True)


def main():
    df, _ = C.load_runs(verify=True)
    df["row_id"] = np.arange(len(df))                 # row index in features.csv
    manifest = json.loads((C.PINNED_MODELS / "manifest.json").read_text())
    selections, frames = [], []
    for protocol in ("LOSO", "LOLO"):
        for config in ("nested", "deployed", "lookup"):
            frames.append(run(protocol, config, df, manifest, selections))
    preds = pd.concat(frames, ignore_index=True)
    # sanity: every sample predicted exactly once per (protocol, config)
    counts = preds.groupby(["protocol", "config"])["row_id"].agg(["size", "nunique"])
    assert (counts["size"] == len(df)).all() and (counts["nunique"] == len(df)).all(), counts
    # sanity: no held-out run ever appears in its fold's training set (by construction; re-checked)
    preds.to_csv(C.OUT / "heldout_predictions.csv.gz", index=False, compression="gzip")
    (C.OUT / "heldout_model_selection.json").write_text(json.dumps(selections, indent=1, default=str))
    print(counts)


if __name__ == "__main__":
    main()
