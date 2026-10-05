"""Step 2b — are the twin-derived operating points stable under refitting?

The manuscript's controller thresholds come from the pinned surrogates via
upf_digital_twin.twin.threshold_derivation.derive_thresholds (v0.4.0):
  energy break-even  = first grid load where USR power >= DPDK power
  QoS limit          = last grid load before the first USR is_safe == False
  decision threshold = min(break-even, QoS limit) - 10 Mbps
on a 1-Mbps grid from 1 to 500 Mbps, with twin-style inputs (UL := DL).

Here the same rule is applied to every outer-fold model of step 2 (refit on
that fold's training runs with the configuration recorded in
heldout_model_selection.json — deterministic, so identical to the step-2
models), and to the pinned models for reference. Folds of DPDK and USR are
paired by index for the break-even.

Outputs: threshold_stability.csv, threshold_grid_predictions.csv.gz
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor

import common as C
from s02_heldout_cv import fold_ids, inner_oof, make_pipe

warnings.filterwarnings("ignore")
GRID_GBPS = np.linspace(0.001, 0.5, 500)          # derive_thresholds defaults
MARGIN_GBPS = 0.010


def template(sel: dict):
    if sel["model_type"] == "constant":
        return DummyRegressor(strategy="constant", constant=sel["constant"])
    return make_pipe(sel["model_type"], sel["params"])


def derive(dpdk_power, usr_power, usr_loss, usr_delay, delay_budget, loss_budget):
    grid = GRID_GBPS
    above = usr_power >= dpdk_power
    breakeven = float(grid[np.argmax(above)]) if above.any() else float(grid[-1])
    safe = (usr_loss <= loss_budget) & (usr_delay <= delay_budget)
    if safe.all():
        qos = float(grid[-1])
    elif not safe.any():
        qos = 0.0
    else:
        i = int(np.argmax(~safe))
        qos = float(grid[i - 1]) if i > 0 else 0.0
    # number of safe->unsafe / unsafe->safe flips on the grid (a monotone
    # physical cliff has exactly one)
    flips = int(np.sum(safe[1:] != safe[:-1]))
    return dict(energy_breakeven_mbps=1000 * breakeven, qos_limit_mbps=1000 * qos,
                decision_mbps=1000 * max(0.0, min(breakeven, qos) - MARGIN_GBPS),
                usr_safe_flips_on_grid=flips,
                usr_unsafe_share_100_400mbps=float(np.mean(~safe[(grid > 0.1) & (grid <= 0.4)])))


def grid_predict(variant_models, X):
    l1 = [m.predict(X) for m in variant_models["l1"]]
    power = np.maximum(0.0, variant_models["l2"].predict(np.column_stack([X, *l1])))
    return dict(thr=l1[0], cpu=l1[1], loss=l1[2], delay=l1[3], power=power)


def refit_fold(sub, outer, inner_fn, k, sels):
    tr = sub[outer != k]
    inner_ids = inner_fn(tr)
    X = tr[C.LITE_FEATURES].to_numpy()
    l1_models, oof = [], []
    for t in C.L1_TARGETS:
        y = tr[t].to_numpy()
        notna = ~np.isnan(y)
        tpl = template(sels[("layer1", t)])
        l1_models.append(clone(tpl).fit(X[notna], y[notna]))
        oof.append(inner_oof(tpl, X, np.nan_to_num(y), notna, inner_ids))
    l2 = clone(template(sels[("layer2", C.T_POWER)])).fit(np.column_stack([X, *oof]), tr[C.T_POWER].to_numpy())
    return {"l1": l1_models, "l2": l2}


def main():
    delay_budget, loss_budget = C.qos_budget()
    df, _ = C.load_runs(verify=False)
    selections = json.loads((C.OUT / "heldout_model_selection.json").read_text())
    X = np.column_stack([GRID_GBPS * 1e6] * 2)
    rows, grids = [], []

    pinned = {}
    for upf, (variant, _) in C.UPFS.items():
        pinned[upf] = grid_predict({
            "l1": [joblib.load(C.PINNED_MODELS / "layer1" / f"{variant}__{t}__lite.pkl") for t in C.L1_TARGETS],
            "l2": joblib.load(C.PINNED_MODELS / "layer2" / f"{variant}__{C.T_POWER}__lite.pkl")}, X)
        grids.append(pd.DataFrame({"source": "pinned", "protocol": "-", "config": "deployed", "outer_fold": -1,
                                   "upf": upf, "load_gbps": GRID_GBPS, **pinned[upf]}))
    rows.append(dict(source="pinned (manuscript twin)", protocol="-", config="deployed", outer_fold=-1,
                     **derive(pinned["DPDK"]["power"], pinned["USR"]["power"], pinned["USR"]["loss"],
                              pinned["USR"]["delay"], delay_budget, loss_budget)))

    for protocol in ("LOSO", "LOLO"):
        for config in ("nested", "deployed"):
            fold_preds = {}
            for upf in C.UPFS:
                sub = df[df["upf"] == upf].reset_index(drop=True)
                outer, inner_fn = fold_ids(sub, protocol)
                for k in np.unique(outer):
                    sels = {(s["layer"], s["target"]): s for s in selections
                            if s["protocol"] == protocol and s["config"] == config
                            and s["upf"] == upf and s["outer_fold"] == int(k)}
                    fold_preds[(upf, int(k))] = grid_predict(refit_fold(sub, outer, inner_fn, k, sels), X)
                    grids.append(pd.DataFrame({"source": "refit", "protocol": protocol, "config": config,
                                               "outer_fold": int(k), "upf": upf, "load_gbps": GRID_GBPS,
                                               **fold_preds[(upf, int(k))]}))
            for k in range(5):
                d, u = fold_preds[("DPDK", k)], fold_preds[("USR", k)]
                rows.append(dict(source="refit on training runs of outer fold", protocol=protocol, config=config,
                                 outer_fold=k, **derive(d["power"], u["power"], u["loss"], u["delay"],
                                                        delay_budget, loss_budget)))
            print(protocol, config, "done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(C.OUT / "threshold_stability.csv", index=False)
    pd.concat(grids, ignore_index=True).to_csv(C.OUT / "threshold_grid_predictions.csv.gz", index=False, compression="gzip")
    print(res.to_string())


if __name__ == "__main__":
    main()
