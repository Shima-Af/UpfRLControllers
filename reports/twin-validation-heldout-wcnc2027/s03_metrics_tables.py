"""Step 3 — metrics, safety classification and publication tables.

Metric definitions (e = predicted - measured, over held-out samples)
  R2       1 - sum(e^2) / sum((y - mean(y))^2); undefined if var(y) = 0
  MAE      mean |e|                RMSE   sqrt(mean e^2)
  MedAE    median |e|              P95AE  95th percentile of |e|
  bias     mean e
  NMAE     MAE  / mean(y)          NRMSE  RMSE / mean(y)        (mean-normalised;
           CV(RMSE) convention; undefined when mean(y) = 0)
  NMAE_rng MAE  / (max y - min y)  NRMSE_rng RMSE / (max y - min y)
  MAE/budget  MAE divided by the QoS budget (delay 200 us, loss 5 pkts/3 s)
95 % CIs: percentile cluster bootstrap over profiling runs (2000 resamples).

QoS safety (scenario_rl.yaml upf.qos_budget; same rule as upf_profile.py)
  measured unsafe  = loss > 5 pkts per 3-s LoadCore interval OR delay > 200 us
  predicted unsafe = predicted loss > 5 OR predicted delay > 200
  positive class = unsafe; samples with undefined measured delay are excluded
  false-safe rate   = FN / (TP + FN)  (measured unsafe, predicted safe)
  false-unsafe rate = FP / (FP + TN)  (measured safe,   predicted unsafe)

Outputs: twin_validation_table.{csv,tex,md}, safety_by_load_level.csv,
deployed_twin_in_sample_by_level.csv, rl_test_load_coverage.csv,
validation_numbers.json
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import common as C

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(C.RANDOM_STATE)
N_BOOT = 2000
DELAY_BUDGET, LOSS_BUDGET = C.qos_budget()
BAND = C.near_boundary_band()
OUTPUTS = {"power": ("power_watts", "W"),
           "delay": ("downlink_one_way_delay_distribution__weighted_mean_delay_us", "us"),
           "loss": ("gtpu_packets_dn__packets_lost_delta", "pkts/3s")}
BUDGET = {"delay": DELAY_BUDGET, "loss": LOSS_BUDGET}
SUBSETS = {
    "all": lambda d: np.ones(len(d), bool),
    # load subsets use the nominal profiled level of the run, so whole runs enter or leave together
    "near_boundary": lambda d: ((d["level_gbps"] >= BAND[0]) & (d["level_gbps"] <= BAND[1])).to_numpy(),
    "load_le_0.5gbps": lambda d: (d["level_gbps"] <= 0.5).to_numpy(),
    "load_gt_0.5gbps": lambda d: (d["level_gbps"] > 0.5).to_numpy(),
    # sensitivity only: the first retained 3-s sample of every run pairs the new
    # offered load with a Scaphandre reading from the traffic ramp-up
    "excl_first_sample_of_run": lambda d: (d.groupby("run_dir")["row_id"].rank(method="first") > 1).to_numpy(),
}


def boot_weights(n_runs: int) -> np.ndarray:
    idx = RNG.integers(0, n_runs, size=(N_BOOT, n_runs))
    return np.apply_along_axis(np.bincount, 1, idx, minlength=n_runs).astype(float)


def regression(y, p, runs, out):
    e = p - y
    ae = np.abs(e)
    res = dict(n_samples=int(len(y)), n_runs=int(pd.Series(runs).nunique()),
               mae=ae.mean(), rmse=np.sqrt(np.mean(e ** 2)), medae=np.median(ae),
               p95ae=np.quantile(ae, 0.95), bias=e.mean(),
               meas_mean=y.mean(), meas_min=y.min(), meas_max=y.max(), meas_std=y.std(),
               frac_meas_zero=float(np.mean(y == 0)))
    var = np.sum((y - y.mean()) ** 2)
    res["r2"] = 1 - np.sum(e ** 2) / var if var > 0 else np.nan
    res["nmae_mean"] = res["mae"] / res["meas_mean"] if res["meas_mean"] > 0 else np.nan
    res["nrmse_mean"] = res["rmse"] / res["meas_mean"] if res["meas_mean"] > 0 else np.nan
    rng_ = res["meas_max"] - res["meas_min"]
    res["nmae_range"] = res["mae"] / rng_ if rng_ > 0 else np.nan
    res["nrmse_range"] = res["rmse"] / rng_ if rng_ > 0 else np.nan
    res["mae_over_budget"] = res["mae"] / BUDGET[out] if out in BUDGET else np.nan
    # cluster bootstrap over runs
    g = pd.DataFrame({"r": runs, "n": 1, "ae": ae, "e2": e ** 2, "y": y, "y2": y ** 2}).groupby("r").sum()
    W = boot_weights(len(g))
    n = W @ g["n"].to_numpy()
    mae_b = (W @ g["ae"].to_numpy()) / n
    rmse_b = np.sqrt((W @ g["e2"].to_numpy()) / n)
    sst = W @ g["y2"].to_numpy() - (W @ g["y"].to_numpy()) ** 2 / n
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_b = np.where(sst > 0, 1 - (W @ g["e2"].to_numpy()) / sst, np.nan)
    for k, b in (("mae", mae_b), ("rmse", rmse_b), ("r2", r2_b)):
        ok = b[np.isfinite(b)]
        res[f"{k}_ci_lo"], res[f"{k}_ci_hi"] = (np.quantile(ok, [0.025, 0.975]) if len(ok) else (np.nan, np.nan))
    return res


def classification(meas_loss, meas_delay, pred_loss, pred_delay, runs, criterion):
    if criterion == "combined":
        mu = (meas_loss > LOSS_BUDGET) | (meas_delay > DELAY_BUDGET)
        pu = (pred_loss > LOSS_BUDGET) | (pred_delay > DELAY_BUDGET)
    elif criterion == "loss_only":
        mu, pu = meas_loss > LOSS_BUDGET, pred_loss > LOSS_BUDGET
    else:
        mu, pu = meas_delay > DELAY_BUDGET, pred_delay > DELAY_BUDGET
    tp, fn = int(np.sum(mu & pu)), int(np.sum(mu & ~pu))
    fp, tn = int(np.sum(~mu & pu)), int(np.sum(~mu & ~pu))
    res = dict(n_samples=len(mu), n_runs=int(pd.Series(runs).nunique()), tp=tp, fn=fn, fp=fp, tn=tn,
               measured_unsafe_prevalence=(tp + fn) / len(mu), predicted_unsafe_share=(tp + fp) / len(mu),
               false_safe_rate=fn / (tp + fn) if tp + fn else np.nan,
               false_unsafe_rate=fp / (fp + tn) if fp + tn else np.nan,
               precision_unsafe=tp / (tp + fp) if tp + fp else np.nan,
               recall_unsafe=tp / (tp + fn) if tp + fn else np.nan,
               accuracy=(tp + tn) / len(mu))
    g = pd.DataFrame({"r": runs, "tp": mu & pu, "fn": mu & ~pu, "fp": ~mu & pu, "tn": ~mu & ~pu}).groupby("r").sum()
    W = boot_weights(len(g))
    T = {k: W @ g[k].to_numpy(float) for k in ("tp", "fn", "fp", "tn")}
    with np.errstate(divide="ignore", invalid="ignore"):
        for name, b in (("false_safe_rate", T["fn"] / (T["tp"] + T["fn"])),
                        ("false_unsafe_rate", T["fp"] / (T["fp"] + T["tn"]))):
            ok = b[np.isfinite(b)]
            res[f"{name}_ci_lo"], res[f"{name}_ci_hi"] = (np.quantile(ok, [0.025, 0.975]) if len(ok) else (np.nan, np.nan))
    return res


def evaluate_frame(d, meta, pred_suffix=""):
    rows = []
    for upf in ("DPDK", "USR"):
        du = d[d["upf"] == upf]
        for sname, sfun in SUBSETS.items():
            ds = du[sfun(du)]
            if ds.empty:
                continue
            load = (ds["load_gbps"].min(), ds["load_gbps"].max())
            for out, (col, unit) in OUTPUTS.items():
                ok = ds[col].notna().to_numpy()
                y, p = ds[col].to_numpy()[ok], ds[f"pred_{out}{pred_suffix}"].to_numpy()[ok]
                rows.append(dict(section="regression", **meta, upf=upf, output=out, unit=unit, subset=sname,
                                 load_min_gbps=load[0], load_max_gbps=load[1],
                                 **regression(y, p, ds["run_dir"].to_numpy()[ok], out)))
            ok = ds[OUTPUTS["delay"][0]].notna().to_numpy()
            for crit in ("combined", "loss_only", "delay_only"):
                rows.append(dict(section="safety", **meta, upf=upf, output=f"is_unsafe[{crit}]", unit="-", subset=sname,
                                 load_min_gbps=load[0], load_max_gbps=load[1],
                                 **classification(ds[OUTPUTS["loss"][0]].to_numpy()[ok], ds[OUTPUTS["delay"][0]].to_numpy()[ok],
                                                  ds[f"pred_loss{pred_suffix}"].to_numpy()[ok], ds[f"pred_delay{pred_suffix}"].to_numpy()[ok],
                                                  ds["run_dir"].to_numpy()[ok], crit)))
    return rows


def pinned_predict(frame, variant, twin_inputs):
    X = np.column_stack([frame[C.DL_TX].to_numpy()] * 2) if twin_inputs else frame[C.LITE_FEATURES].to_numpy()
    l1 = [joblib.load(C.PINNED_MODELS / "layer1" / f"{variant}__{t}__lite.pkl").predict(X) for t in C.L1_TARGETS]
    power = joblib.load(C.PINNED_MODELS / "layer2" / f"{variant}__{C.T_POWER}__lite.pkl").predict(np.column_stack([X, *l1]))
    if twin_inputs:
        power = np.maximum(0.0, power)
    return dict(pred_loss=l1[2], pred_delay=l1[3], pred_power=power)


# ── formatting helpers ──────────────────────────────────────────────────────
def sig(x, n=2):
    if x is None or not np.isfinite(x):
        return "n/a"
    if x == 0:
        return "0"
    mag = int(np.floor(np.log10(abs(x))))
    if abs(x) >= 1000:                               # large values: integer with thousands separators
        return f"{x:,.0f}"
    return f"{x:.{max(0, n - 1 - mag)}f}"


def fmt_r2(x):
    return "n/a" if not np.isfinite(x) else f"{x:.3f}"


def fmt_pct(x):
    return "n/a" if not np.isfinite(x) else f"{100 * x:.1f}"


def fmt_load(a, b):
    return f"{sig(a, 1) if a >= 1e-3 else '≈0'}–{sig(b, 2)}"


def main():
    preds = pd.read_csv(C.OUT / "heldout_predictions.csv.gz")
    df, runs = C.load_runs(verify=False)
    rows = []

    # held-out: every protocol x config, twin-style inputs (primary) and profiling-style inputs
    for (protocol, config), d in preds.groupby(["protocol", "config"]):
        rows += evaluate_frame(d, dict(evidence="held-out", protocol=protocol, config=config, inputs="twin (UL:=DL)"))
        rows += evaluate_frame(d, dict(evidence="held-out", protocol=protocol, config=config, inputs="profiling (measured UL)"),
                               pred_suffix="_prof")

    # NOT held-out, for the manuscript discrepancy: pinned models on the original
    # row-level test split, and in-sample on all rows
    base = df.copy()
    base["row_id"] = np.arange(len(base))
    for upf, (variant, is_dpdk) in C.UPFS.items():
        sub = base[base["is_dpdk"] == is_dpdk]
        _, te = train_test_split(sub, test_size=0.2, random_state=C.RANDOM_STATE)
        for twin_inputs, label in ((False, "profiling (measured UL)"), (True, "twin (UL:=DL)")):
            for frame, ev, proto in ((te, "NOT held-out: original row-level test split", "train.py random 80/20 rows"),
                                     (sub, "NOT held-out: in-sample (all rows)", "resubstitution")):
                f = frame.copy()
                for k, v in pinned_predict(f, variant, twin_inputs).items():
                    f[k] = v
                rows += [r for r in evaluate_frame(f, dict(evidence=ev, protocol=proto, config="pinned pickles", inputs=label))
                         if r["upf"] == upf]

    table = pd.DataFrame(rows)
    lead = ["section", "evidence", "protocol", "config", "inputs", "upf", "output", "unit", "subset",
            "n_runs", "n_samples", "load_min_gbps", "load_max_gbps"]
    table = table[lead + [c for c in table.columns if c not in lead]]
    table.to_csv(C.OUT / "twin_validation_table.csv", index=False, float_format="%.6g")

    # ── safety by load level (held-out LOSO/LOLO nested) + deployed in-sample ──
    lvl_rows = []
    for (protocol, config), d in preds[preds["config"].isin(["nested", "deployed"])].groupby(["protocol", "config"]):
        for (upf, lvl), g in d.groupby(["upf", "level_gbps"]):
            ok = g[OUTPUTS["delay"][0]].notna()
            g = g[ok]
            mu = (g[OUTPUTS["loss"][0]] > LOSS_BUDGET) | (g[OUTPUTS["delay"][0]] > DELAY_BUDGET)
            pu = (g["pred_loss"] > LOSS_BUDGET) | (g["pred_delay"] > DELAY_BUDGET)
            lvl_rows.append(dict(evidence="held-out", protocol=protocol, config=config, upf=upf, level_gbps=lvl,
                                 n_samples=len(g), measured_unsafe_share=mu.mean(), predicted_unsafe_share=pu.mean(),
                                 false_safe=int((mu & ~pu).sum()), false_unsafe=int((~mu & pu).sum()),
                                 measured_loss_median=g[OUTPUTS["loss"][0]].median(), measured_loss_p95=g[OUTPUTS["loss"][0]].quantile(.95),
                                 predicted_loss_median=g["pred_loss"].median(),
                                 measured_delay_median=g[OUTPUTS["delay"][0]].median(), predicted_delay_median=g["pred_delay"].median(),
                                 measured_power_median=g[OUTPUTS["power"][0]].median(), predicted_power_median=g["pred_power"].median()))
    pd.DataFrame(lvl_rows).to_csv(C.OUT / "safety_by_load_level.csv", index=False, float_format="%.6g")

    ins = []
    for upf, (variant, _) in C.UPFS.items():
        g0 = df[df["upf"] == upf]
        p = pinned_predict(g0, variant, twin_inputs=True)
        g0 = g0.assign(**p)
        for lvl, g in g0.groupby("level_gbps"):
            ok = g[OUTPUTS["delay"][0]].notna()
            mu = (g.loc[ok, OUTPUTS["loss"][0]] > LOSS_BUDGET) | (g.loc[ok, OUTPUTS["delay"][0]] > DELAY_BUDGET)
            X = np.array([[lvl * 1e6, lvl * 1e6]])
            l1 = [joblib.load(C.PINNED_MODELS / "layer1" / f"{variant}__{t}__lite.pkl").predict(X)[0] for t in C.L1_TARGETS]
            pw = max(0.0, joblib.load(C.PINNED_MODELS / "layer2" / f"{variant}__{C.T_POWER}__lite.pkl").predict(np.array([[*X[0], *l1]]))[0])
            runm = g.groupby("run_dir").agg(l=(OUTPUTS["loss"][0], "mean"), dl=(OUTPUTS["delay"][0], "mean"))
            ins.append(dict(evidence="NOT held-out: pinned twin models, in-sample", upf=upf, level_gbps=lvl, n_samples=len(g),
                            measured_unsafe_share=mu.mean(),
                            measured_loss_mean=g[OUTPUTS["loss"][0]].mean(), measured_delay_mean=g[OUTPUTS["delay"][0]].mean(),
                            measured_level_mean_within_budget=bool(g[OUTPUTS["loss"][0]].mean() <= LOSS_BUDGET
                                                                   and g[OUTPUTS["delay"][0]].mean() <= DELAY_BUDGET),
                            runs_with_mean_within_budget=int(((runm.l <= LOSS_BUDGET) & (runm.dl <= DELAY_BUDGET)).sum()),
                            measured_loss_median=g[OUTPUTS["loss"][0]].median(),
                            measured_loss_p95=g[OUTPUTS["loss"][0]].quantile(.95), twin_loss_at_level=l1[2],
                            measured_delay_median=g[OUTPUTS["delay"][0]].median(), twin_delay_at_level=l1[3],
                            measured_power_median=g[OUTPUTS["power"][0]].median(), twin_power_at_level=pw,
                            twin_is_safe_at_level=bool(l1[2] <= LOSS_BUDGET and l1[3] <= DELAY_BUDGET)))
    pd.DataFrame(ins).to_csv(C.OUT / "deployed_twin_in_sample_by_level.csv", index=False, float_format="%.6g")

    # ── RL test-slice loads vs profiled range (manuscript evaluation) ───────
    scen = C.load_scenario()
    alpha = float(scen["traffic"]["alpha"])
    loads = np.load(C.REPO_ROOT / "data" / "external" / "traffic_forecaster" / "targets_test.npy")[:, 0, :] * alpha
    lv = C.DESIGN_LEVELS_GBPS
    edges = [(-np.inf, lv[1])] + [(lv[i], lv[i + 1]) for i in range(1, len(lv) - 1)] + [(lv[-1], np.inf)]
    cov = []
    grid = np.linspace(0.0, 6.0, 6001)
    pinned_grid = np.load(C.OUT / "pinned_predictions_sklearn1.8.0.npz")["USR_grid"]   # cols thr,cpu,loss,delay,power
    usr_safe_grid = (pinned_grid[:, 2] <= LOSS_BUDGET) & (pinned_grid[:, 3] <= DELAY_BUDGET)
    usr_safe_at = usr_safe_grid[np.clip(np.round(loads.ravel() * 1000).astype(int), 0, 6000)]
    for lo, hi in edges:
        m = ((loads > lo) & (loads <= hi)).ravel()
        cov.append(dict(bin_low_gbps=lo, bin_high_gbps=hi, n_steps=int(m.sum()), share_of_steps=float(m.mean()),
                        twin_usr_is_safe_share=float(usr_safe_at[m].mean()) if m.any() else np.nan))
    cov = pd.DataFrame(cov)
    cov.to_csv(C.OUT / "rl_test_load_coverage.csv", index=False, float_format="%.6g")

    # ── publication tables (primary: LOSO, nested, twin inputs) ─────────────
    prim = table[(table.evidence == "held-out") & (table.protocol == "LOSO") & (table.config == "nested")
                 & (table.inputs == "twin (UL:=DL)")]
    lolo = table[(table.evidence == "held-out") & (table.protocol == "LOLO") & (table.config == "nested")
                 & (table.inputs == "twin (UL:=DL)")]
    label = {"power": "Power (W)", "delay": r"Delay ($\mu$s)", "loss": "Loss (pkts/3~s)"}
    label_md = {"power": "Power (W)", "delay": "Delay (µs)", "loss": "Loss (pkts/3 s)"}
    upf_name = {"DPDK": "DPDK", "USR": "OAI/USR"}

    reg = prim[(prim.section == "regression") & (prim.subset == "all")]
    saf = pd.concat([
        prim[(prim.section == "safety") & (prim.output == "is_unsafe[combined]") & prim.subset.isin(["all", "near_boundary"])].assign(proto="LOSO"),
        lolo[(lolo.section == "safety") & (lolo.output == "is_unsafe[combined]") & lolo.subset.isin(["all", "near_boundary"])].assign(proto="LOLO"),
    ])

    tex = [r"% Generated by reports/twin-validation-heldout-wcnc2027/s03_metrics_tables.py -- do not edit by hand.",
           r"\begin{table*}[t]", r"\centering", r"\footnotesize",
           r"\caption{Held-out validation of the twin surrogates (leave-one-sweep-out, nested run-grouped CV; "
           r"every sample predicted by a model that neither fitted nor selected on its run). "
           r"NMAE/NRMSE: MAE/RMSE divided by the mean measured value. Loss is per 3-s LoadCore interval.}",
           r"\label{tab:twin_validation}", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
           r"UPF & Output & Runs/samples & Load (Gbps) & $R^2$ & MAE & RMSE & NMAE (\%) & NRMSE (\%) & MedAE & P95AE \\",
           r"\midrule"]
    md = ["**Table A — regression, held-out (LOSO, nested run-grouped CV, twin-style inputs)**", "",
          "| UPF | Output | Runs/samples | Load (Gbps) | R² | MAE | RMSE | NMAE (%) | NRMSE (%) | MedAE | P95AE |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for upf in ("DPDK", "USR"):
        for out in ("power", "delay", "loss"):
            r = reg[(reg.upf == upf) & (reg.output == out)].iloc[0]
            n = 3 if out == "power" else 2
            cells = [upf_name[upf], label[out], f"{int(r.n_runs)}/{int(r.n_samples):,}", fmt_load(r.load_min_gbps, r.load_max_gbps),
                     fmt_r2(r.r2), sig(r.mae, n), sig(r.rmse, n), fmt_pct(r.nmae_mean), fmt_pct(r.nrmse_mean),
                     sig(r.medae, n), sig(r.p95ae, n)]
            tex.append(" & ".join(cells).replace(",", "{,}").replace("≈0", r"$\approx$0").replace("–", "--") + r" \\")
            md.append("| " + " | ".join([cells[0], label_md[out], *cells[2:]]) + " |")
        if upf == "DPDK":
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}", r"\vspace{2mm}", "",
            r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
            r"Protocol & UPF / load subset & Samples & TP & FN & FP & TN & False-safe (\%) & False-unsafe (\%) & Prec.$_\mathrm{unsafe}$ & Rec.$_\mathrm{unsafe}$ \\",
            r"\midrule"]
    md += ["", f"**Table B — QoS safety classification, held-out (unsafe = loss > {LOSS_BUDGET:g} pkts/3 s or delay > {DELAY_BUDGET:g} µs; positive = unsafe)**", "",
           "| Protocol | UPF / load subset | Samples | TP | FN | FP | TN | False-safe (%) | False-unsafe (%) | Precision (unsafe) | Recall (unsafe) |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    sub_name = {"all": "all loads", "near_boundary": f"{1000 * BAND[0]:.0f}–{1000 * BAND[1]:.0f} Mbps"}
    for proto in ("LOSO", "LOLO"):
        for upf in ("DPDK", "USR"):
            for s in ("all", "near_boundary"):
                q = saf[(saf.proto == proto) & (saf.upf == upf) & (saf.subset == s)]
                if q.empty:
                    continue
                r = q.iloc[0]
                prec = "n/a" if not np.isfinite(r.precision_unsafe) else f"{r.precision_unsafe:.2f}"
                rec = "n/a" if not np.isfinite(r.recall_unsafe) else f"{r.recall_unsafe:.2f}"
                cells = [proto, f"{upf_name[upf]}, {sub_name[s]}", f"{int(r.n_samples):,}", f"{int(r.tp):,}", f"{int(r.fn):,}", f"{int(r.fp):,}", f"{int(r.tn):,}",
                         fmt_pct(r.false_safe_rate), fmt_pct(r.false_unsafe_rate), prec, rec]
                tex.append(" & ".join(cells).replace(",", "{,}").replace("–", "--").replace("{,} ", ", ") + r" \\")
                md.append("| " + " | ".join(cells) + " |")
        if proto == "LOSO":
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    md += ["", "LOSO = leave-one-sweep-out (a held-out run's load level is still profiled by other sweeps). "
           "LOLO = leave-load-level-out (all runs at the held-out level withheld; tests interpolation to unprofiled loads). "
           f"Near-boundary band = within a factor of 3 of the twin's USR QoS limit (149 Mbps). "
           "NMAE/NRMSE = MAE/RMSE ÷ mean measured value (n/a when the mean is 0). "
           "Full metric set, 95 % run-bootstrap CIs, other configurations and the non-held-out reference rows are in twin_validation_table.csv."]
    (C.OUT / "twin_validation_table.tex").write_text("\n".join(tex) + "\n")
    (C.OUT / "twin_validation_table.md").write_text("\n".join(md) + "\n")

    # ── numbers cited in the report ─────────────────────────────────────────
    thr = pd.read_csv(C.OUT / "threshold_stability.csv") if (C.OUT / "threshold_stability.csv").exists() else None
    nums = {
        "band_gbps": BAND, "budgets": {"delay_us": DELAY_BUDGET, "loss_pkts_per_3s": LOSS_BUDGET},
        "rl_test": {"alpha": alpha, "steps": int(loads.size), "max_gbps": float(loads.max()),
                    "steps_above_5gbps": int((loads > 5.0).sum()),
                    "share_in_0.1_0.6gbps_gaps": float(((loads > 0.1) & (loads < 0.6)).mean())},
    }
    if thr is not None:
        nums["threshold_stability"] = thr.to_dict(orient="records")
    (C.OUT / "validation_numbers.json").write_text(json.dumps(nums, indent=2, default=float))
    print("\n".join(md))
    print(cov.to_string())


if __name__ == "__main__":
    main()
