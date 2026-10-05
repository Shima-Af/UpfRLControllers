"""Step 4 — metrics tables, rank stability, verdict, decision-boundary analysis.

Implements analysis_plan.md exactly (pre-registered before the alternative-world
runs). Everything that is not in the plan is written to files with a
``posthoc_`` prefix.

Inputs : results/main/*.csv, results/recal/*.csv, rollouts/main/**.npz,
         bundle_thresholds.csv, bundles/*/bundle_provenance.json,
         reproduction/table1_traceability.json
Outputs: evaluation_results.csv (Experiment A), fixed_action_replay.csv
         (Experiment B), per_cluster_results.csv, controller_world_summary.csv,
         rank_stability.csv, fold_summary.csv, uncertainty_decomposition.csv,
         verdict.json, recalibrated_hysteresis.csv, decision_boundary.csv,
         decision_boundary_summary.csv, boundary_measured_labels.csv,
         boundary_usage_intervals.csv, calibration_summary.tex, analysis_numbers.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import rb_common as C
from evaluate_worlds import safe

RANKED = ["MAPPO", "IPPO", "Centralized PPO", "Hysteresis (fixed)", "Always-DPDK"]
SUPP = ["Shared-PPO", "Always-USR (supplementary)"]
METRICS = ["reward", "energy_penalty", "qos_penalty", "switch_penalty", "cooldown_penalty", "energy_wh",
           "energy_wh_steady", "qos_violation_rate", "delay_violation_rate", "loss_violation_rate",
           "qscore_below_tau_rate", "usr_share", "switches", "switches_per_cluster_day", "n_budget_violations",
           "severity_mean", "severity_max", "q_shortfall_mean"]
KEY = ["reward", "energy_wh", "qos_violation_rate", "usr_share", "switches_per_cluster_day",
       "energy_penalty", "qos_penalty", "switch_penalty", "cooldown_penalty"]
SET_LABEL = {"S1": "manuscript checkpoints (MAPPO/IPPO traced to Table I; centralized = v0.4 substitute set)",
             "S2": "same-code revision set (cohort R)", "S3": "v0.4 re-scored set (cohort P)"}
BASE_IDS = {b["family"]: b["ckpt_id"] for b in C.BASELINES}
WORLD_ORDER = [w["world"] for w in C.worlds()]
TIER = {w["world"]: w["tier"] for w in C.worlds()}
DESIGN_MBPS = [60, 80, 100, 200, 400, 600]
INTERVALS = {100: (89.4, 141.4), 200: (141.4, 282.8), 400: (282.8, 489.9), 600: (489.9, 692.8)}


def controller_sets(reg) -> dict[str, dict[str, list[str]]]:
    def ids(cohort, fam):
        return sorted([r["ckpt_id"] for r in reg if cohort in r["cohorts"] and r["family"] == fam])
    base = {"Hysteresis (fixed)": [BASE_IDS["Hysteresis (fixed)"]], "Always-DPDK": [BASE_IDS["Always-DPDK"]],
            "Always-USR (supplementary)": [BASE_IDS["Always-USR (supplementary)"]]}
    return {
        "S1": {"MAPPO": ids("T", "MAPPO"), "IPPO": ids("T", "IPPO"), "Centralized PPO": ids("P", "Centralized PPO"), **base},
        "S2": {"MAPPO": ids("R", "MAPPO"), "IPPO": ids("R", "IPPO"), "Centralized PPO": ids("R", "Centralized PPO"),
               "Shared-PPO": ids("R", "Shared-PPO"), **base},
        "S3": {"MAPPO": ids("P", "MAPPO"), "IPPO": ids("P", "IPPO"), "Centralized PPO": ids("P", "Centralized PPO"), **base},
    }


def load_results(reg):
    df = pd.concat([pd.read_csv(f) for f in sorted((C.HERE / "results" / "main").glob("*__*.csv"))], ignore_index=True)
    coh = {r["ckpt_id"]: "+".join(r["cohorts"]) for r in reg}
    df["cohorts"] = df.ckpt_id.map(coh).fillna("baseline")
    df["cluster"] = df.cluster.astype(str)
    pinned = df[(df.world == "pinned") & (df.experiment == "A")].set_index(["ckpt_id", "cluster"])[METRICS]
    d = df.join(pinned, on=["ckpt_id", "cluster"], rsuffix="_pinned")
    for m in METRICS:
        d[f"diff_from_pinned_{m}"] = d[m] - d[f"{m}_pinned"]
    d = d.drop(columns=[f"{m}_pinned" for m in METRICS])
    d["world"] = pd.Categorical(d.world, WORLD_ORDER, ordered=True)
    return d.sort_values(["experiment", "world", "family", "seed", "ckpt_id"]).reset_index(drop=True)


def pareto(scores: dict[str, tuple[float, float]]):
    names = list(scores)
    dom = {(a, b): (scores[a][0] <= scores[b][0] and scores[a][1] <= scores[b][1]
                    and (scores[a][0] < scores[b][0] or scores[a][1] < scores[b][1])) for a in names for b in names if a != b}
    nondom = [b for b in names if not any(dom[(a, b)] for a in names if a != b)]
    return dom, nondom


def world_stats(allr: pd.DataFrame, members: dict[str, list[str]], world: str, exp: str) -> dict:
    sub = allr[(allr.world == world) & (allr.experiment == exp)]
    per = {c: sub[sub.ckpt_id.isin(ids)].set_index("seed") for c, ids in members.items()}
    mean = {c: per[c][["reward", "energy_wh", "qos_violation_rate"]].mean() for c in RANKED}
    med = {c: per[c].reward.median() for c in RANKED}
    out = dict(world=world, tier=TIER[world], experiment=exp,
               **{f"n_{c}": len(per[c]) for c in RANKED})
    for key, asc, lab in (("reward", False, "reward"), ("energy_wh", True, "energy"), ("qos_violation_rate", True, "qos")):
        order = sorted(RANKED, key=lambda c: mean[c][key], reverse=not asc)
        out[f"ranking_{lab}"] = " > ".join(order)
        for c in RANKED:
            out[f"rank_{lab}__{c}"] = order.index(c) + 1
    order_med = sorted(RANKED, key=lambda c: med[c], reverse=True)
    out["ranking_reward_median"] = " > ".join(order_med)
    out["rank_reward_median__MAPPO"] = order_med.index("MAPPO") + 1
    out["first_reward"] = out["ranking_reward"].split(" > ")[0]
    for c in RANKED:
        out[f"mean_reward__{c}"] = mean[c]["reward"]
        out[f"mean_energy_wh__{c}"] = mean[c]["energy_wh"]
        out[f"mean_qosv__{c}"] = mean[c]["qos_violation_rate"]
    m, i = per["MAPPO"], per["IPPO"]
    common = sorted(set(m.index) & set(i.index))
    dI = (m.loc[common, "reward"] - i.loc[common, "reward"])
    hy = float(per["Hysteresis (fixed)"].reward.iloc[0])
    dH = m.reward - hy
    out.update(delta_IPPO=float(dI.mean()), delta_IPPO_n_pairs=len(common), delta_IPPO_seeds_favour_MAPPO=int((dI > 0).sum()),
               delta_IPPO_seed_sd=float(dI.std(ddof=1)), delta_IPPO_seed_min=float(dI.min()), delta_IPPO_seed_max=float(dI.max()),
               delta_IPPO_median=float(dI.median()),
               delta_hyst=float(dH.mean()), delta_hyst_seeds_favour_MAPPO=int((dH > 0).sum()),
               delta_hyst_seed_min=float(dH.min()), delta_hyst_seed_max=float(dH.max()),
               delta_DPDK=float(m.reward.mean() - per["Always-DPDK"].reward.iloc[0]),
               delta_central=float(m.reward.mean() - per["Centralized PPO"].reward.mean()),
               energy_MAPPO_minus_IPPO=float(mean["MAPPO"]["energy_wh"] - mean["IPPO"]["energy_wh"]),
               qosv_MAPPO_minus_IPPO=float(mean["MAPPO"]["qos_violation_rate"] - mean["IPPO"]["qos_violation_rate"]),
               energy_MAPPO_minus_hyst=float(mean["MAPPO"]["energy_wh"] - mean["Hysteresis (fixed)"]["energy_wh"]),
               qosv_MAPPO_minus_hyst=float(mean["MAPPO"]["qos_violation_rate"] - mean["Hysteresis (fixed)"]["qos_violation_rate"]))
    dom, nondom = pareto({c: (mean[c]["energy_wh"], mean[c]["qos_violation_rate"]) for c in RANKED})
    out.update(pareto_nondominated=" ; ".join(nondom), P1_MAPPO_nondominated="MAPPO" in nondom,
               P2_MAPPO_dominates_IPPO=dom[("MAPPO", "IPPO")],
               P2_MAPPO_dominates_hysteresis=dom[("MAPPO", "Hysteresis (fixed)")],
               P2_MAPPO_dominates_central=dom[("MAPPO", "Centralized PPO")],
               pareto_dominance_pairs=" ; ".join(f"{a}>{b}" for (a, b), v in dom.items() if v))
    return out


def add_stability(rs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, exp), g in rs.groupby(["set", "experiment"]):
        p = rs[(rs.set == s) & (rs.world == "pinned") & (rs.experiment == "A")].iloc[0]
        pr = np.array([p[f"rank_reward__{c}"] for c in RANKED])
        for _, r in g.iterrows():
            wr = np.array([r[f"rank_reward__{c}"] for c in RANKED])
            r = r.copy()
            r["kendall_tau_b_vs_pinned"] = stats.kendalltau(pr, wr).statistic
            r["spearman_vs_pinned"] = stats.spearmanr(pr, wr).statistic
            for lab in ("energy", "qos"):
                r[f"kendall_tau_b_{lab}_vs_pinned"] = stats.kendalltau(
                    [p[f"rank_{lab}__{c}"] for c in RANKED], [r[f"rank_{lab}__{c}"] for c in RANKED]).statistic
            r["ranking_reward_changed"] = r["ranking_reward"] != p["ranking_reward"]
            for k in ("delta_IPPO", "delta_hyst", "delta_DPDK", "delta_central"):
                r[f"{k}_sign_change"] = np.sign(r[k]) != np.sign(p[k])
                r[f"{k}_ratio_to_pinned"] = r[k] / p[k] if p[k] != 0 else np.nan
            raw_same = all(np.sign(r[k]) == np.sign(p[k]) for k in ("energy_MAPPO_minus_IPPO", "qosv_MAPPO_minus_IPPO",
                                                                    "energy_MAPPO_minus_hyst", "qosv_MAPPO_minus_hyst"))
            r["raw_energy_qos_signs_unchanged"] = raw_same
            p2_ok = all((not p[k]) or r[k] for k in ("P2_MAPPO_dominates_IPPO", "P2_MAPPO_dominates_hysteresis",
                                                    "P2_MAPPO_dominates_central"))
            r["P2_pinned_relations_hold"] = p2_ok
            r["pareto_nondominated_changed"] = r["pareto_nondominated"] != p["pareto_nondominated"]
            r["m_condition"] = bool(0.5 <= r["delta_IPPO_ratio_to_pinned"] <= 2 and 0.5 <= r["delta_hyst_ratio_to_pinned"] <= 2
                                    and raw_same and p2_ok)
            rows.append(r)
    return pd.DataFrame(rows)


def verdict(g: pd.DataFrame) -> dict:
    n1 = int((g.first_reward == "MAPPO").sum())
    rI = int((g.delta_IPPO <= 0).sum())
    rH = int((g.delta_hyst <= 0).sum())
    p1 = int((~g.P1_MAPPO_nondominated.astype(bool)).sum())
    m = int(g.m_condition.astype(bool).sum())
    if n1 >= 9 and rI == 0 and rH == 0 and p1 == 0 and m >= 9:
        v = "A"
    elif n1 >= 8 and rI <= 1 and rH <= 1 and p1 <= 2:
        v = "B"
    else:
        v = "C"
    return dict(n_worlds=len(g), n1_MAPPO_first=n1, r_I_delta_IPPO_nonpositive=rI, r_H_delta_hyst_nonpositive=rH,
                p1_MAPPO_dominated=p1, m_magnitude_raw_pareto_ok=m, verdict=v,
                worlds_MAPPO_not_first=g.loc[g.first_reward != "MAPPO", "world"].astype(str).tolist(),
                worlds_delta_IPPO_nonpositive=g.loc[g.delta_IPPO <= 0, "world"].astype(str).tolist(),
                worlds_delta_hyst_nonpositive=g.loc[g.delta_hyst <= 0, "world"].astype(str).tolist(),
                worlds_MAPPO_dominated=g.loc[~g.P1_MAPPO_nondominated.astype(bool), "world"].astype(str).tolist(),
                worlds_failing_m=g.loc[~g.m_condition.astype(bool), "world"].astype(str).tolist())


def fmt_mmm(x):
    return f"{np.median(x):.0f} [{np.min(x):.0f}, {np.max(x):.0f}]"


# ---------------------------------------------------------------------------
# decision boundary
# ---------------------------------------------------------------------------

def world_usr_safe(world: dict, loads: np.ndarray) -> np.ndarray:
    from upf_digital_twin import DigitalTwin
    from src.utils.config import load_yaml
    tw = DigitalTwin(scenario_cfg=load_yaml("configs/scenario_rl.yaml"), paths_cfg=C.paths_cfg_for(world), project_root=C.ROOT)
    return tw.evaluate_batch("USR", loads.ravel())["is_safe"].reshape(loads.shape)


def boundary(sets, thr: pd.DataFrame):
    tr = np.load(C.ROLLOUTS / "main" / "traffic_test.npz")
    load_mbps, fc_mbps = tr["load_gbps"] * 1000, tr["forecast_gbps"] * 1000
    region = (load_mbps >= 50) & (load_mbps <= 600)
    region_f = (fc_mbps >= 50) & (fc_mbps <= 600)
    ids = sorted({i for s in sets.values() for v in s.values() for i in v})
    rows = []
    usr_safe = {}
    for w in C.worlds():
        usr_safe[w["world"]] = world_usr_safe(w, tr["load_gbps"])
    pinned_act = {i: np.load(C.ROLLOUTS / "main" / "pinned" / "A" / f"{safe(i)}.npz")["action"] for i in ids}
    for w in C.worlds():
        q = float(thr.loc[w["world"], "qos_limit_mbps"])
        us = usr_safe[w["world"]]
        for exp in ("A", "B"):
            for i in ids:
                f = C.ROLLOUTS / "main" / w["world"] / exp / f"{safe(i)}.npz"
                if not f.exists():
                    continue
                d = np.load(f)
                usr, safe_, act = d["usr"], d["is_safe"], d["action"]
                ur = usr & region
                r = dict(world=w["world"], tier=w["tier"], experiment=exp, ckpt_id=i,
                         n_region=int(region.sum()), pct_region=100 * float(region.mean()),
                         n_region_by_forecast=int(region_f.sum()), pct_region_by_forecast=100 * float(region_f.mean()),
                         action_change_vs_pinned_region=float((act != pinned_act[i])[region].mean()),
                         action_change_vs_pinned_all=float((act != pinned_act[i]).mean()),
                         usr_share_region=float(usr[region].mean()),
                         usr_steps_region=int(ur.sum()),
                         usr_region_predicted_safe_share=float(safe_[ur].mean()) if ur.any() else np.nan,
                         usr_region_predicted_unsafe_share=float((~safe_[ur]).mean()) if ur.any() else np.nan,
                         world_usr_unsafe_share_region_loads=float((~us[region]).mean()),
                         world_qos_limit_mbps=q, usr_share_above_world_qos_limit=float(usr[load_mbps > q].mean()) if (load_mbps > q).any() else np.nan,
                         usr_share_above_149=float(usr[load_mbps > 149].mean()))
                for lv, (lo, hi) in INTERVALS.items():
                    msk = (load_mbps >= lo) & (load_mbps < hi)
                    r[f"n_steps_{lv}Mbps_interval"] = int(msk.sum())
                    r[f"usr_share_{lv}Mbps_interval"] = float(usr[msk].mean()) if msk.any() else np.nan
                rows.append(r)
    return pd.DataFrame(rows), usr_safe


def measured_labels():
    sys.path.insert(0, str(C.TASKA))
    import common as TA  # noqa: E402
    from upf_digital_twin import DigitalTwin
    from src.utils.config import load_yaml
    df, _ = TA.load_runs(verify=False)
    delay_b, loss_b = C.qos_budget()
    usr = df[(df.upf == "USR") & df.level_gbps.isin([x / 1000 for x in DESIGN_MBPS])].copy()
    usr = usr[usr[TA.T_DELAY].notna() & usr[TA.T_LOSS].notna()]
    usr["measured_unsafe"] = (usr[TA.T_LOSS] > loss_b) | (usr[TA.T_DELAY] > delay_b)
    rows = []
    for w in C.worlds():
        tw = DigitalTwin(scenario_cfg=load_yaml("configs/scenario_rl.yaml"), paths_cfg=C.paths_cfg_for(w), project_root=C.ROOT)
        pu = ~tw.evaluate_batch("USR", usr.load_gbps.to_numpy())["is_safe"]
        if w["tier"] == "reference":
            held = np.zeros(len(usr), bool)
        else:
            prov = json.loads((C.BUNDLES / f"{w['protocol']}_{w['config']}_fold{w['fold']}" / "bundle_provenance.json").read_text())
            held = usr.run_dir.isin(prov["upf"]["USR"]["heldout_runs"]).to_numpy()
        for lv in DESIGN_MBPS:
            for subset, msk in (("all", np.ones(len(usr), bool)), ("held-out of this world's fit", held),
                                ("in this world's training data", ~held)):
                sel = (usr.level_gbps.to_numpy() == lv / 1000) & msk
                if not sel.any():
                    continue
                mu, p = usr.measured_unsafe.to_numpy()[sel], pu[sel]
                rows.append(dict(world=w["world"], tier=w["tier"], level_mbps=lv, subset=subset, n_samples=int(sel.sum()),
                                 n_runs=int(usr.run_dir[sel].nunique()), measured_unsafe_share=float(mu.mean()),
                                 predicted_unsafe_share=float(p.mean()), false_safe=int((mu & ~p).sum()),
                                 false_unsafe=int((~mu & p).sum()),
                                 false_safe_share_of_measured_unsafe=float((mu & ~p).sum() / mu.sum()) if mu.any() else np.nan,
                                 false_unsafe_share_of_measured_safe=float((~mu & p).sum() / (~mu).sum()) if (~mu).any() else np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TeX
# ---------------------------------------------------------------------------

def write_tex(rs, summ, verdicts, recal):
    def block(s, exp, label):
        g = rs[(rs.set == s) & (rs.experiment == exp)]
        pin = g[g.world == "pinned"].iloc[0]
        prim = g[g.tier == "primary"]
        lines = []
        for c in RANKED:
            n = int(pin[f"n_{c}"])
            name = {"Hysteresis (fixed)": "Hysteresis (fixed, $b{=}50$)", "Centralized PPO": "Centralized PPO" + ("$^\\dagger$" if s != "S2" else "")}.get(c, c)
            r = prim[f"mean_reward__{c}"].to_numpy()
            e = prim[f"mean_energy_wh__{c}"].to_numpy()
            q = 100 * prim[f"mean_qosv__{c}"].to_numpy()
            lines.append(f"{name} & {n if c in ('MAPPO', 'IPPO', 'Centralized PPO') else '--'} & ${pin[f'mean_reward__{c}']:.0f}$ & "
                         f"${np.median(r):.0f}$ [${r.min():.0f}$, ${r.max():.0f}$] & {int((prim.first_reward == c).sum())}/10 & "
                         f"{pin[f'mean_energy_wh__{c}']:.0f} & {np.median(e):.0f} [{e.min():.0f}, {e.max():.0f}] & "
                         f"{100 * pin[f'mean_qosv__{c}']:.2f} & {np.median(q):.2f} [{q.min():.2f}, {q.max():.2f}] \\\\")
        for k, lab in (("delta_IPPO", "MAPPO $-$ IPPO (paired)"), ("delta_hyst", "MAPPO $-$ hysteresis")):
            x = prim[k].to_numpy()
            fav = prim[f"{k}_seeds_favour_MAPPO"].to_numpy()
            lines.append(f"\\multicolumn{{2}}{{l}}{{{lab}}} & ${pin[k]:+.0f}$ & ${np.median(x):+.0f}$ [${x.min():+.0f}$, ${x.max():+.0f}$] & "
                         f"\\multicolumn{{5}}{{l}}{{sign reversals {int((x <= 0).sum())}/10; seeds favouring MAPPO per world "
                         f"{fav.min() if fav.min() == fav.max() else f'{fav.min()}--{fav.max()}'} of {int(pin['n_MAPPO'])}}} \\\\")
        v = verdicts[f"{s}|primary|{exp}"]
        return lines, v
    out = [r"% Generated by s04_analyze.py — calibration-robustness evaluation (zero-shot, no retraining).",
           r"\begin{table*}[!t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
           r"\caption{Controller comparison across ten alternative surrogate calibrations derived from the existing "
           r"profiling data (five leave-one-sweep-out and five leave-load-level-out refits of the deployed model "
           r"configuration). Closed-policy zero-shot evaluation: saved policies, unchanged parameters, same test slice, "
           r"reward and switching model. Values are seed means; fold columns give median [min, max] over the ten worlds "
           r"(not independent experiments; no confidence interval is implied). QoSv: QoS-violation rate.}",
           r"\label{tab:calibration_robustness}",
           r"\begin{tabular}{@{}lcrrcrrrr@{}}", r"\toprule",
           r"Controller & $n$ & \multicolumn{2}{c}{Reward} & Ranked & \multicolumn{2}{c}{Energy (Wh)} & \multicolumn{2}{c}{QoSv (\%)} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
           r" & & pinned & 10 folds & first & pinned & 10 folds & pinned & 10 folds \\", r"\midrule"]
    for s, title in (("S1", "Manuscript checkpoints"), ("S2", "Same-code retrained set")):
        lines, v = block(s, "A", title)
        out.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\textit{{{title}}} --- pre-registered verdict: {v['verdict']}}} \\")
        out += lines
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    out += [r"\end{tabular}",
            r"\par\smallskip{\raggedright\footnotesize $^\dagger$The manuscript's centralized-PPO checkpoints are not recoverable; "
            r"the four-seed set re-scored in the repository under the pinned twin is used instead. Hysteresis thresholds "
            r"are held at their published values in every world.\par}",
            r"\end{table*}"]
    (C.HERE / "calibration_summary.tex").write_text("\n".join(out) + "\n")


def main():
    reg = C.checkpoint_registry()
    sets = controller_sets(reg)
    d = load_results(reg)
    for s, mem in sets.items():
        ids = {i for v in mem.values() for i in v}
        d[f"in_{s}"] = d.ckpt_id.isin(ids)
    base_cols = ["experiment", "world", "tier", "protocol", "config", "fold", "family", "seed", "ckpt_id", "cohorts",
                 "in_S1", "in_S2", "in_S3", "cluster"]
    diff_cols = [f"diff_from_pinned_{m}" for m in METRICS]
    allr = d[d.cluster == "all"]
    allr[allr.experiment == "A"][base_cols + METRICS + diff_cols].to_csv(C.HERE / "evaluation_results.csv", index=False)
    allr[allr.experiment == "B"][base_cols + METRICS + diff_cols].to_csv(C.HERE / "fixed_action_replay.csv", index=False)
    d[d.cluster != "all"][base_cols + METRICS + diff_cols].to_csv(C.HERE / "per_cluster_results.csv", index=False)

    # summaries per set / controller / world / experiment
    srows = []
    for s, mem in sets.items():
        for c, ids in mem.items():
            g = allr[allr.ckpt_id.isin(ids)]
            for (w, exp), h in g.groupby(["world", "experiment"], observed=True):
                r = dict(set=s, controller=c, world=w, tier=TIER[w], experiment=exp, n_seeds=len(h))
                for m in KEY + ["delay_violation_rate", "loss_violation_rate", "severity_mean", "q_shortfall_mean"]:
                    x = h[m].to_numpy()
                    r.update({f"{m}_mean": x.mean(), f"{m}_sd": x.std(ddof=1) if len(x) > 1 else np.nan,
                              f"{m}_median": np.median(x), f"{m}_min": x.min(), f"{m}_max": x.max()})
                r["severity_max_max"] = h.severity_max.max()
                r["diff_from_pinned_reward_mean"] = h.diff_from_pinned_reward.mean()
                r["diff_from_pinned_energy_wh_mean"] = h.diff_from_pinned_energy_wh.mean()
                r["diff_from_pinned_qos_violation_rate_mean"] = h.diff_from_pinned_qos_violation_rate.mean()
                srows.append(r)
    summ = pd.DataFrame(srows)
    summ.to_csv(C.HERE / "controller_world_summary.csv", index=False)

    rs = []
    for s, mem in sets.items():
        for exp in ("A", "B"):
            for w in WORLD_ORDER:
                rs.append(dict(set=s, **world_stats(allr, mem, w, exp)))
    rs = add_stability(pd.DataFrame(rs))
    rs["world"] = pd.Categorical(rs.world, WORLD_ORDER, ordered=True)
    rs = rs.sort_values(["set", "experiment", "world"])
    lead = ["set", "experiment", "world", "tier", "ranking_reward", "ranking_energy", "ranking_qos", "first_reward",
            "delta_IPPO", "delta_IPPO_seeds_favour_MAPPO", "delta_IPPO_sign_change", "delta_hyst",
            "delta_hyst_seeds_favour_MAPPO", "delta_hyst_sign_change", "kendall_tau_b_vs_pinned", "spearman_vs_pinned",
            "P1_MAPPO_nondominated", "pareto_nondominated", "m_condition"]
    rs = rs[lead + [c for c in rs.columns if c not in lead]]
    rs.to_csv(C.HERE / "rank_stability.csv", index=False)

    verdicts = {}
    for s in sets:
        for tier in ("primary", "secondary"):
            for exp in ("A", "B"):
                g = rs[(rs.set == s) & (rs.tier == tier) & (rs.experiment == exp)]
                verdicts[f"{s}|{tier}|{exp}"] = dict(set=s, set_label=SET_LABEL[s], tier=tier, experiment=exp,
                                                     role=("PRIMARY VERDICT" if (s, tier, exp) == ("S1", "primary", "A")
                                                           else "secondary"), **verdict(g))
    (C.HERE / "verdict.json").write_text(json.dumps(verdicts, indent=1, default=str))

    # per protocol best / median / worst and complete folds
    frows = []
    for (s, exp), g in rs[rs.tier != "reference"].groupby(["set", "experiment"]):
        for (tier, proto), h in g.groupby([g.tier, g.world.astype(str).str[:4]]):
            for stat in ("delta_IPPO", "delta_hyst", "delta_DPDK", "delta_central", "kendall_tau_b_vs_pinned",
                         *[f"mean_reward__{c}" for c in RANKED], *[f"mean_energy_wh__{c}" for c in RANKED],
                         *[f"mean_qosv__{c}" for c in RANKED]):
                h2 = h.sort_values(stat)
                x = h2[stat].to_numpy()
                row = dict(set=s, experiment=exp, tier=tier, protocol=proto, statistic=stat,
                           pinned=float(rs[(rs.set == s) & (rs.experiment == "A") & (rs.world == "pinned")][stat].iloc[0]),
                           worst=x[0], median=float(np.median(x)), best=x[-1],
                           worst_world=str(h2.world.iloc[0]), best_world=str(h2.world.iloc[-1]))
                if "qosv" in stat or "energy" in stat:
                    row.update(worst=x[-1], best=x[0], worst_world=str(h2.world.iloc[-1]), best_world=str(h2.world.iloc[0]))
                for _, r in h.iterrows():
                    row[f"fold{int(str(r.world)[-1])}"] = r[stat]
                frows.append(row)
    pd.DataFrame(frows).to_csv(C.HERE / "fold_summary.csv", index=False)

    # seed vs calibration uncertainty (reported separately, never combined)
    urows = []
    for s, mem in sets.items():
        for c in [*RANKED, *[x for x in SUPP if x in mem]]:
            for exp in ("A", "B"):
                for m in ("reward", "energy_wh", "qos_violation_rate"):
                    g = summ[(summ.set == s) & (summ.controller == c) & (summ.experiment == exp)]
                    pin = summ[(summ.set == s) & (summ.controller == c) & (summ.experiment == "A") & (summ.world == "pinned")].iloc[0]
                    for tier in ("primary", "secondary"):
                        h = g[g.tier == tier]
                        urows.append(dict(set=s, controller=c, experiment=exp, metric=m, tier=tier, n_seeds=int(pin.n_seeds),
                                          seed_sd_pinned=pin[f"{m}_sd"], seed_range_pinned=pin[f"{m}_max"] - pin[f"{m}_min"],
                                          seed_sd_median_over_worlds=h[f"{m}_sd"].median(),
                                          pinned_seed_mean=pin[f"{m}_mean"],
                                          calibration_median_of_seed_means=h[f"{m}_mean"].median(),
                                          calibration_min_of_seed_means=h[f"{m}_mean"].min(),
                                          calibration_max_of_seed_means=h[f"{m}_mean"].max(),
                                          calibration_range_of_seed_means=h[f"{m}_mean"].max() - h[f"{m}_mean"].min()))
    pd.DataFrame(urows).to_csv(C.HERE / "uncertainty_decomposition.csv", index=False)

    # recalibrated hysteresis (secondary)
    rec = pd.concat([pd.read_csv(f) for f in sorted((C.HERE / "results" / "recal").glob("*__A.csv"))], ignore_index=True)
    rec = rec[rec.cluster.astype(str) == "all"]
    best = rec.loc[rec.groupby("world").reward.idxmax()].assign(selection="best band on test slice (manuscript protocol)")
    b50 = rec[rec.band_mbps == 50].assign(selection="b = 50 Mbps")
    recal = pd.concat([best, b50])
    fixed = allr[(allr.ckpt_id == BASE_IDS["Hysteresis (fixed)"]) & (allr.experiment == "A")].set_index("world")
    recal["fixed_hysteresis_reward"] = recal.world.map(fixed.reward.to_dict()).astype(float)
    for s in ("S1", "S2"):
        g = rs[(rs.set == s) & (rs.experiment == "A")].set_index("world")
        recal[f"{s}_MAPPO_mean_reward"] = recal.world.map(g["mean_reward__MAPPO"].to_dict()).astype(float)
        recal[f"{s}_MAPPO_minus_recalibrated"] = recal[f"{s}_MAPPO_mean_reward"] - recal.reward
    recal["tier"] = recal.world.map(TIER)
    recal = recal[["world", "tier", "selection", "t_up_mbps", "band_mbps", "reward", "energy_wh", "qos_violation_rate",
                   "usr_share", "switches_per_cluster_day", "fixed_hysteresis_reward", "S1_MAPPO_mean_reward",
                   "S1_MAPPO_minus_recalibrated", "S2_MAPPO_mean_reward", "S2_MAPPO_minus_recalibrated"]]
    recal.to_csv(C.HERE / "recalibrated_hysteresis.csv", index=False)

    # decision boundary
    thr = pd.read_csv(C.HERE / "bundle_thresholds.csv").set_index("world")
    db, _ = boundary(sets, thr)
    db = db.merge(allr[["ckpt_id", "family", "seed"]].drop_duplicates(), on="ckpt_id", how="left")
    db.to_csv(C.HERE / "decision_boundary.csv", index=False)
    dsum = []
    for s, mem in sets.items():
        for c, ids in mem.items():
            g = db[db.ckpt_id.isin(ids)]
            num = [x for x in g.columns if x not in ("world", "tier", "experiment", "ckpt_id", "family", "seed")]
            for (w, exp), h in g.groupby(["world", "experiment"]):
                dsum.append(dict(set=s, controller=c, world=w, tier=TIER[w], experiment=exp, n_seeds=len(h),
                                 **{x: h[x].mean() for x in num}))
    dsum = pd.DataFrame(dsum)
    dsum["world"] = pd.Categorical(dsum.world, WORLD_ORDER, ordered=True)
    dsum = dsum.sort_values(["set", "controller", "experiment", "world"])
    dsum = dsum.merge(thr[["energy_breakeven_mbps", "qos_limit_mbps", "decision_mbps"]].reset_index(), on="world", how="left")
    dsum.to_csv(C.HERE / "decision_boundary_summary.csv", index=False)
    ml = measured_labels()
    ml.to_csv(C.HERE / "boundary_measured_labels.csv", index=False)
    ui = []
    for lv in INTERVALS:
        m_all = ml[(ml.level_mbps == lv) & (ml.subset == "all")].set_index("world")
        for _, r in dsum[dsum.experiment == "A"].iterrows():
            ui.append(dict(set=r.set, controller=r.controller, world=r.world, tier=r.tier, level_mbps=lv,
                           interval_mbps=f"[{INTERVALS[lv][0]}, {INTERVALS[lv][1]})",
                           n_cluster_steps_in_interval=r[f"n_steps_{lv}Mbps_interval"],
                           usr_share_in_interval=r[f"usr_share_{lv}Mbps_interval"],
                           measured_usr_unsafe_share_at_level=m_all.loc[str(r.world), "measured_unsafe_share"],
                           world_predicted_usr_unsafe_share_at_level=m_all.loc[str(r.world), "predicted_unsafe_share"],
                           world_false_safe_samples_at_level=m_all.loc[str(r.world), "false_safe"],
                           world_false_unsafe_samples_at_level=m_all.loc[str(r.world), "false_unsafe"],
                           world_qos_limit_mbps=r.qos_limit_mbps))
    pd.DataFrame(ui).to_csv(C.HERE / "boundary_usage_intervals.csv", index=False)

    write_tex(rs, summ, verdicts, recal)
    nums = dict(verdicts={k: {kk: vv for kk, vv in v.items() if not kk.startswith("worlds_")} for k, v in verdicts.items()},
                n_episodes_A=int((allr.experiment == "A").sum()), n_episodes_B=int((allr.experiment == "B").sum()))
    (C.HERE / "analysis_numbers.json").write_text(json.dumps(nums, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(json.dumps({k: (v["verdict"], v["n1_MAPPO_first"], v["r_I_delta_IPPO_nonpositive"], v["r_H_delta_hyst_nonpositive"],
                          v["p1_MAPPO_dominated"], v["m_magnitude_raw_pareto_ok"]) for k, v in verdicts.items()}, indent=0))


if __name__ == "__main__":
    main()
