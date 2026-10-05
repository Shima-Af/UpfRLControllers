"""Metrics, aggregates, paired tests, decision rule and LaTeX tables.

Reads rollouts/*.npz written by evaluate_all.py. Every statistic and the
decision rule follow analysis_plan.md (fixed before the results existed).

Outputs:
  all_seed_metrics.csv                per controller x seed x cluster ('all' = fleet)
  aggregate_metrics.csv               mean, SD, median, t- and bootstrap 95 % CIs
  paired_comparisons.csv              seed-matched contrasts for every metric
  decision.json                       inputs and outcome of the pre-registered rule
  centralized_ppo_8seed_summary.tex
  shared_policy_ablation_summary.tex
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
K = 10
STEP_MIN = 15.0
PRIMARY = ["MAPPO", "Shared-PPO", "IPPO", "Centralized PPO"]
BASELINES = ["Hysteresis (b=50 Mbps)", "always-DPDK"]
BOOT_SEED = 20260914
N_BOOT = 10_000
COMPONENTS = ["energy_term", "qos_penalty", "switch_penalty", "cooldown_penalty"]
METRICS = {  # name: (label, better)
    "reward": ("Test reward (higher is better)", "higher"),
    "energy_wh": ("Energy [Wh]", "lower"),
    "qos_violation_rate": ("QoS-violation rate (is_safe = False)", "lower"),
    "qscore_below_tau_rate": ("q_score < tau rate", "lower"),
    "usr_share": ("USR share", "neutral"),
    "switches_per_cluster_day": ("Switches per cluster per day", "lower"),
    "cost_energy_term": ("Reward component: alpha*SEC (cost)", "lower"),
    "cost_qos_penalty": ("Reward component: QoS penalty (cost)", "lower"),
    "cost_switch_penalty": ("Reward component: switching penalty (cost)", "lower"),
    "cost_cooldown_penalty": ("Reward component: cooldown penalty (cost)", "lower"),
}


def run_metrics(npz: Path) -> list[dict]:
    d = np.load(npz, allow_pickle=True)
    steps = int(d["steps"])
    days = steps * STEP_MIN / (24 * 60)
    act = d["action"].astype(int)
    sw = (act[1:] != act[:-1]).sum(axis=0)                     # per cluster
    usr = d["usr"]
    sw_realised = (usr[1:] != usr[:-1]).sum(axis=0)
    assert np.array_equal(sw, sw_realised), "requested and realised switches differ"
    comp_sum = sum(d[c] for c in COMPONENTS)
    assert np.allclose(d["reward"], -comp_sum, atol=1e-6), "reward != -(sum of components)"
    rows = []
    for cl in ["all", *range(K)]:
        sl = slice(None) if cl == "all" else slice(cl, cl + 1)
        n_cs = steps * (K if cl == "all" else 1)
        r = dict(
            cluster=cl,
            reward=float(d["reward"][:, sl].sum()),
            energy_wh=float(d["power_watts"][:, sl].sum() * float(d["step_h"])),
            qos_violation_rate=float((~d["is_safe"][:, sl]).sum() / n_cs),
            qscore_below_tau_rate=float((d["q_score"][:, sl] < float(d["tau"])).sum() / n_cs),
            usr_share=float(usr[:, sl].sum() / n_cs),
            switches=int(sw[sl].sum()),
            switches_per_cluster_day=float(sw[sl].sum() / (K if cl == "all" else 1) / days),
            switching_energy_wh=float(d["switching_energy_wh"][:, sl].sum()),
            mean_load_gbps=float(d["actual_load_gbps"][:, sl].mean()),
        )
        for c in COMPONENTS:
            r[f"cost_{c}"] = float(d[c][:, sl].sum())
        rows.append(r)
    return rows


def t_ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return (x.mean() - h, x.mean() + h)


def boot_ci(x, rng):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    m = x[idx].mean(axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def exact_signflip_p(d):
    d = np.asarray(d, float)
    obs = abs(d.mean())
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=len(d))))
    null = np.abs((signs * d).mean(axis=1))
    return float(np.mean(null >= obs - 1e-12))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    manifest = json.loads((HERE / "rollouts" / "manifest.json").read_text())
    rows = []
    for m in manifest:
        for r in run_metrics(HERE / m["file"]):
            rows.append(dict(controller=m["set"], seed=m["seed"], checkpoint=m["checkpoint"],
                             group=("primary" if m["set"] in PRIMARY else
                                    "baseline" if m["set"] in BASELINES else "legacy"), **r))
    df = pd.DataFrame(rows)
    lead = ["group", "controller", "seed", "cluster", "checkpoint"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    df.to_csv(HERE / "all_seed_metrics.csv", index=False, float_format="%.6g")

    percl = df[df.cluster.astype(str) != "all"]
    cols = ["reward", "energy_wh", "qos_violation_rate", "usr_share", "switches_per_cluster_day",
            "cost_energy_term", "cost_qos_penalty", "cost_switch_penalty", "cost_cooldown_penalty", "mean_load_gbps"]
    pcs = percl.groupby(["group", "controller", "cluster"])[cols].agg(["mean", "std", "median", "min", "max"])
    pcs.columns = [f"{a}_{b}" for a, b in pcs.columns]
    pcs.insert(0, "n_seeds", percl.groupby(["group", "controller", "cluster"]).size())
    pcs.reset_index().to_csv(HERE / "per_cluster_summary.csv", index=False, float_format="%.6g")

    fleet = df[df.cluster.astype(str) == "all"]
    rng = np.random.default_rng(BOOT_SEED)
    agg = []
    for ctrl, g in fleet.groupby("controller", sort=False):
        for met in [*METRICS, "switching_energy_wh"]:
            x = g.sort_values("seed")[met].to_numpy(float)
            lo, hi = t_ci(x)
            blo, bhi = boot_ci(x, rng)
            agg.append(dict(controller=ctrl, group=g.group.iloc[0], metric=met, n=len(x),
                            seeds=" ".join(map(str, sorted(g.seed))),
                            mean=x.mean(), sd=x.std(ddof=1) if len(x) > 1 else np.nan, median=np.median(x),
                            min=x.min(), max=x.max(), t_ci_lo=lo, t_ci_hi=hi, boot_ci_lo=blo, boot_ci_hi=bhi))
    agg = pd.DataFrame(agg)
    agg.to_csv(HERE / "aggregate_metrics.csv", index=False, float_format="%.6g")

    # ── paired comparisons ─────────────────────────────────────────────────
    contrasts = [("MAPPO", "Shared-PPO", "Delta_crit (centralized vs local critic)"),
                 ("Shared-PPO", "IPPO", "Delta_share (shared vs independent policies; confounded, see plan)"),
                 ("MAPPO", "IPPO", "G (MAPPO vs IPPO)"),
                 ("MAPPO", "Centralized PPO", "MAPPO vs centralized PPO"),
                 ("Shared-PPO", "Centralized PPO", "Shared-PPO vs centralized PPO (exploratory)")]
    pc = []
    for a, b, label in contrasts:
        ga = fleet[fleet.controller == a].set_index("seed")
        gb = fleet[fleet.controller == b].set_index("seed")
        common = sorted(set(ga.index) & set(gb.index))
        if len(common) < 2:
            continue
        for met, (mlabel, better) in METRICS.items():
            d = (ga.loc[common, met] - gb.loc[common, met]).to_numpy(float)
            idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
            bm = d[idx].mean(axis=1)
            sd = d.std(ddof=1)
            try:
                w = stats.wilcoxon(d, alternative="two-sided", method="exact").pvalue if np.any(d != 0) else 1.0
            except ValueError:
                w = np.nan
            dz = d.mean() / sd if sd > 0 else np.nan
            n = len(d)
            gz = dz * (1 - 3 / (4 * (n - 1) - 1)) if np.isfinite(dz) else np.nan
            a_better = (d > 0) if better == "higher" else (d < 0) if better == "lower" else np.zeros_like(d, bool)
            pc.append(dict(contrast=label, A=a, B=b, metric=met, metric_label=mlabel, better=better, n_pairs=n,
                           seeds=" ".join(map(str, common)),
                           per_seed_diff_A_minus_B=json.dumps([round(float(v), 6) for v in d]),
                           mean_diff=d.mean(), sd_diff=sd, median_diff=np.median(d),
                           boot_ci_lo=np.percentile(bm, 2.5), boot_ci_hi=np.percentile(bm, 97.5),
                           perm_p_two_sided_exact=exact_signflip_p(d), wilcoxon_p_two_sided_exact=w,
                           cohens_dz=dz, hedges_gz=gz,
                           n_seeds_A_better=int(a_better.sum()) if better != "neutral" else np.nan,
                           relative_to_B_mean=d.mean() / abs(gb.loc[common, met].mean()) if gb.loc[common, met].mean() != 0 else np.nan))
    pc = pd.DataFrame(pc)
    prim = pc[(pc.metric == "reward") & pc.contrast.str.match(r"^(Delta_crit|Delta_share|G |MAPPO vs centralized)")]
    pc["holm_p_primary_reward_family"] = np.nan
    pc.loc[prim.index, "holm_p_primary_reward_family"] = holm(prim.perm_p_two_sided_exact.to_numpy())
    pc.to_csv(HERE / "paired_comparisons.csv", index=False, float_format="%.6g")

    # ── pre-registered decision rule ───────────────────────────────────────
    decision = {}
    if all(c in set(fleet.controller) for c in ("MAPPO", "Shared-PPO", "IPPO")):
        def row(prefix):
            return pc[(pc.metric == "reward") & pc.contrast.str.startswith(prefix)].iloc[0]
        rc, rs, rg = row("Delta_crit"), row("Delta_share"), row("G ")
        sig = lambda r: bool((r.boot_ci_lo > 0 or r.boot_ci_hi < 0) and r.perm_p_two_sided_exact < 0.05)
        M = 0.01 * abs(fleet[fleet.controller == "MAPPO"].reward.mean())
        s1 = rc.mean_diff > 0 and sig(rc) and rc.mean_diff >= M
        s4 = (not s1) and rc.mean_diff <= 0
        s2_conditions = bool(rg.mean_diff > 0 and sig(rg) and sig(rs) and rs.mean_diff >= 0.5 * rg.mean_diff)
        if s1:
            outcome = "S1"
        elif s4:
            outcome = "S4"
        elif s2_conditions:
            outcome = "S2"
        else:
            outcome = "S3"
        decision = dict(
            materiality_M=M,
            delta_crit=dict(mean=rc.mean_diff, ci=[rc.boot_ci_lo, rc.boot_ci_hi], p=rc.perm_p_two_sided_exact, significant=sig(rc)),
            delta_share=dict(mean=rs.mean_diff, ci=[rs.boot_ci_lo, rs.boot_ci_hi], p=rs.perm_p_two_sided_exact, significant=sig(rs)),
            G=dict(mean=rg.mean_diff, ci=[rg.boot_ci_lo, rg.boot_ci_hi], p=rg.perm_p_two_sided_exact, significant=sig(rg)),
            S1=bool(s1), S4=bool(s4), S2_conditions_hold=s2_conditions, outcome=outcome,
            statements={"S1": "The centralized critic materially improves over a shared local critic.",
                        "S2": "Parameter sharing explains most of MAPPO's advantage.",
                        "S3": "The ablation is inconclusive.",
                        "S4": "MAPPO does not outperform the ablation."},
        )
        (HERE / "decision.json").write_text(json.dumps(decision, indent=2, default=float))

    if decision:
        sensitivity(fleet, decision)
    write_tex(agg, pc, fleet)
    print(agg[agg.metric.isin(["reward", "energy_wh", "qos_violation_rate", "usr_share", "switches_per_cluster_day"])]
          .pivot_table(index="controller", columns="metric", values="mean", sort=False).round(4).to_string())
    print(pc[pc.metric == "reward"][["contrast", "n_pairs", "mean_diff", "boot_ci_lo", "boot_ci_hi",
                                      "perm_p_two_sided_exact", "holm_p_primary_reward_family", "cohens_dz",
                                      "n_seeds_A_better"]].round(4).to_string())
    print(json.dumps(decision, indent=1, default=float))


def hodges_lehmann(d):
    d = np.asarray(d, float)
    w = [(d[i] + d[j]) / 2 for i in range(len(d)) for j in range(i, len(d))]
    return float(np.median(w))


def sensitivity(fleet, decision):
    """POST-HOC robustness checks (not part of the pre-registered rule)."""
    W = fleet.pivot(index="seed", columns="controller", values="reward")
    dpdk = fleet[fleet.controller == "always-DPDK"].reward.iloc[0] if (fleet.controller == "always-DPDK").any() else None
    rng = np.random.default_rng(BOOT_SEED)
    out = {"note": "post-hoc sensitivity analyses; the pre-registered outcome is decision.json['outcome']"}
    contr = {"delta_crit": ("MAPPO", "Shared-PPO"), "delta_share": ("Shared-PPO", "IPPO"), "G": ("MAPPO", "IPPO")}
    M = decision["materiality_M"]
    for key, (a, b) in contr.items():
        d = (W[a] - W[b]).dropna()
        loo = []
        for s_out in d.index:
            dd = d.drop(s_out).to_numpy()
            bm = dd[rng.integers(0, len(dd), size=(N_BOOT, len(dd)))].mean(axis=1)
            loo.append(dict(dropped_seed=int(s_out), mean=float(dd.mean()), ci=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                            p=exact_signflip_p(dd)))
        out[key] = dict(A=a, B=b, per_seed={int(k): float(v) for k, v in d.items()}, median=float(d.median()),
                        hodges_lehmann=hodges_lehmann(d.to_numpy()), leave_one_seed_out=loo,
                        loo_mean_range=[min(x["mean"] for x in loo), max(x["mean"] for x in loo)],
                        loo_sets_significant=sum((x["ci"][0] > 0 or x["ci"][1] < 0) and x["p"] < 0.05 for x in loo),
                        loo_sets_significant_and_material=(sum((x["ci"][0] > 0) and x["p"] < 0.05 and x["mean"] >= M for x in loo)
                                                          if key == "delta_crit" else None))
    if dpdk is not None:
        Wt = W[["MAPPO", "Shared-PPO", "IPPO"]].dropna()          # the 8 training seeds (drops baselines' seed 0)
        bad = {c: [int(s) for s in Wt.index if Wt.loc[s, c] < dpdk] for c in ("MAPPO", "Shared-PPO", "IPPO")}
        excl = sorted(set(sum(bad.values(), [])))
        keep = [s for s in Wt.index if s not in excl]
        res = {}
        for key, (a, b) in contr.items():
            dd = (W.loc[keep, a] - W.loc[keep, b]).to_numpy()
            bm = dd[rng.integers(0, len(dd), size=(N_BOOT, len(dd)))].mean(axis=1)
            res[key] = dict(n=len(dd), mean=float(dd.mean()), median=float(np.median(dd)),
                            ci=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))], p=exact_signflip_p(dd))
        g = res["G"]["mean"]
        res["share_of_G"] = {"delta_crit": res["delta_crit"]["mean"] / g if g else None,
                             "delta_share": res["delta_share"]["mean"] / g if g else None}
        out["excluding_seeds_with_a_trio_run_worse_than_always_DPDK"] = dict(always_dpdk_reward=float(dpdk), runs_worse=bad,
                                                                            excluded_seeds=excl, **res)
    (HERE / "sensitivity_posthoc.json").write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return out


def _fmt(x, nd=0):
    return "--" if x is None or not np.isfinite(x) else f"{x:,.{nd}f}".replace(",", "{,}").replace("-", "$-$")


def write_tex(agg, pc, fleet):
    A = agg.set_index(["controller", "metric"])
    days = 1009 * STEP_MIN / (24 * 60)

    def cell(ctrl, met, nd, scale=1.0):
        if (ctrl, met) not in A.index:
            return "--"
        r = A.loc[(ctrl, met)]
        return _fmt(r["mean"] * scale, nd)

    lines = [r"% Generated by reports/wcnc2027-shared-policy-ablation/analyze.py -- do not edit by hand.",
             r"\begin{table*}[!t]", r"\centering",
             r"\caption{Held-out test slice ($K{=}10$, 1{,}009 steps, deterministic actions). All learned "
             r"controllers retrained with identical code on seeds \{1,7,13,23,42,64,77,99\}; mean $\pm$ SD, "
             r"median and 95\% t-CI of the reward over seeds. Wh: modelled UPF-attributed energy; QoSv: "
             r"QoS-violation rate; Sw/d: switches per cluster per day.}",
             r"\label{tab:central8}", r"\footnotesize", r"\setlength{\tabcolsep}{2.6pt}",
             r"\begin{tabular}{@{}lcrrrrrrr@{}}", r"\toprule",
             r"Controller & $n$ & Reward & Median & 95\% CI & Wh & QoSv\% & USR\% & Sw/d \\", r"\midrule"]
    for ctrl in [*PRIMARY, *BASELINES]:
        if (ctrl, "reward") not in A.index:
            continue
        r = A.loc[(ctrl, "reward")]
        n = int(r["n"])
        rew = _fmt(r["mean"]) + (r"$\,\pm\,$" + _fmt(r["sd"]) if n > 1 else "")
        ci = f"[{_fmt(r['t_ci_lo'])}, {_fmt(r['t_ci_hi'])}]" if n > 1 else "--"
        lines.append(" & ".join([ctrl.replace("b=50 Mbps", "$b{=}50$\\,Mbps"), str(n) if n > 1 else "--", rew,
                                 _fmt(r["median"]) if n > 1 else "--", ci,
                                 cell(ctrl, "energy_wh", 0), cell(ctrl, "qos_violation_rate", 2, 100),
                                 cell(ctrl, "usr_share", 1, 100), cell(ctrl, "switches_per_cluster_day", 2)]) + r" \\")
        if ctrl == PRIMARY[-1]:
            lines.append(r"\midrule")
    legacy = [c for c in A.index.get_level_values(0).unique() if c.startswith("legacy Centralized")]
    if legacy:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{9}{@{}l}{\emph{Existing centralized-PPO checkpoints, same evaluator:}} \\")
        for ctrl in legacy:
            r = A.loc[(ctrl, "reward")]
            lines.append(" & ".join([ctrl.replace("legacy Centralized PPO ", "").replace("_", r"\_"), str(int(r["n"])),
                                     _fmt(r["mean"]) + r"$\,\pm\,$" + _fmt(r["sd"]), _fmt(r["median"]),
                                     f"[{_fmt(r['t_ci_lo'])}, {_fmt(r['t_ci_hi'])}]",
                                     cell(ctrl, "energy_wh", 0), cell(ctrl, "qos_violation_rate", 2, 100),
                                     cell(ctrl, "usr_share", 1, 100), cell(ctrl, "switches_per_cluster_day", 2)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (HERE / "centralized_ppo_8seed_summary.tex").write_text("\n".join(lines) + "\n")

    P = pc[pc.contrast.str.match(r"^(Delta_crit|Delta_share|G )")]
    lab = {"Delta_crit": r"MAPPO $-$ Shared-PPO (critic)", "Delta_share": r"Shared-PPO $-$ IPPO (sharing\,$^\dagger$)",
           "G ": r"MAPPO $-$ IPPO (total)"}
    lines = [r"% Generated by reports/wcnc2027-shared-policy-ablation/analyze.py -- do not edit by hand.",
             r"\begin{table*}[!t]", r"\centering",
             r"\caption{Seed-matched ablation on the held-out test slice (8 seeds). Differences $A-B$ of the "
             r"mean over seeds with paired-bootstrap 95\% CI (10{,}000 resamples), exact two-sided sign-flip "
             r"permutation $p$ (Holm-adjusted across the primary reward contrasts in parentheses), Cohen's "
             r"$d_z$, and seeds on which $A$ is better. $^\dagger$Also differs in implementation and "
             r"hyper-parameters.}",
             r"\label{tab:ablation}", r"\footnotesize", r"\setlength{\tabcolsep}{2.4pt}",
             r"\begin{tabular}{@{}llrcrrc@{}}", r"\toprule",
             r"Contrast & Metric & $\Delta$ & 95\% CI & $p$ (Holm) & $d_z$ & $A$ better \\", r"\midrule"]
    for key in ("Delta_crit", "Delta_share", "G "):
        for met, nd, sc, name in (("reward", 0, 1, "Reward"), ("energy_wh", 1, 1, "Wh"),
                                  ("qos_violation_rate", 3, 100, "QoSv\\%"), ("switches_per_cluster_day", 2, 1, "Sw/d")):
            q = P[(P.contrast.str.startswith(key)) & (P.metric == met)]
            if q.empty:
                continue
            r = q.iloc[0]
            p = f"{r.perm_p_two_sided_exact:.3f}"
            if met == "reward" and np.isfinite(r.holm_p_primary_reward_family):
                p += f" ({r.holm_p_primary_reward_family:.3f})"
            lines.append(" & ".join([lab[key] if met == "reward" else "", name, _fmt(r.mean_diff * sc, nd),
                                     f"[{_fmt(r.boot_ci_lo * sc, nd)}, {_fmt(r.boot_ci_hi * sc, nd)}]", p,
                                     _fmt(r.cohens_dz, 2), f"{int(r.n_seeds_A_better)}/{int(r.n_pairs)}"]) + r" \\")
        if key != "G ":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (HERE / "shared_policy_ablation_summary.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
