"""Step 2 — non-negotiable reproduction check (run BEFORE any alternative world).

Part 1  Pinned twin as installed and pinned by the repository (UPF_NDT v0.4.0
        @95de456, switching_costs thesis-v1.2). New evaluator (results/main,
        world 'pinned') against the two recorded evaluations of the same
        checkpoints under the same twin:
          1a Task-C per-step rollouts (reports/wcnc2027-shared-policy-ablation/
             rollouts/*.npz, float64): every metric and reward component, and
             the per-step action sequence
          1b reports/phase-7/multiseed_summary_v04twin.json per-seed records
          1c internal: Experiment-B replay of the pinned actions in the pinned
             world must equal Experiment A exactly
Part 2  Manuscript Table I (reports/paper-mascots/paper.tex). Table I predates
        twin v0.4.0 (reports/phase-7/switching_fix_reeval.md), so it is
        re-evaluated with the pre-fix twin (vendored UPF_NDT v0.3.0 @171146a +
        switching_costs thesis-v1.1, md5 e1456485, the DVC-locked file of that
        time; results/table1_v030). The steady-state surrogate bundle is the
        same pinned bundle in both twin versions.
          2a per-seed traceability against multiseed_summary_n8.json (the record
             behind Table I's MAPPO and IPPO rows): total AND all 10 per-cluster
             rewards must match
          2b every Table I row against its printed (rounded) values
          2c centralized PPO: exhaustive search over all 4-distinct-seed
             combinations of every surviving centralized checkpoint

Tolerances
  recorded full-precision values (1a, 1b, 2a): |diff| <= 0.01 reward units /
    0.01 Wh; rates and counts exact to 1e-9; per-step actions identical
  printed Table I values (2b, 2c): half a unit in the last printed digit —
    reward mean and SD +-0.5, Wh +-0.5, QoSv% +-0.005, USR% +-0.05, Sw/d +-0.005
    (SD is the population SD, ddof=0, as in the n8 record)

Outputs: reproduction/*.csv, reproduction/table1_traceability.json,
         reproduction/reproduction_numbers.json
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import rb_common as C
from evaluate_worlds import safe

OUT = C.HERE / "reproduction"
TASKC = C.ROOT / "reports" / "wcnc2027-shared-policy-ablation"
PAPER = C.ROOT / "reports" / "paper-mascots" / "paper.tex"
TABLE1 = {   # paper.tex lines 540-545 (verbatim values)
    "MAPPO": dict(reward=-5655, sd=98, n=8, wh=1966, qosv=0.54, usr=8.1, swd=2.92),
    "IPPO": dict(reward=-5812, sd=110, n=8, wh=1973, qosv=0.57, usr=8.2, swd=1.99),
    "Centralized PPO": dict(reward=-8542, sd=1744, n=4, wh=2066, qosv=1.18, usr=4.3, swd=0.44),
    "Hysteresis (b=50)": dict(reward=-6261, sd=None, n=None, wh=1995, qosv=0.68, usr=7.5, swd=1.06),
    "Always-DPDK": dict(reward=-7201, sd=None, n=None, wh=2071, qosv=0.38, usr=0.0, swd=0.00),
}
TT = dict(reward=0.5, sd=0.5, wh=0.5, qosv=0.005, usr=0.05, swd=0.005)
EPS_R, EPS_RATE = 0.01, 1e-9


def load(tag):
    return pd.read_csv(C.HERE / "results" / tag / "pinned__A.csv")


def rel(p):
    return str(Path(p).relative_to(C.ROOT)) if str(p).startswith(str(C.ROOT)) else p


def taskc_metrics(npz: Path) -> dict:
    d = np.load(npz, allow_pickle=True)
    steps = int(d["steps"])
    act = d["action"].astype(int)
    return dict(reward=float(d["reward"].sum()), energy_wh=float(d["power_watts"].sum() * float(d["step_h"])),
                qos_violation_rate=float((~d["is_safe"]).sum() / (steps * C.K)),
                usr_share=float(d["usr"].sum() / (steps * C.K)),
                switches=int((act[1:] != act[:-1]).sum()),
                energy_penalty=float(d["energy_term"].sum()), qos_penalty=float(d["qos_penalty"].sum()),
                switch_penalty=float(d["switch_penalty"].sum()), cooldown_penalty=float(d["cooldown_penalty"].sum()),
                action=act)


def part1(main: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    allr = main[main.cluster == "all"].copy()
    allr["relpath"] = [rel(C.EXP / r.split("|", 1)[1]) if r.split("|", 1)[1] != "-" else "-" for r in allr.ckpt_id]
    rows = []
    # ---- 1a Task C per-step rollouts
    man = json.loads((TASKC / "rollouts" / "manifest.json").read_text())
    base_map = {"always-DPDK": "Always-DPDK|-", "Hysteresis (b=50 Mbps)": "Hysteresis (fixed, t_up=81, b=50 Mbps)|-"}
    for m in man:
        if m["set"] in base_map:
            ours = allr[allr.ckpt_id == base_map[m["set"]]]
        else:
            ckrel = m["checkpoint"] if m["set"] not in ("IPPO",) and "ippo" not in m["set"].lower() else m["checkpoint"]
            ours = allr[allr.relpath == ckrel]
        ref = taskc_metrics(TASKC / m["file"])
        if len(ours) != 1:
            rows.append(dict(check="1a Task-C rollout", item=f"{m['set']} seed {m['seed']}", metric="match",
                             ours=np.nan, ref=np.nan, abs_diff=np.nan, tol=0, passed=False,
                             note=f"{len(ours)} matching checkpoints in new evaluation"))
            continue
        o = ours.iloc[0]
        steps = np.load(C.ROLLOUTS / "main" / "pinned" / "A" / f"{safe(o.ckpt_id)}.npz")
        for k in ("reward", "energy_wh", "energy_penalty", "qos_penalty", "switch_penalty", "cooldown_penalty",
                  "qos_violation_rate", "usr_share", "switches"):
            tol = EPS_RATE if k in ("qos_violation_rate", "usr_share", "switches") else EPS_R
            diff = abs(float(o[k]) - float(ref[k]))
            rows.append(dict(check="1a Task-C rollout", item=f"{o.ckpt_id}", metric=k, ours=float(o[k]),
                             ref=float(ref[k]), abs_diff=diff, tol=tol, passed=diff <= tol, note=m["set"]))
        same = bool(np.array_equal(steps["action"].astype(int), ref["action"]))
        rows.append(dict(check="1a Task-C rollout", item=o.ckpt_id, metric="per-step action sequence (T x K)",
                         ours=float(same), ref=1.0, abs_diff=0.0 if same else 1.0, tol=0, passed=same, note=m["set"]))
    # ---- 1b v04twin per-seed records
    v04 = json.loads((C.ROOT / "reports/phase-7/multiseed_summary_v04twin.json").read_text())["aggregated"]
    sets = {"MAPPO": ("MAPPO", C.LEGACY_MAPPO_V04, "mappo_best.pt"),
            "IPPO-ensemble": ("IPPO", C.LEGACY_IPPO_V04, None),
            "centralised-PPO": ("Centralized PPO", C.LEGACY_CENTRAL_V04, "ppo_multi_site.zip")}
    for key, (fam, dirs, fname) in sets.items():
        for s, rec in v04[key]["per_seed"].items():
            cid = f"{fam}|{dirs[int(s)]}" + (f"/{fname}" if fname else "")
            o = allr[allr.ckpt_id == cid].iloc[0]
            pc = main[(main.ckpt_id == cid) & (main.cluster != "all")].sort_values("cluster", key=lambda x: x.astype(int))
            pairs = [("reward", float(rec["total_reward_unweighted"]), EPS_R),
                     ("energy_wh", float(rec["total_energy_wh"]), EPS_R),
                     ("qos_violation_rate", float(rec["agg_unsafe_rate"]), EPS_RATE),
                     ("usr_share", float(rec["agg_usr_rate"]), EPS_RATE),
                     ("switches", float(rec["agg_n_switches"]), EPS_RATE)]
            for k, refv, tol in pairs:
                diff = abs(float(o[k]) - refv)
                rows.append(dict(check="1b v04twin.json", item=cid, metric=k, ours=float(o[k]), ref=refv,
                                 abs_diff=diff, tol=tol, passed=diff <= tol, note=f"{key} seed {s}"))
            d = float(np.max(np.abs(pc.reward.to_numpy() - np.array(rec["per_cluster_total_reward"], float))))
            rows.append(dict(check="1b v04twin.json", item=cid, metric="per-cluster reward (max over 10)",
                             ours=np.nan, ref=np.nan, abs_diff=d, tol=EPS_R, passed=d <= EPS_R, note=f"{key} seed {s}"))
    rec = v04["always-DPDK"]["per_seed"]["0"]
    o = allr[allr.ckpt_id == "Always-DPDK|-"].iloc[0]
    for k, refv in (("reward", rec["total_reward_unweighted"]), ("energy_wh", rec["total_energy_wh"])):
        diff = abs(float(o[k]) - float(refv))
        rows.append(dict(check="1b v04twin.json", item="Always-DPDK", metric=k, ours=float(o[k]), ref=float(refv),
                         abs_diff=diff, tol=EPS_R, passed=diff <= EPS_R, note="always-DPDK"))
    # ---- 1c replay == closed loop in the pinned world
    b = pd.read_csv(C.HERE / "results" / "main" / "pinned__B.csv")
    num = [c for c in main.columns if c not in ("tag", "twin", "world", "tier", "protocol", "config", "fold", "experiment",
                                                "ckpt_id", "family", "seed", "cohorts", "source", "cluster", "seconds")]
    mm = main.merge(b, on=["ckpt_id", "cluster"], suffixes=("_A", "_B"))
    worst = max(float(np.max(np.abs(mm[f"{c}_A"] - mm[f"{c}_B"]) / np.maximum(1.0, np.abs(mm[f"{c}_A"])))) for c in num)
    acts_equal = all(np.array_equal(np.load(C.ROLLOUTS / "main" / "pinned" / "A" / f"{safe(c)}.npz")["action"],
                                    np.load(C.ROLLOUTS / "main" / "pinned" / "B" / f"{safe(c)}.npz")["action"])
                     for c in mm.ckpt_id.unique())
    rows.append(dict(check="1c replay == closed loop (pinned)", item=f"{mm.ckpt_id.nunique()} controllers x 11 rows",
                     metric="all metrics (relative diff) and per-step actions", ours=np.nan, ref=np.nan, abs_diff=worst,
                     tol=1e-12, passed=bool(worst <= 1e-12 and acts_equal),
                     note=f"actions identical: {acts_equal}; residual is floating-point round-off (<= 2 ulp)"))
    df = pd.DataFrame(rows)
    return df, dict(n_checks=len(df), n_failed=int((~df.passed).sum()))


def agg_row(df_all: pd.DataFrame) -> dict:
    r = df_all.reward.to_numpy()
    return dict(reward=float(r.mean()), sd=float(r.std(ddof=0)) if len(r) > 1 else None, n=len(r),
                wh=float(df_all.energy_wh.mean()), qosv=100 * float(df_all.qos_violation_rate.mean()),
                usr=100 * float(df_all.usr_share.mean()), swd=float(df_all.switches_per_cluster_day.mean()))


def part2(v030: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    n8 = json.loads((C.ROOT / "reports/phase-7/multiseed_summary_n8.json").read_text())["aggregated"]
    allr = v030[v030.cluster == "all"]
    pcl = v030[v030.cluster != "all"].copy()
    pcl["k"] = pcl.cluster.astype(int)
    trace, traced = [], {"MAPPO": {}, "IPPO": {}}
    for key, fam in (("MAPPO", "MAPPO"), ("IPPO-ensemble", "IPPO")):
        for s, rec in n8[key]["per_seed"].items():
            ref_pc = np.array(rec["per_cluster_total_reward"], float)
            cands = allr[(allr.family == fam) & (allr.seed == int(s))]
            best, matches = None, []
            for _, o in cands.iterrows():
                pc = pcl[pcl.ckpt_id == o.ckpt_id].sort_values("k").reward.to_numpy()
                d_tot = abs(o.reward - rec["total_reward_unweighted"])
                d_pc = float(np.max(np.abs(pc - ref_pc)))
                if best is None or d_tot < best[1]:
                    best = (o, d_tot, d_pc)
                if d_tot <= EPS_R and d_pc <= EPS_R:
                    matches.append(o.ckpt_id)
            o, d_tot, d_pc = best
            if matches:   # several byte-identical copies may match: take the earliest-created directory
                def _mtime(cid):
                    p = C.EXP / cid.split("|", 1)[1]
                    return (p if p.is_dir() else p.parent).stat().st_mtime
                traced[fam][int(s)] = min(matches, key=_mtime)
            ok = int(s) in traced[fam]
            trace.append(dict(controller=fam, seed=int(s), n8_reward=rec["total_reward_unweighted"],
                              matched_checkpoint=traced[fam].get(int(s), ""), all_matching_checkpoints=" ; ".join(matches),
                              matching_sha256=" ; ".join(sorted({C.sha256_many(C.checkpoint_files(dict(kind="ippo" if fam == "IPPO" else "mappo", path=str(C.EXP / m.split("|", 1)[1])))) [:12] for m in matches})),
                              closest_checkpoint=o.ckpt_id,
                              closest_reward=o.reward, closest_abs_diff_total=d_tot, closest_max_abs_diff_per_cluster=d_pc,
                              n_candidates=len(cands), traced=ok))
    rec = n8["always-DPDK"]["per_seed"]["0"]
    o = allr[allr.ckpt_id == "Always-DPDK|-"].iloc[0]
    trace.append(dict(controller="Always-DPDK", seed=0, n8_reward=rec["total_reward_unweighted"], matched_checkpoint="-",
                      closest_checkpoint="-", closest_reward=o.reward,
                      closest_abs_diff_total=abs(o.reward - rec["total_reward_unweighted"]),
                      closest_max_abs_diff_per_cluster=np.nan, n_candidates=1,
                      traced=abs(o.reward - rec["total_reward_unweighted"]) <= EPS_R))
    trace = pd.DataFrame(trace)

    rows = []

    def compare(name, got, note):
        ref = TABLE1[name]
        for k in ("reward", "sd", "n", "wh", "qosv", "usr", "swd"):
            if ref[k] is None:
                continue
            g = got.get(k)
            if k == "n":
                ok = g == ref[k]
                diff = abs((g or 0) - ref[k])
            else:
                ok = g is not None and abs(g - ref[k]) <= TT[k]
                diff = abs(g - ref[k]) if g is not None else np.nan
            rows.append(dict(row=name, metric=k, table1=ref[k], reproduced=g, abs_diff=diff,
                             tol=TT.get(k, 0), passed=bool(ok), note=note))

    for fam, name in (("MAPPO", "MAPPO"), ("IPPO", "IPPO")):
        ids = list(traced[fam].values())
        sub = allr[allr.ckpt_id.isin(ids)]
        compare(name, agg_row(sub) if len(sub) else {}, f"{len(ids)}/8 seeds traced: {sorted(traced[fam])}")
    compare("Hysteresis (b=50)", agg_row(allr[allr.family == "Hysteresis (fixed)"]), "t_up 81, t_down 31 Mbps, cooldown 1")
    compare("Always-DPDK", agg_row(allr[allr.family == "Always-DPDK"]), "")

    # ---- 2c centralized exhaustive search
    cen = allr[allr.family == "Centralized PPO"].reset_index(drop=True)
    ref = TABLE1["Centralized PPO"]
    idx = np.array(list(itertools.combinations(range(len(cen)), 4)))
    ss = np.sort(cen.seed.to_numpy()[idx], axis=1)
    idx = idx[(np.diff(ss, axis=1) > 0).all(axis=1)]             # four distinct seeds
    R = cen.reward.to_numpy()[idx]
    A = dict(reward=R.mean(1), sd=R.std(1, ddof=0), wh=cen.energy_wh.to_numpy()[idx].mean(1),
             qosv=100 * cen.qos_violation_rate.to_numpy()[idx].mean(1),
             usr=100 * cen.usr_share.to_numpy()[idx].mean(1),
             swd=cen.switches_per_cluster_day.to_numpy()[idx].mean(1))
    score = sum(np.abs(A[k] - ref[k]) / TT[k] for k in A)
    within = np.all([np.abs(A[k] - ref[k]) <= TT[k] for k in A], axis=0)
    rs_within = (np.abs(A["reward"] - ref["reward"]) <= TT["reward"]) & (np.abs(A["sd"] - ref["sd"]) <= TT["sd"])
    order = np.argsort(score)[:20]
    combos = pd.DataFrame([dict(score=float(score[i]), members=" ; ".join(cen.ckpt_id.to_numpy()[idx[i]]),
                                seeds=sorted(cen.seed.to_numpy()[idx[i]].tolist()), n=4,
                                **{k: float(A[k][i]) for k in A},
                                reward_sd_within_tol=bool(rs_within[i]), all_within_tol=bool(within[i]))
                           for i in order])
    combos.to_csv(OUT / "table1_centralized_search_top20.csv", index=False)
    n_combos, n_rs_within, n_within = len(idx), int(rs_within.sum()), int(within.sum())
    if n_within:
        hit = combos[combos.all_within_tol].iloc[0].to_dict()
        compare("Centralized PPO", hit, f"traced to {hit['members']}")
    else:
        b0 = combos.iloc[0].to_dict()
        compare("Centralized PPO", b0, f"NOT traced: {n_combos} four-distinct-seed combinations of {len(cen)} "
                                       f"checkpoints searched ({n_rs_within} match reward mean and SD only); "
                                       f"closest shown ({b0['members']})")
    extras = []
    reg_ids = {r["ckpt_id"] for r in C.checkpoint_registry()}
    for fam in ("MAPPO", "IPPO"):
        for s, cid in traced[fam].items():
            if cid not in reg_ids:
                o = allr[allr.ckpt_id == cid].iloc[0]
                kind = "mappo" if fam == "MAPPO" else "ippo"
                path = C.EXP / cid.split("|", 1)[1]
                extras.append(dict(ckpt_id=cid, family=fam, kind=kind, seed=int(s), path=str(path),
                                   source=str(o.source), cohorts=["T"]))
    info = dict(traced=traced, n_centralized_combinations=n_combos, n_centralized_candidates=len(cen),
                n_centralized_combinations_reward_and_sd_within_tol=n_rs_within,
                n_centralized_combinations_all_within_tol=n_within,
                centralized_traced=bool(n_within > 0), extra_entries=extras,
                traceable_ckpt_ids=[c for f in traced.values() for c in f.values()])
    return trace, pd.DataFrame(rows), info


def main():
    OUT.mkdir(exist_ok=True)
    main_df = load("main")
    p1, s1 = part1(main_df)
    p1.to_csv(OUT / "part1_pinned_v040_checks.csv", index=False)
    v030 = load("table1_v030")
    trace, p2, info = part2(v030)
    trace.to_csv(OUT / "part2_n8_per_seed_traceability.csv", index=False)
    p2.to_csv(OUT / "part2_table1_rows.csv", index=False)
    info["paper_tex_sha256"] = C.sha256(PAPER)
    (OUT / "table1_traceability.json").write_text(json.dumps(info, indent=1, default=str))
    summ = dict(part1=s1, part1_by_check=p1.groupby("check").passed.agg(["size", "sum"]).reset_index().to_dict("records"),
                part2_rows=p2.groupby("row").passed.all().to_dict(), traced=info["traced"],
                centralized_traced=info["centralized_traced"])
    (OUT / "reproduction_numbers.json").write_text(json.dumps(summ, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(p1.groupby("check").passed.agg(["size", "sum"]))
    if (~p1.passed).any():
        print(p1[~p1.passed].to_string())
    print(trace.to_string())
    print(p2.to_string())
    print(json.dumps({k: v for k, v in info.items() if k != "extra_entries"}, default=str, indent=1))


if __name__ == "__main__":
    main()
