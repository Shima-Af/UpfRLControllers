"""Supporting tables from EXISTING saved records (run after m01_stepwise_analysis.py).

  penalized_exits.csv            USR->DPDK exits at steps where USR is QoS-risky, with jump and context features
  exposure_by_run.csv            per run: USR share at 50-81 Mbps, risk onsets while in USR, penalized exits, selected step
  hold_vs_onset_risk_by_band.csv one-step value of holding USR by load band (test traffic, twin risk flag)
  onset_risk_by_split_proxy.csv  P(next load >= 149 Mbps | band) on train / val / test traffic (proxy for the twin flag)
  supporting_numbers.json        forecast timing, cross-cluster lead, proxy agreement, seed-level associations

No policy is run and no environment is stepped.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAL = ROOT / "reports" / "wcnc2027-calibration-robustness" / "rollouts" / "main"
ABL = ROOT / "reports" / "wcnc2027-shared-policy-ablation" / "rollouts"
DATA = ROOT / "data" / "external" / "traffic_forecaster"
SEEDS = [1, 7, 13, 23, 42, 64, 77, 99]


def main():
    tr = np.load(CAL / "traffic_test.npz")
    L, F = tr["load_gbps"] * 1000, tr["forecast_gbps"] * 1000
    T, K = L.shape
    AU, AD = np.load(CAL / "pinned" / "A" / "Always-USR__-.npz"), np.load(CAL / "pinned" / "A" / "Always-DPDK__-.npz")
    risky = (AU["qos_penalty"] - AD["qos_penalty"]) > 0
    save = (AD["energy_term"] - AU["energy_term"]).astype(float)
    nums = {}

    # ---- forecast timing --------------------------------------------------------------
    c = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])  # noqa: E731
    nums["forecast_timing"] = dict(corr_F_t_L_t=c(F, L), corr_F_t_L_t_plus_1=c(F[:-1], L[1:]), corr_F_t_L_t_minus_1=c(F[1:], L[:-1]),
                                   mae_F_t_L_t=float(np.abs(F - L).mean()), mae_persistence=float(np.abs(L[1:] - L[:-1]).mean()),
                                   targets_test_equals_rollout_load=bool(np.allclose(np.load(DATA / "targets_test.npy")[:, 0, :] * 1000, L)))

    # ---- penalized exits ----------------------------------------------------------------
    sw = pd.read_csv(HERE / "switch_steps.csv")
    ex = sw[(sw.direction == "USR->DPDK (exit)") & (sw.cf_class_at_t == "USR QoS-risky")].copy()
    ex["forecast_prev_mbps"] = [F[t - 1, k] for t, k in zip(ex.t, ex.cluster)]
    ex["jump_ratio"] = ex.load_mbps / ex.load_prev_mbps.clip(lower=1)
    ex["load_rising_at_t_minus_1"] = [bool(L[t - 1, k] > L[t - 2, k]) if t >= 2 else False for t, k in zip(ex.t, ex.cluster)]

    def crossed(t, k, lag):
        a, b = t - lag, t - lag - 1
        if b < 0:
            return 0
        n = int(((L[a, :] >= 149) & (L[b, :] < 149)).sum())
        return n - int(L[a, k] >= 149 and L[b, k] < 149)
    ex["other_clusters_crossed_149_at_t_minus_1"] = [crossed(t, k, 1) for t, k in zip(ex.t, ex.cluster)]
    ex["other_clusters_crossed_149_at_t"] = [crossed(t, k, 0) for t, k in zip(ex.t, ex.cluster)]
    ex["step_of_day"] = ex.t % 96
    ex.to_csv(HERE / "penalized_exits.csv", index=False)
    ev = [(t, k) for t in range(2, T) for k in range(K) if L[t - 1, k] < 149 <= L[t, k]]
    nums["penalized_exits"] = dict(
        per_seed_mean=ex.groupby("controller").size().div(8).to_dict(),
        median_by_controller=ex.groupby("controller")[["load_prev_mbps", "load_mbps", "forecast_mbps", "forecast_prev_mbps",
                                                       "q_score", "predicted_loss", "qos_penalty"]].median().round(3).to_dict("index"),
        mean_qos_penalty=float(ex.qos_penalty.mean()),
        share_other_cluster_crossed_one_step_earlier=ex.groupby("controller").other_clusters_crossed_149_at_t_minus_1.apply(lambda x: float((x > 0).mean())).to_dict(),
        base_rate_all_upward_149_crossings=dict(
            n=len(ev),
            share_other_crossed_one_step_earlier=float(np.mean([crossed(t, k, 1) > 0 for t, k in ev])),
            share_other_crossed_same_step=float(np.mean([crossed(t, k, 0) > 0 for t, k in ev]))),
        unique_events=int(len(ex.drop_duplicates(["t", "cluster"]))),
        unique_events_by_cluster=ex.drop_duplicates(["t", "cluster"]).cluster.value_counts().to_dict())

    # ---- exposure per run ----------------------------------------------------------------
    sel = pd.read_csv(HERE / "checkpoint_selection.csv")
    onset = np.zeros((T, K), bool)
    onset[1:] = (~risky[:-1]) & risky[1:]
    band = (L >= 50) & (L < 81)
    rows = []
    for ctrl in ("MAPPO", "Shared-PPO"):
        for s in SEEDS:
            u = np.load(ABL / f"{ctrl}__seed{s}.npz")["usr"].astype(bool)
            exposed = np.zeros((T, K), bool)
            exposed[1:] = u[:-1] & onset[1:]
            q = sel[(sel.controller == ctrl) & (sel.seed == s)].iloc[0]
            rows.append(dict(controller=ctrl, seed=s, usr_steps=int(u.sum()), usr_share_50_81=float(u[band].mean()),
                             usr_steps_in_risky_state=int((u & risky).sum()),
                             risk_onsets=int(onset.sum()), exposed_onsets=int(exposed.sum()),
                             penalized_exits=int(((ex.controller == ctrl) & (ex.seed == s)).sum()),
                             best_step=int(q.best_step), best_val_return=float(q.best_val_return)))
    r = pd.DataFrame(rows)
    r["exits_per_exposed_onset"] = r.penalized_exits / r.exposed_onsets.clip(lower=1)
    r.to_csv(HERE / "exposure_by_run.csv", index=False)
    w = r.pivot(index="seed", columns="controller")
    dec = pd.read_csv(HERE / "decomposition_by_seed.csv")
    d_cost = -dec.groupby("seed").delta_reward.sum()
    paired = {}
    for col in ("usr_share_50_81", "exposed_onsets", "penalized_exits", "best_step"):
        d = w[col]["Shared-PPO"] - w[col]["MAPPO"]
        paired[col] = dict(mean=float(d.mean()), seeds_shared_higher=int((d > 0).sum()), seeds_shared_lower=int((d < 0).sum()))
    sp = lambda a, b: dict(rho=float(stats.spearmanr(a, b).statistic), p=float(stats.spearmanr(a, b).pvalue))  # noqa: E731
    nums["exposure"] = dict(
        means=r.groupby("controller")[["usr_steps", "usr_share_50_81", "exposed_onsets", "penalized_exits", "best_step",
                                       "usr_steps_in_risky_state"]].mean().to_dict("index"),
        usr_steps_in_risky_state_by_run={f"{c} seed {sd}": int(v) for (c, sd), v in
                                         r.set_index(["controller", "seed"]).usr_steps_in_risky_state[lambda x: x > 0].items()},
        exits_equal_exposed_onsets_runs=int((r.penalized_exits == r.exposed_onsets).sum()),
        paired_shared_minus_mappo=paired,
        spearman_16_runs=dict(usr_share_50_81_vs_exposed_onsets=sp(r.usr_share_50_81, r.exposed_onsets),
                              best_step_vs_usr_share_50_81=sp(r.best_step, r.usr_share_50_81),
                              best_step_vs_penalized_exits=sp(r.best_step, r.penalized_exits)),
        spearman_8_seeds=dict(d_usr_share_50_81_vs_d_cost=sp(w.usr_share_50_81["Shared-PPO"] - w.usr_share_50_81["MAPPO"], d_cost),
                              d_exposed_onsets_vs_d_cost=sp(w.exposed_onsets["Shared-PPO"] - w.exposed_onsets["MAPPO"], d_cost)),
        d_cost_per_seed=d_cost.round(2).to_dict())

    # ---- one-step value of holding USR by band (test, twin flag) ------------------------
    qpen = float(ex.qos_penalty.mean())
    rising = np.zeros((T, K), bool)
    rising[1:] = L[1:] > L[:-1]
    hrows = []
    for lo, hi in [(10, 25), (25, 50), (50, 81), (81, 149)]:
        m = (L[:-1] >= lo) & (L[:-1] < hi) & (~risky[:-1])
        for lab, cond in (("", np.ones_like(m)), (" | load rising vs t-1", rising[:-1]), (" | load not rising", ~rising[:-1])):
            mm = m & cond
            on = mm & risky[1:]
            p = on.sum() / max(1, mm.sum())
            hrows.append(dict(band=f"{lo}–{hi}{lab}", n_steps=int(mm.sum()), n_onsets_next_step=int(on.sum()), p_onset_next=p,
                              mean_sec_saving_per_usr_step=float(save[:-1][mm].mean()), expected_exit_qos_penalty=p * qpen,
                              net_expected_per_step_hold=float(save[:-1][mm].mean()) - p * qpen))
    pd.DataFrame(hrows).to_csv(HERE / "hold_vs_onset_risk_by_band.csv", index=False)

    # ---- onset risk by split (proxy: load >= 149 Mbps) -----------------------------------
    nums["proxy_vs_twin_flag_test"] = dict(agreement=float(((L >= 149) == risky).mean()),
                                           risky_but_below_149=int((risky & (L < 149)).sum()),
                                           above_149_but_not_risky=int(((L >= 149) & ~risky).sum()))
    srows = []
    for split in ("train", "val", "test"):
        Ls = np.load(DATA / f"targets_{split}.npy")[:, 0, :] * 1000
        for lo, hi in [(25, 50), (50, 81), (81, 149)]:
            m = (Ls[:-1] >= lo) & (Ls[:-1] < hi)
            on = m & (Ls[1:] >= 149)
            srows.append(dict(split=split, steps=Ls.shape[0], band=f"{lo}–{hi}", n_band_steps=int(m.sum()),
                              p_next_step_ge_149=float(on.sum() / max(1, m.sum())), n_onsets=int(on.sum())))
        srows.append(dict(split=split, steps=Ls.shape[0], band="load < 1 Mbps", n_band_steps=int((Ls < 1).sum())))
    pd.DataFrame(srows).to_csv(HERE / "onset_risk_by_split_proxy.csv", index=False)

    # ---- disagreement context ------------------------------------------------------------
    cells = pd.read_parquet(HERE / "disagreement_cells.parquet")
    nums["disagreements"] = cells.groupby("direction").apply(lambda g: dict(
        n_total=len(g), per_seed=len(g) / 8, share_same_observation=float(g.same_observation.mean()),
        share_prev_upf_differs=float((g.prev_usr_mappo != g.prev_usr_shared).mean()),
        median_load_mbps=float(g.load_mbps.median()))).to_dict()
    (HERE / "supporting_numbers.json").write_text(json.dumps(nums, indent=1, default=str))
    print(json.dumps(nums, indent=1, default=str))


if __name__ == "__main__":
    main()
