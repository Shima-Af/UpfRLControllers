"""Where and why do MAPPO and Shared-PPO disagree? Analysis of EXISTING per-step records only.

Inputs (all saved on 2026-09-14; nothing is re-run):
  reports/wcnc2027-shared-policy-ablation/rollouts/{MAPPO,Shared-PPO}__seed<N>.npz
      float64 per-step (T=1009, K=10): action, realised UPF (usr), is_safe, q_score,
      reward, energy_term (alpha*SEC), qos_penalty, switch_penalty, cooldown_penalty,
      power_watts, power_watts_steady, switching_energy_wh, actual_load_gbps
  reports/wcnc2027-calibration-robustness/rollouts/main/pinned/A/<ctrl>__...seed<N>-mappo_best.pt.npz
      same checkpoints, same pinned twin, identical action arrays; adds delay_us, predicted_loss (float32)
  reports/wcnc2027-calibration-robustness/rollouts/main/pinned/A/Always-{DPDK,USR}__-.npz
      per-step outcomes of each UPF held for the whole episode = steady-state counterfactual of
      choosing that UPF at step t (exact except at t = 0 for Always-USR, the initial switch)
  reports/wcnc2027-calibration-robustness/rollouts/main/traffic_test.npz
      actual load and one-step forecast (Gbps) seen by both controllers

Observation reconstruction (single_site_upf_env._obs, sha256 f1feb81a...):
  [load_t, load_{t-8..t-1} (0 before start), forecast_t, prev realised UPF (1=USR), Q_{t-1} (1.0 at t=0),
   min(SEC_{t-1}, 1e3) (0 at t=0), min(1, steps_since_switch/4) (steps_since_switch starts at 4)]
  SEC_t = power_watts_steady_t / max(1000*load_t, 1e-3); checked against energy_term = 100*SEC.

Outputs: decomposition_by_seed.csv, disagreement_cells_summary.csv, load_bin_summary.csv,
         strata_summary.csv, cluster_summary.csv, usr_usage_by_risk.csv, stepwise_numbers.json,
         disagreement_cells.parquet (every disagreement cell, all 8 seeds)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ABL = ROOT / "reports" / "wcnc2027-shared-policy-ablation" / "rollouts"
CAL = ROOT / "reports" / "wcnc2027-calibration-robustness" / "rollouts" / "main"
SEEDS = [1, 7, 13, 23, 42, 64, 77, 99]
K, STEP_H, TAU, ALPHA, CD_PERIOD = 10, 0.25, 0.9, 100.0, 4
COSTS = ["energy_term", "qos_penalty", "switch_penalty", "cooldown_penalty"]
LOAD_BINS = [0, 10, 25, 50, 81, 149, 300, 600, np.inf]
LOAD_LABELS = ["<10", "10–25", "25–50", "50–81", "81–149", "149–300", "300–600", "≥600"]
PRE = {"MAPPO": "mappo", "Shared-PPO": "shared_ppo"}


def load_run(ctrl: str, seed: int) -> dict:
    a = np.load(ABL / f"{ctrl}__seed{seed}.npz", allow_pickle=True)
    b = np.load(CAL / "pinned" / "A" / f"{ctrl}__wcnc2027_ablation-{PRE[ctrl]}_seed{seed}-mappo_best.pt.npz")
    assert np.array_equal(a["action"], b["action"])
    d = {k: a[k].astype(np.float64) for k in ["reward", *COSTS, "power_watts", "power_watts_steady",
                                             "switching_energy_wh", "q_score", "actual_load_gbps"]}
    d.update(usr=a["usr"].astype(bool), action=a["action"].astype(int), is_safe=a["is_safe"].astype(bool),
             delay_us=b["delay_us"].astype(np.float64), predicted_loss=b["predicted_loss"].astype(np.float64))
    return d


def derived(d: dict, L: np.ndarray, F: np.ndarray) -> dict:
    T = L.shape[0]
    u = d["usr"]
    prev = np.vstack([np.zeros((1, K), bool), u[:-1]])
    sw = u != prev
    ss_before = np.zeros((T, K), int)
    ss = np.full(K, CD_PERIOD)
    for t in range(T):
        ss_before[t] = ss
        ss = np.where(sw[t], 0, ss + 1)
    sec = d["power_watts_steady"] / np.maximum(L * 1000.0, 1e-3)
    prevQ = np.vstack([np.ones((1, K)), d["q_score"][:-1]])
    prevSEC = np.vstack([np.zeros((1, K)), np.minimum(sec[:-1], 1e3)])
    cd = np.minimum(1.0, ss_before / CD_PERIOD)
    hist = np.zeros((T, K, 8))
    for j in range(1, 9):
        hist[j:, :, 8 - j] = L[:-j]
    obs = np.concatenate([L[..., None], hist, F[..., None], prev[..., None].astype(float),
                          prevQ[..., None], prevSEC[..., None], cd[..., None]], axis=-1).astype(np.float32)
    return dict(prev=prev, sw=sw, ss_before=ss_before, sec=sec, obs=obs,
                sec_check=float(np.max(np.abs(ALPHA * sec - d["energy_term"]) / np.maximum(1.0, d["energy_term"]))))


def main():
    tr = np.load(CAL / "traffic_test.npz")
    L, F = tr["load_gbps"].astype(np.float64), tr["forecast_gbps"].astype(np.float64)
    T = L.shape[0]
    Lm, Fm = 1000 * L, 1000 * F
    AD, AU = np.load(CAL / "pinned" / "A" / "Always-DPDK__-.npz"), np.load(CAL / "pinned" / "A" / "Always-USR__-.npz")
    cf = dict(
        usr_qpen=AU["qos_penalty"].astype(float), dpdk_qpen=AD["qos_penalty"].astype(float),
        usr_unsafe=~AU["is_safe"], dpdk_unsafe=~AD["is_safe"],
        usr_q=AU["q_score"].astype(float), dpdk_q=AD["q_score"].astype(float),
        d_sec_term=(AU["energy_term"] - AD["energy_term"]).astype(float),               # USR - DPDK
        d_wh_steady=((AU["power_watts_steady"] - AD["power_watts_steady"]) * STEP_H).astype(float),
        usr_delay=AU["delay_us"].astype(float), usr_loss=AU["predicted_loss"].astype(float))
    first_step = np.zeros((T, K), bool)
    first_step[0] = True
    # counterfactual USR risk class of each cluster-step (steady state)
    extra_qpen = cf["usr_qpen"] - cf["dpdk_qpen"]
    risk = np.where(first_step, "first step",
                    np.where(extra_qpen > 0, "USR QoS-risky",
                             np.where(cf["d_sec_term"] < 0, "USR safe saving", "USR no saving")))
    bins = pd.cut(Lm.ravel(), LOAD_BINS, right=False, labels=LOAD_LABELS).to_numpy().astype(str).reshape(T, K)
    f_cross81 = (Fm < 81) != (Lm < 81)
    f_cross149 = (Fm < 149) != (Lm < 149)
    prevL = np.vstack([Lm[:1], Lm[:-1]])
    trans81 = (prevL < 81) != (Lm < 81)
    trans149 = (prevL < 149) != (Lm < 149)
    rel_fe = np.abs(Fm - Lm) / np.maximum(Lm, 1.0)
    fe_bin = pd.cut(rel_fe.ravel(), [0, 0.1, 0.25, 0.5, np.inf], right=False,
                    labels=["<10%", "10–25%", "25–50%", "≥50%"]).to_numpy().astype(str).reshape(T, K)

    dec_rows, cell_frames, strata_rows, lb_rows, cl_rows, risk_rows, switch_rows = [], [], [], [], [], [], []
    checks = {}
    for s in SEEDS:
        M, S = load_run("MAPPO", s), load_run("Shared-PPO", s)
        assert np.allclose(M["actual_load_gbps"], L) and np.allclose(S["actual_load_gbps"], L)
        dM, dS = derived(M, L, F), derived(S, L, F)
        checks[s] = dict(sec_recon_max_rel_err=max(dM["sec_check"], dS["sec_check"]))
        uM, uS = M["usr"], S["usr"]
        same_state = (dM["prev"] == dS["prev"]) & (dM["ss_before"] == dS["ss_before"])
        same_obs = np.all(dM["obs"] == dS["obs"], axis=-1)
        cat = np.where(uM == uS, np.where(same_state, "agree, same state", "agree, different prev/cooldown"),
                       np.where(uS, "Shared-PPO USR / MAPPO DPDK", "MAPPO USR / Shared-PPO DPDK"))
        delta = {c: S[c] - M[c] for c in ["reward", *COSTS]}
        delta["wh_total"] = (S["power_watts"] - M["power_watts"]) * STEP_H
        delta["wh_steady"] = (S["power_watts_steady"] - M["power_watts_steady"]) * STEP_H
        delta["viol"] = (~S["is_safe"]).astype(int) - (~M["is_safe"]).astype(int)
        delta["q_below_tau"] = (S["q_score"] < TAU).astype(int) - (M["q_score"] < TAU).astype(int)
        agree_same = cat == "agree, same state"
        # finer label for agreeing cells whose state differs: who switches at t, and in which direction
        swS_only = dS["sw"] & ~dM["sw"]
        swM_only = dM["sw"] & ~dS["sw"]
        sub_agree = np.where(swS_only & ~uS, "Shared-PPO exits USR, MAPPO already in DPDK",
                    np.where(swS_only & uS, "Shared-PPO enters USR, MAPPO already in USR",
                    np.where(swM_only & ~uM, "MAPPO exits USR, Shared-PPO already in DPDK",
                    np.where(swM_only & uM, "MAPPO enters USR, Shared-PPO already in USR",
                    np.where(dM["sw"] & dS["sw"], "both switch", "no switch, cooldown counter differs")))))
        checks[s]["max_abs_delta_reward_when_agree_same_state"] = float(np.abs(delta["reward"][agree_same]).max())

        # static counterfactual part of each disagreement: sign * (USR - DPDK) steady-state difference
        sign = np.where(cat == "Shared-PPO USR / MAPPO DPDK", 1.0, np.where(cat == "MAPPO USR / Shared-PPO DPDK", -1.0, 0.0))
        static = dict(energy_term=sign * cf["d_sec_term"], qos_penalty=sign * extra_qpen, wh_steady=sign * cf["d_wh_steady"])

        # ---- fleet decomposition by category and USR-risk class ----
        sub = np.where(sign != 0, risk, np.where(agree_same, "-", sub_agree))
        risk_at_t = risk
        for cname in np.unique(cat):
            for rname in np.unique(sub[cat == cname]):
                m = (cat == cname) & (sub == rname)
                row = dict(seed=s, category=cname, usr_risk_class=rname, n_cells=int(m.sum()),
                           n_cf_usr_qos_risky_at_t=int((m & (risk == "USR QoS-risky")).sum()),
                           n_same_observation=int((m & same_obs).sum()),
                           n_switch_step_either=int((m & (dM["sw"] | dS["sw"])).sum()),
                           n_prev_upf_differs=int((m & (dM["prev"] != dS["prev"])).sum()))
                for c, v in delta.items():
                    row[f"delta_{c}"] = float(v[m].sum())
                for c, v in static.items():
                    row[f"static_{c}"] = float(v[m].sum())
                dec_rows.append(row)

        # ---- every disagreement cell ----
        dis = sign != 0
        tt, kk = np.nonzero(dis)
        cell_frames.append(pd.DataFrame(dict(
            seed=s, t=tt, cluster=kk, direction=cat[dis], load_mbps=Lm[dis], forecast_mbps=Fm[dis],
            load_bin=bins[dis], usr_risk_class=risk[dis], forecast_rel_err=rel_fe[dis],
            forecast_crosses_81=f_cross81[dis], forecast_crosses_149=f_cross149[dis],
            load_crosses_81=trans81[dis], load_crosses_149=trans149[dis],
            prev_usr_mappo=dM["prev"][dis], prev_usr_shared=dS["prev"][dis],
            steps_since_switch_mappo=dM["ss_before"][dis], steps_since_switch_shared=dS["ss_before"][dis],
            switch_step_mappo=dM["sw"][dis], switch_step_shared=dS["sw"][dis], same_observation=same_obs[dis],
            q_mappo=M["q_score"][dis], q_shared=S["q_score"][dis], safe_mappo=M["is_safe"][dis], safe_shared=S["is_safe"][dis],
            delay_mappo=M["delay_us"][dis], delay_shared=S["delay_us"][dis],
            loss_mappo=M["predicted_loss"][dis], loss_shared=S["predicted_loss"][dis],
            cf_usr_q=cf["usr_q"][dis], cf_dpdk_q=cf["dpdk_q"][dis], cf_extra_qos_penalty_usr=extra_qpen[dis],
            cf_delta_sec_term_usr_minus_dpdk=cf["d_sec_term"][dis], cf_delta_wh_steady_usr_minus_dpdk=cf["d_wh_steady"][dis],
            **{f"delta_{c}": v[dis] for c, v in delta.items()})))

        # ---- strata: disagreement rates ----
        strata = {
            "all cluster-steps": np.ones((T, K), bool),
            "forecast and actual on opposite sides of 81 Mbps": f_cross81,
            "forecast and actual on opposite sides of 149 Mbps": f_cross149,
            "load crossed 81 Mbps since previous step": trans81,
            "load crossed 149 Mbps since previous step": trans149,
            **{f"relative forecast error {b}": fe_bin == b for b in ["<10%", "10–25%", "25–50%", "≥50%"]},
            **{f"counterfactual: {r}": risk == r for r in ["USR safe saving", "USR QoS-risky", "USR no saving"]},
            "same previous UPF and cooldown state": same_state,
            "different previous UPF or cooldown state": ~same_state,
        }
        for name, m in strata.items():
            n = int(m.sum())
            strata_rows.append(dict(seed=s, stratum=name, n_cells=n,
                                    rate_shared_usr_mappo_dpdk=float((m & (cat == "Shared-PPO USR / MAPPO DPDK")).sum() / max(n, 1)),
                                    rate_mappo_usr_shared_dpdk=float((m & (cat == "MAPPO USR / Shared-PPO DPDK")).sum() / max(n, 1)),
                                    delta_qos_penalty=float(delta["qos_penalty"][m].sum()),
                                    delta_energy_term=float(delta["energy_term"][m].sum())))

        # ---- load bins ----
        for b in LOAD_LABELS:
            m = bins == b
            lb_rows.append(dict(seed=s, load_bin=b, n_cells=int(m.sum()),
                                usr_share_mappo=float(uM[m].mean()) if m.any() else np.nan,
                                usr_share_shared=float(uS[m].mean()) if m.any() else np.nan,
                                n_shared_usr_mappo_dpdk=int((m & (cat == "Shared-PPO USR / MAPPO DPDK")).sum()),
                                n_mappo_usr_shared_dpdk=int((m & (cat == "MAPPO USR / Shared-PPO DPDK")).sum()),
                                cf_usr_qos_risky_share=float((risk[m] == "USR QoS-risky").mean()) if m.any() else np.nan,
                                **{f"delta_{c}": float(delta[c][m].sum()) for c in ["reward", *COSTS, "wh_total", "viol"]}))

        # ---- clusters ----
        for k in range(K):
            cl_rows.append(dict(seed=s, cluster=k, median_load_mbps=float(np.median(Lm[:, k])),
                                usr_share_mappo=float(uM[:, k].mean()), usr_share_shared=float(uS[:, k].mean()),
                                n_disagreements=int((sign[:, k] != 0).sum()),
                                **{f"delta_{c}": float(delta[c][:, k].sum()) for c in ["reward", *COSTS, "wh_total", "viol"]}))

        # ---- every realised switching step of each controller ----
        for ctrl, D, dd, other, do in (("MAPPO", M, dM, S, dS), ("Shared-PPO", S, dS, M, dM)):
            tt, kk = np.nonzero(dd["sw"])
            for t, k in zip(tt, kk):
                if t == 0:
                    continue
                exit_ = not D["usr"][t, k]
                # how many consecutive steps before t the OTHER controller was already in the new UPF
                j, lead = t - 1, 0
                while j >= 0 and other["usr"][j, k] == D["usr"][t, k]:
                    lead += 1
                    j -= 1
                switch_rows.append(dict(
                    seed=s, controller=ctrl, t=int(t), cluster=int(k), direction="USR->DPDK (exit)" if exit_ else "DPDK->USR (entry)",
                    load_mbps=float(Lm[t, k]), load_prev_mbps=float(Lm[t - 1, k]), forecast_mbps=float(Fm[t, k]),
                    cf_class_at_t=str(risk[t, k]), cf_usr_q_at_t=float(cf["usr_q"][t, k]),
                    cf_usr_loss_at_t=float(cf["usr_loss"][t, k]), cf_usr_delay_at_t=float(cf["usr_delay"][t, k]),
                    cf_class_at_t_minus_1=str(risk[t - 1, k]),
                    qos_penalty=float(D["qos_penalty"][t, k]), q_score=float(D["q_score"][t, k]),
                    is_safe=bool(D["is_safe"][t, k]), predicted_loss=float(D["predicted_loss"][t, k]),
                    delay_us=float(D["delay_us"][t, k]), energy_term=float(D["energy_term"][t, k]),
                    switch_penalty=float(D["switch_penalty"][t, k]), cooldown_penalty=float(D["cooldown_penalty"][t, k]),
                    other_controller_same_upf_at_t=bool(other["usr"][t, k] == D["usr"][t, k]),
                    other_switches_at_t=bool(do["sw"][t, k]),
                    other_already_in_new_upf_steps=int(lead),
                    other_qos_penalty_at_t=float(other["qos_penalty"][t, k])))

        # ---- each controller's USR use by counterfactual risk class ----
        for ctrl, D, dd in (("MAPPO", M, dM), ("Shared-PPO", S, dS)):
            for r in ["USR safe saving", "USR QoS-risky", "USR no saving", "first step"]:
                m = risk == r
                um = D["usr"] & m
                risk_rows.append(dict(seed=s, controller=ctrl, usr_risk_class=r, n_cells_in_class=int(m.sum()),
                                      usr_steps=int(um.sum()), usr_share_in_class=float(um.sum() / max(m.sum(), 1)),
                                      qos_penalty_on_usr_steps=float(D["qos_penalty"][um].sum()),
                                      energy_term_on_usr_steps=float(D["energy_term"][um].sum()),
                                      switch_steps_into_usr=int((dd["sw"] & D["usr"] & m).sum()),
                                      switch_steps_out_of_usr=int((dd["sw"] & ~D["usr"] & m).sum())))

    sw_df = pd.DataFrame(switch_rows)
    sw_df.to_csv(HERE / "switch_steps.csv", index=False)
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(HERE / "decomposition_by_seed.csv", index=False)
    cells = pd.concat(cell_frames, ignore_index=True)
    cells.to_parquet(HERE / "disagreement_cells.parquet", index=False)
    strata = pd.DataFrame(strata_rows)
    strata.to_csv(HERE / "strata_summary.csv", index=False)
    lb = pd.DataFrame(lb_rows)
    lb.to_csv(HERE / "load_bin_summary.csv", index=False)
    cl = pd.DataFrame(cl_rows)
    cl.to_csv(HERE / "cluster_summary.csv", index=False)
    rk = pd.DataFrame(risk_rows)
    rk.to_csv(HERE / "usr_usage_by_risk.csv", index=False)

    # disagreement-cell summary (per seed x direction x risk class) + pooled
    g = cells.groupby(["seed", "direction", "usr_risk_class"])
    dcs = g.agg(n_cells=("t", "size"), share_same_observation=("same_observation", "mean"),
                share_prev_upf_differs=("prev_usr_mappo", lambda x: np.nan),
                median_load_mbps=("load_mbps", "median"),
                share_forecast_crosses_81=("forecast_crosses_81", "mean"),
                share_forecast_crosses_149=("forecast_crosses_149", "mean"),
                share_switch_step_mappo=("switch_step_mappo", "mean"),
                share_switch_step_shared=("switch_step_shared", "mean"),
                delta_qos_penalty=("delta_qos_penalty", "sum"), delta_energy_term=("delta_energy_term", "sum"),
                delta_switch_penalty=("delta_switch_penalty", "sum"), delta_cooldown_penalty=("delta_cooldown_penalty", "sum"),
                delta_reward=("delta_reward", "sum"), delta_wh_total=("delta_wh_total", "sum"),
                delta_viol=("delta_viol", "sum")).reset_index()
    prevdiff = cells.assign(pd_=cells.prev_usr_mappo != cells.prev_usr_shared).groupby(
        ["seed", "direction", "usr_risk_class"]).pd_.mean().reset_index(drop=True)
    dcs["share_prev_upf_differs"] = prevdiff.values
    dcs.to_csv(HERE / "disagreement_cells_summary.csv", index=False)

    # ---- headline numbers -------------------------------------------------------------
    tot = dec.groupby("seed")[[c for c in dec.columns if c.startswith("delta_")]].sum()
    by_cat = dec.groupby(["seed", "category"])[["delta_reward", "delta_energy_term", "delta_qos_penalty",
                                                "delta_switch_penalty", "delta_cooldown_penalty", "delta_wh_total", "n_cells"]].sum()
    by_risk = dec[dec.category.str.contains("/")].groupby(["seed", "category", "usr_risk_class"])[
        ["delta_reward", "delta_energy_term", "delta_qos_penalty", "delta_switch_penalty", "delta_cooldown_penalty",
         "static_energy_term", "static_qos_penalty", "delta_wh_total", "delta_wh_steady", "n_cells",
         "n_same_observation", "n_switch_step_either", "n_prev_upf_differs"]].sum()
    nums = dict(
        checks=checks,
        totals_per_seed=tot.round(3).to_dict("index"),
        totals_mean_over_seeds=tot.mean().round(3).to_dict(),
        by_category_mean_over_seeds=by_cat.groupby("category").mean().round(3).to_dict("index"),
        by_category_and_risk_mean_over_seeds=by_risk.groupby(["category", "usr_risk_class"]).mean().round(3).reset_index().to_dict("records"),
        by_category_and_risk_n_seeds_positive_delta_qos={" | ".join(k): int(v) for k, v in by_risk.assign(
            pos=lambda x: x.delta_qos_penalty > 0).groupby(["category", "usr_risk_class"]).pos.sum().items()},
        disagreement_cells_total=int(len(cells)), disagreement_cells_per_seed=cells.groupby("seed").size().to_dict(),
        share_disagreements_same_observation=float(cells.same_observation.mean()),
        share_disagreements_prev_upf_differs=float((cells.prev_usr_mappo != cells.prev_usr_shared).mean()),
    )
    (HERE / "stepwise_numbers.json").write_text(json.dumps(nums, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(json.dumps({k: v for k, v in nums.items() if k not in ("totals_per_seed",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
