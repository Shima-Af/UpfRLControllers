"""Phase 9 — Fleet-level evaluation with telecom KPIs.

Runs MAPPO (one or more checkpoints) plus the operational baselines
(per-site hysteresis, always-DPDK, always-USR) on the test split,
across all K clusters, and reports:
  - per-cluster KPIs (energy, delay-budget compliance, switches, ...)
  - fleet-level aggregates
  - energy savings vs hysteresis and vs always-DPDK
in telecom-grade units (Wh, %, μs).

Usage
-----
    python research/phase9/fleet_eval.py
        --mappo-ckpt experiments/mappo_budget_off_seed42_<ts>/mappo_best.pt
        --mappo-ckpt experiments/mappo_budget_off_seed1_<ts>/mappo_best.pt
        --mappo-ckpt experiments/mappo_budget_off_seed77_<ts>/mappo_best.pt

If --mappo-ckpt is repeated, MAPPO results are reported as mean ± std
across the supplied seeds.
"""
from __future__ import annotations

import argparse
import copy
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.envs.multi_agent_upf_env import MultiAgentUPFEnv  # noqa: E402
from src.trainers.mappo import Actor  # noqa: E402
from src.baselines.hysteresis import MultiAgentHysteresis  # noqa: E402
from src.baselines.threshold_derivation import (  # noqa: E402
    derive_thresholds, load_forecast_mae_gbps,
)
from src.utils.config import load_yaml  # noqa: E402


def _load_scenario_cfg() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "scenario_rl.yaml")


def _load_paths_cfg() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "digital_twin_paths.yaml")


def _load_actor(ckpt_path: Path) -> Actor:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    actor = Actor(ckpt["obs_dim"], ckpt["n_actions"], cfg.get("actor_hidden", 64))
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


@torch.no_grad()
def _mappo_policy_fn(actor: Actor, agents: list[str]):
    def _fn(obs_dict):
        obs = np.stack([obs_dict[a] for a in agents]).astype(np.float32)
        action, _ = actor.act(torch.from_numpy(obs), deterministic=True)
        return {a: int(action[k].item()) for k, a in enumerate(agents)}
    return _fn


def _const_policy_fn(action_id: int):
    def _fn(obs_dict):
        return {a: int(action_id) for a in obs_dict.keys()}
    return _fn


def _derive_hysteresis_thresholds_gbps(
    scenario_cfg: dict, paths_cfg: dict
) -> tuple[float, float]:
    """Derive (t_up, t_down) in Gbps from twin profiling models."""
    from upf_digital_twin import DigitalTwin
    twin = DigitalTwin(
        scenario_cfg=scenario_cfg, paths_cfg=paths_cfg, project_root=REPO_ROOT,
    )
    tf = paths_cfg.get("traffic_forecaster", {})
    alpha = float(
        tf.get("alpha")
        or scenario_cfg.get("traffic", {}).get("alpha")
        or 1.0
    )
    summary_rel = tf.get(
        "forecast_eval_summary",
        "data/external/traffic_forecaster/forecast_eval_summary.json",
    )
    mae_gbps = load_forecast_mae_gbps(
        REPO_ROOT / summary_rel, alpha_gbps_per_norm=alpha, K=10,
    )
    spec = derive_thresholds(twin, safety_margin_mbps=10.0, forecast_mae_gbps=mae_gbps)
    return float(spec.t_up_gbps), float(spec.t_down_gbps)


def _rollout(env: MultiAgentUPFEnv, policy_fn, seed: int) -> dict:
    """Run one full episode under the policy_fn; return per-step traces."""
    obs, _ = env.reset(seed=seed)
    agents = list(env.possible_agents)
    K = env.K
    step_h = float(env.step_h)
    tau = float(env.tau)

    # Per-step arrays.
    actions_t: list[list[int]] = []
    power_w_t: list[float] = []
    energy_wh_t: list[float] = []
    delay_us_kt: list[list[float]] = []
    loss_pkts_kt: list[list[float]] = []
    q_kt: list[list[float]] = []
    safe_kt: list[list[int]] = []
    selected_upf_kt: list[list[str]] = []
    last_act = [None] * K
    n_switches_per_cluster = [0] * K
    steps = 0

    while env.agents:
        action = policy_fn(obs)
        obs, _r, _t, _u, info = env.step(action)
        actions_t.append([int(action[a]) for a in agents])
        delay_us_kt.append([float(info[a]["delay_us"]) for a in agents])
        loss_pkts_kt.append([float(info[a]["predicted_loss"]) for a in agents])
        q_kt.append([float(info[a]["q_score"]) for a in agents])
        safe_kt.append([int(info[a]["is_safe"]) for a in agents])
        selected_upf_kt.append([str(info[a]["selected_upf"]) for a in agents])
        power_w_t.append(sum(float(info[a]["power_watts_steady"]) for a in agents))
        energy_wh_t.append(power_w_t[-1] * step_h)
        for k, a in enumerate(agents):
            ak = int(action[a])
            if last_act[k] is not None and last_act[k] != ak:
                n_switches_per_cluster[k] += 1
            last_act[k] = ak
        steps += 1

    return {
        "K": K, "steps": steps, "step_h": step_h, "tau": tau,
        "actions": np.asarray(actions_t, dtype=np.int8),  # (T, K)
        "delay_us": np.asarray(delay_us_kt, dtype=np.float64),
        "loss_pkts": np.asarray(loss_pkts_kt, dtype=np.float64),
        "q_score": np.asarray(q_kt, dtype=np.float64),
        "is_safe": np.asarray(safe_kt, dtype=np.int8),
        "selected_upf": np.asarray(selected_upf_kt, dtype=object),
        "fleet_power_w": np.asarray(power_w_t, dtype=np.float64),
        "fleet_step_energy_wh": np.asarray(energy_wh_t, dtype=np.float64),
        "n_switches_per_cluster": np.asarray(n_switches_per_cluster, dtype=np.int64),
    }


def _compute_kpis(roll: dict, scenario_cfg: dict) -> dict:
    """Compute telecom-grade KPIs from a rollout dict."""
    K = roll["K"]
    T = roll["steps"]
    step_h = roll["step_h"]

    qos_cfg = scenario_cfg.get("upf", {}).get("qos_budget", {})
    delay_budget_us = float(qos_cfg.get("delay_budget_us", 200.0))
    loss_budget = float(qos_cfg.get("max_loss_pkts_per_interval", 5.0))

    total_energy_wh = float(roll["fleet_step_energy_wh"].sum())
    mean_fleet_power_w = float(roll["fleet_power_w"].mean())

    # Delay budget compliance: % of (cluster, step) pairs where delay <= budget.
    delay = roll["delay_us"]
    loss = roll["loss_pkts"]
    delay_compliance = float((delay <= delay_budget_us).mean())
    loss_compliance = float((loss <= loss_budget).mean())
    joint_compliance = float(
        ((delay <= delay_budget_us) & (loss <= loss_budget)).mean()
    )
    p95_delay_us = float(np.percentile(delay, 95))
    max_delay_us = float(delay.max())
    mean_delay_us = float(delay.mean())

    # Mode usage.
    sel = roll["selected_upf"]
    dpdk_rate = float((sel == "DPDK").mean())
    usr_rate = 1.0 - dpdk_rate

    # Switching.
    n_switches_total = int(roll["n_switches_per_cluster"].sum())
    episode_days = T * step_h / 24.0
    switches_per_cluster_per_day = float(
        roll["n_switches_per_cluster"].mean() / max(episode_days, 1e-9)
    )

    return {
        "total_energy_wh": total_energy_wh,
        "mean_fleet_power_w": mean_fleet_power_w,
        "p95_fleet_power_w": float(np.percentile(roll["fleet_power_w"], 95)),
        "delay_compliance_pct": 100.0 * delay_compliance,
        "loss_compliance_pct": 100.0 * loss_compliance,
        "joint_qos_compliance_pct": 100.0 * joint_compliance,
        "mean_delay_us": mean_delay_us,
        "p95_delay_us": p95_delay_us,
        "max_delay_us": max_delay_us,
        "dpdk_rate_pct": 100.0 * dpdk_rate,
        "usr_rate_pct": 100.0 * usr_rate,
        "n_switches_total": n_switches_total,
        "switches_per_cluster_per_day": switches_per_cluster_per_day,
        "episode_days": float(episode_days),
        "K": K,
        "steps": T,
    }


def _per_cluster_kpis(roll: dict, scenario_cfg: dict) -> pd.DataFrame:
    """KPIs broken down per cluster."""
    K = roll["K"]
    step_h = roll["step_h"]
    qos_cfg = scenario_cfg.get("upf", {}).get("qos_budget", {})
    delay_budget_us = float(qos_cfg.get("delay_budget_us", 200.0))
    loss_budget = float(qos_cfg.get("max_loss_pkts_per_interval", 5.0))

    rows = []
    for k in range(K):
        delay_k = roll["delay_us"][:, k]
        loss_k = roll["loss_pkts"][:, k]
        sel_k = roll["selected_upf"][:, k]
        rows.append({
            "cluster": k,
            "delay_compliance_pct": 100.0 * float((delay_k <= delay_budget_us).mean()),
            "loss_compliance_pct": 100.0 * float((loss_k <= loss_budget).mean()),
            "mean_delay_us": float(delay_k.mean()),
            "p95_delay_us": float(np.percentile(delay_k, 95)),
            "dpdk_rate_pct": 100.0 * float((sel_k == "DPDK").mean()),
            "n_switches": int(roll["n_switches_per_cluster"][k]),
        })
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--mappo-ckpt", type=Path, action="append", default=[],
        help="MAPPO checkpoint(s). Repeatable for multi-seed.",
    )
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for env reset only.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--cluster-indices", type=str, default=None,
                   help="Comma-separated subset (e.g. '8,9'). Default: all.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    scenario_cfg = _load_scenario_cfg()
    paths_cfg = _load_paths_cfg()

    cluster_indices = None
    if args.cluster_indices is not None:
        cluster_indices = [int(x) for x in args.cluster_indices.split(",") if x.strip()]

    # Derive hysteresis thresholds once from the twin.
    t_up_gbps, t_down_gbps = _derive_hysteresis_thresholds_gbps(scenario_cfg, paths_cfg)
    print(f"Hysteresis thresholds: t_up={t_up_gbps*1000:.2f} Mbps, "
          f"t_down={t_down_gbps*1000:.2f} Mbps")

    # Build envs (fresh per policy to avoid state leak).
    def _make_env() -> MultiAgentUPFEnv:
        return MultiAgentUPFEnv(
            split=args.split,
            scenario_cfg=copy.deepcopy(scenario_cfg),
            cluster_indices=cluster_indices,
        )

    # ----- Run policies -----
    results: list[dict] = []
    per_cluster_frames: dict[str, pd.DataFrame] = {}

    # Always-DPDK
    env = _make_env()
    roll = _rollout(env, _const_policy_fn(0), args.seed)
    kpi = _compute_kpis(roll, scenario_cfg)
    kpi["policy"] = "always-DPDK"
    results.append(kpi)
    per_cluster_frames["always-DPDK"] = _per_cluster_kpis(roll, scenario_cfg)
    print(f"  [done] always-DPDK  energy={kpi['total_energy_wh']:.1f}Wh  "
          f"delay-OK={kpi['delay_compliance_pct']:.2f}%")

    # Always-USR
    env = _make_env()
    roll = _rollout(env, _const_policy_fn(1), args.seed)
    kpi = _compute_kpis(roll, scenario_cfg)
    kpi["policy"] = "always-USR"
    results.append(kpi)
    per_cluster_frames["always-USR"] = _per_cluster_kpis(roll, scenario_cfg)
    print(f"  [done] always-USR   energy={kpi['total_energy_wh']:.1f}Wh  "
          f"delay-OK={kpi['delay_compliance_pct']:.2f}%")

    # Per-site hysteresis variants.
    # (1) Twin-derived (conservative; matches phase-7 pipeline).
    env = _make_env()
    hyst = MultiAgentHysteresis(
        agents=list(env.possible_agents),
        t_up_gbps=t_up_gbps,
        t_down_gbps=t_down_gbps,
        cooldown_steps=1,
    )
    hyst.reset()

    def _hyst_fn(obs_dict):
        return hyst(obs_dict)

    roll = _rollout(env, _hyst_fn, args.seed)
    kpi = _compute_kpis(roll, scenario_cfg)
    kpi["policy"] = "hysteresis-derived"
    results.append(kpi)
    per_cluster_frames["hysteresis-derived"] = _per_cluster_kpis(roll, scenario_cfg)
    print(f"  [done] hysteresis-derived  energy={kpi['total_energy_wh']:.1f}Wh  "
          f"delay-OK={kpi['delay_compliance_pct']:.2f}%")

    # (2) Operator-tuned (aggressive; switches to USR at low load to save energy).
    # 100 Mbps switching threshold, +/- 10 Mbps dead-band.
    env = _make_env()
    hyst2 = MultiAgentHysteresis(
        agents=list(env.possible_agents),
        t_up_gbps=0.110, t_down_gbps=0.090,
        cooldown_steps=4,  # match MAPPO's cooldown_period for fair comparison
    )
    hyst2.reset()

    def _hyst2_fn(obs_dict):
        return hyst2(obs_dict)

    roll = _rollout(env, _hyst2_fn, args.seed)
    kpi = _compute_kpis(roll, scenario_cfg)
    kpi["policy"] = "hysteresis-tuned"
    results.append(kpi)
    per_cluster_frames["hysteresis-tuned"] = _per_cluster_kpis(roll, scenario_cfg)
    print(f"  [done] hysteresis-tuned    energy={kpi['total_energy_wh']:.1f}Wh  "
          f"delay-OK={kpi['delay_compliance_pct']:.2f}%")

    # MAPPO (one row per seed; reported as mean ± std at the end).
    mappo_per_cluster_list: list[pd.DataFrame] = []
    for ckpt_path in args.mappo_ckpt:
        env = _make_env()
        actor = _load_actor(ckpt_path)
        roll = _rollout(env, _mappo_policy_fn(actor, list(env.possible_agents)), args.seed)
        kpi = _compute_kpis(roll, scenario_cfg)
        kpi["policy"] = "MAPPO"
        try:
            kpi["ckpt"] = str(ckpt_path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            kpi["ckpt"] = str(ckpt_path)
        results.append(kpi)
        mappo_per_cluster_list.append(_per_cluster_kpis(roll, scenario_cfg))
        print(f"  [done] MAPPO ({ckpt_path.parent.name})  "
              f"energy={kpi['total_energy_wh']:.1f}Wh  "
              f"delay-OK={kpi['delay_compliance_pct']:.2f}%")

    if mappo_per_cluster_list:
        # Stack and average per-cluster MAPPO KPIs across seeds.
        mappo_concat = pd.concat(mappo_per_cluster_list, ignore_index=True)
        per_cluster_frames["MAPPO"] = mappo_concat.groupby("cluster").mean().reset_index()

    df = pd.DataFrame(results)

    # ----- Aggregate MAPPO across seeds -----
    if "ckpt" in df.columns:
        mappo_rows = df[df.policy == "MAPPO"]
        if len(mappo_rows) > 1:
            metric_cols = [c for c in mappo_rows.columns
                           if c not in ("policy", "ckpt", "K", "steps")]
            mappo_agg = mappo_rows[metric_cols].agg(["mean", "std"]).T
            print()
            print(f"=== MAPPO aggregate over {len(mappo_rows)} seeds ===")
            print(mappo_agg.round(3).to_string())

    # ----- Headline summary table -----
    out_dir = args.out_dir or (REPO_ROOT / "research" / "phase9" / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"fleet_eval_{args.split}_long.csv", index=False)

    # Build a clean headline table: one row per policy, mean across seeds for MAPPO.
    headline_cols = [
        "policy", "total_energy_wh", "mean_fleet_power_w",
        "joint_qos_compliance_pct", "delay_compliance_pct", "loss_compliance_pct",
        "p95_delay_us", "switches_per_cluster_per_day",
        "dpdk_rate_pct",
    ]
    if "ckpt" in df.columns:
        agg = df.groupby("policy")[headline_cols[1:]].mean().reset_index()
    else:
        agg = df[headline_cols]

    # Add savings vs always-DPDK and vs hysteresis.
    base_dpdk = float(agg[agg.policy == "always-DPDK"].total_energy_wh.iloc[0])
    # Use the better of the two hysteresis variants as the reference.
    hyst_rows = agg[agg.policy.str.startswith("hysteresis")]
    base_hyst = float(hyst_rows.total_energy_wh.min()) if len(hyst_rows) else base_dpdk
    agg["energy_savings_vs_dpdk_pct"] = 100.0 * (base_dpdk - agg["total_energy_wh"]) / base_dpdk
    agg["energy_savings_vs_hysteresis_pct"] = 100.0 * (base_hyst - agg["total_energy_wh"]) / base_hyst

    headline_path = out_dir / f"fleet_eval_{args.split}_headline.csv"
    agg.to_csv(headline_path, index=False)

    print()
    print("=== Fleet-level headline (mean across seeds for MAPPO) ===")
    print(agg.round(3).to_string(index=False))
    print()
    print(f"Long CSV:     {out_dir / f'fleet_eval_{args.split}_long.csv'}")
    print(f"Headline CSV: {headline_path}")

    # ----- Per-cluster breakdown -----
    pc_rows = []
    for policy, frame in per_cluster_frames.items():
        frame_c = frame.copy()
        frame_c["policy"] = policy
        pc_rows.append(frame_c)
    if pc_rows:
        per_cluster = pd.concat(pc_rows, ignore_index=True)
        pc_path = out_dir / f"fleet_eval_{args.split}_per_cluster.csv"
        per_cluster.to_csv(pc_path, index=False)
        print(f"Per-cluster:  {pc_path}")

    # ----- Headline figure: Pareto (energy vs QoS) -----
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0), dpi=130)
    for _, row in agg.iterrows():
        marker = {"always-DPDK": "s", "always-USR": "^",
                  "hysteresis": "o", "MAPPO": "*"}.get(row.policy, "o")
        size = 200 if row.policy == "MAPPO" else 90
        ax.scatter(row.total_energy_wh, row.joint_qos_compliance_pct,
                   marker=marker, s=size, label=row.policy, edgecolor="black")
    ax.set_xlabel("Total fleet energy (Wh)")
    ax.set_ylabel("Joint QoS compliance (%)")
    ax.set_title(f"Fleet-level Pareto: energy vs QoS  ({args.split} split)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig_path = out_dir / f"fleet_eval_{args.split}_pareto.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Figure:       {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
