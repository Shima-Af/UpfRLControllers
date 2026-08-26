#!/usr/bin/env python3
"""Train + evaluate ONE sweep cell, then write a result JSON.

A cell is (line, alpha, K, arch, seed). Idempotent: if the result JSON
already exists the cell is skipped, so the driver is restart-safe.

Evaluation replicates `mappo.rollout_mappo_episode`'s accounting exactly
(same agg_unsafe_rate denominator, same energy integration) but builds the
env from an explicit scenario_cfg, which the upstream helper cannot do --
it constructs its own default env and would silently ignore alpha.
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np, torch
from src.envs.multi_agent_upf_env import MultiAgentUPFEnv
from src.trainers.mappo import Actor, MAPPOConfig, MAPPO
from src.baselines.hysteresis import MultiAgentHysteresis
from src.baselines.threshold_derivation import derive_thresholds, load_forecast_mae_gbps
from src.utils.config import load_yaml


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def build_cfg(alpha: float, K: int) -> dict:
    cfg = load_yaml(REPO / "configs" / "scenario_rl.yaml")
    return deep_merge(cfg, {"traffic": {"alpha": float(alpha), "selected_k": int(K)}})


def build_paths(K: int) -> dict:
    """Point the forecaster paths at the staged K-specific arrays.

    MultiAgentUPFEnv takes K from targets.shape[2], NOT from
    traffic.selected_k, so without this every cell of a K sweep would load
    the K=10 arrays and silently mislabel itself. Run stage_k.py first.
    """
    paths = load_yaml(REPO / "configs" / "digital_twin_paths.yaml")
    d = REPO / "data" / "external" / f"traffic_forecaster_K{K}"
    if not d.is_dir():
        raise SystemExit(f"K={K} not staged: {d} missing. Run stage_k.py {K}.")
    rel = f"data/external/traffic_forecaster_K{K}"
    tf = dict(paths.get("traffic_forecaster", {}))
    tf["dir"] = rel
    for split in ("train", "val", "test"):
        tf[f"predictions_{split}"] = f"{rel}/predictions_{split}.npy"
        tf[f"targets_{split}"] = f"{rel}/targets_{split}.npy"
    tf["predictions"] = f"{rel}/predictions_test.npy"
    tf["targets"] = f"{rel}/targets_test.npy"
    tf["forecast_eval_summary"] = f"{rel}/forecast_eval_summary.json"
    for name in ("cluster_series.npy", "cluster_assignments.parquet",
                 "cluster_bs_map.json", "bs_locations.parquet"):
        key = name.split(".")[0]
        if key in tf:
            tf[key] = f"{rel}/{name}"
    paths["traffic_forecaster"] = tf
    return paths


def rollout(env, act_fn) -> dict:
    """Deterministic full-episode rollout. Mirrors rollout_mappo_episode."""
    obs_dict, _ = env.reset(seed=0)
    K, step_h, tau = env.K, env.step_h, env.tau
    r = np.zeros(K); e = np.zeros(K)
    unsafe = np.zeros(K, dtype=np.int64); qos = np.zeros(K, dtype=np.int64)
    usr = np.zeros(K, dtype=np.int64); switch = np.zeros(K, dtype=np.int64)
    last: list[int | None] = [None] * K
    steps = 0
    while env.agents:
        action_dict = act_fn(obs_dict)
        obs_dict, reward_dict, _t, _u, info_dict = env.step(action_dict)
        for k, a in enumerate(env.possible_agents):
            ck = info_dict[a]
            r[k] += reward_dict[a]
            e[k] += float(ck["power_watts"]) * step_h
            if not ck["is_safe"]:      unsafe[k] += 1
            if ck["q_score"] < tau:    qos[k] += 1
            if ck["selected_upf"] != "DPDK": usr[k] += 1
            ak = action_dict[a]
            if last[k] is not None and last[k] != ak: switch[k] += 1
            last[k] = ak
        steps += 1
    n = max(1, steps)
    return {
        "steps": steps, "K": K,
        "total_reward": float(r.sum()),
        "per_cluster_reward": r.tolist(),
        "total_energy_wh": float(e.sum()),
        "agg_unsafe_rate": float(unsafe.sum() / (K * n)),
        "agg_qos_violation_rate": float(qos.sum() / (K * n)),
        "agg_usr_rate": float(usr.sum() / (K * n)),
        "agg_n_switches": int(switch.sum()),
    }


def const_fn(env, action: int):
    return lambda od: {a: action for a in env.possible_agents}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--arch", choices=["mappo"], default="mappo")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--line", type=str, default="A")
    p.add_argument("--outroot", type=Path, required=True)
    p.add_argument("--ckpt-dir", type=Path, default=None)
    a = p.parse_args()

    a.outroot.mkdir(parents=True, exist_ok=True)
    tag = f"{a.arch}_line{a.line}_a{a.alpha:g}_K{a.K}_s{a.seed}"
    out_json = a.outroot / f"{tag}.json"
    if out_json.exists():
        print(f"[skip] {tag}"); return 0

    t0 = time.time()
    cfg = build_cfg(a.alpha, a.K)
    paths_cfg = build_paths(a.K)

    train_env = MultiAgentUPFEnv(split="train", scenario_cfg=cfg, paths_cfg=paths_cfg)
    eval_env  = MultiAgentUPFEnv(split="val",   scenario_cfg=cfg, paths_cfg=paths_cfg)
    if train_env.K != a.K:
        raise SystemExit(f"FATAL: requested K={a.K} but env built K={train_env.K}")
    mcfg = MAPPOConfig(total_timesteps=a.steps, seed=a.seed, device="cpu",
                       train_split="train", eval_split="val")
    trainer = MAPPO(train_env, mcfg)
    trainer.learn(eval_env=eval_env, verbose=0, progress_bar=False)
    train_s = time.time() - t0

    ckpt_dir = a.ckpt_dir or (a.outroot / "ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(ckpt_dir / f"{tag}.pt")

    # --- test-split evaluation: learner + the classical baselines at this alpha ---
    dev = torch.device("cpu")
    actor = trainer.actor.to(dev).eval()

    def mappo_fn(od):
        obs = np.stack([od[x] for x in test_env.possible_agents]).astype(np.float32)
        with torch.no_grad():
            act, _ = actor.act(torch.from_numpy(obs).to(dev), deterministic=True)
        return {x: int(act[i].item()) for i, x in enumerate(test_env.possible_agents)}

    results = {}
    test_env = MultiAgentUPFEnv(split="test", scenario_cfg=cfg, paths_cfg=paths_cfg)
    results["mappo"] = rollout(test_env, mappo_fn)

    # thresholds re-derived from the surrogate for THIS alpha. The env does
    # not expose its twin, so build one from the same cfg (the pattern used
    # by research/phase7/evaluate_multiseed.py).
    spec = None
    try:
        from upf_digital_twin import DigitalTwin
        paths = paths_cfg
        twin = DigitalTwin(scenario_cfg=cfg, paths_cfg=paths, project_root=REPO)
        # scenario_rl.yaml carries no `threshold` block: the derived spec is
        # taken from the surrogate, and the reported hysteresis uses the TUNED
        # 20 Mbps band (Table 5.1). The auto band = 2*forecast MAE is ~221 Mbps
        # at alpha=1, which clamps t_down to 0 and degenerates to always-DPDK,
        # so it is recorded separately rather than used as the headline.
        summary = REPO / paths.get("traffic_forecaster", {}).get(
            "forecast_eval_summary",
            "data/external/traffic_forecaster/forecast_eval_summary.json")
        auto_mae = load_forecast_mae_gbps(summary, alpha_gbps_per_norm=a.alpha, K=a.K)
        spec = derive_thresholds(twin, safety_margin_mbps=10.0, forecast_mae_gbps=auto_mae)
    except Exception as exc:
        print(f"[warn] threshold derivation failed: {exc}")

    for name, act in (("always_dpdk", 0), ("always_usr", 1)):
        env = MultiAgentUPFEnv(split="test", scenario_cfg=cfg, paths_cfg=paths_cfg)
        results[name] = rollout(env, const_fn(env, act))

    if spec is not None:
        env = MultiAgentUPFEnv(split="test", scenario_cfg=cfg, paths_cfg=paths_cfg)
        TUNED_BAND_GBPS = 0.020        # Table 5.1's "band 20, cd 1"
        for label, t_up, t_down in (
            ("hysteresis", spec.decision_gbps, max(0.0, spec.decision_gbps - TUNED_BAND_GBPS)),
            ("hysteresis_auto", spec.t_up_gbps, spec.t_down_gbps),
        ):
            e2 = MultiAgentUPFEnv(split="test", scenario_cfg=cfg, paths_cfg=paths_cfg)
            hy = MultiAgentHysteresis(agents=list(e2.possible_agents), t_up_gbps=t_up,
                                      t_down_gbps=t_down, cooldown_steps=1)
            hy.reset()
            results[label] = rollout(e2, lambda od, _h=hy: _h(od))
            results[label]["t_up_mbps"] = t_up * 1000
            results[label]["t_down_mbps"] = t_down * 1000

        env = MultiAgentUPFEnv(split="test", scenario_cfg=cfg, paths_cfg=paths_cfg)
        thr = spec.decision_gbps
        FI = MultiAgentHysteresis.FORECAST_IDX
        results["threshold"] = rollout(
            env, lambda od: {x: (1 if float(od[x][FI]) < thr else 0)
                             for x in env.possible_agents})

    payload = {
        "tag": tag, "line": a.line, "alpha": a.alpha, "K": a.K, "seed": a.seed,
        "arch": a.arch, "total_timesteps": a.steps,
        "train_seconds": round(train_s, 1),
        "wall_seconds": round(time.time() - t0, 1),
        "thresholds": None if spec is None else {
            "decision_gbps": spec.decision_gbps, "qos_limit_gbps": spec.qos_limit_gbps,
            "energy_breakeven_gbps": spec.energy_breakeven_gbps,
            "t_up_gbps": spec.t_up_gbps, "t_down_gbps": spec.t_down_gbps,
        },
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[done] {tag}  train={train_s:.0f}s  total={payload['wall_seconds']:.0f}s  "
          f"reward={results['mappo']['total_reward']:.0f}  "
          f"unsafe={results['mappo']['agg_unsafe_rate']*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
