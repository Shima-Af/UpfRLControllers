"""Deterministic held-out test evaluation of every controller checkpoint.

One rollout per checkpoint on MultiAgentUPFEnv(split="test"), reset(seed=42),
full 1,009-step episode, argmax actions. This is the Phase-7 protocol
(research/phase7/evaluate_multiseed.py). The policy loaders are imported
from that script, so policies are applied exactly as for the paper.

Unlike the Phase-7 evaluator, every per-step, per-cluster quantity is saved
(rollouts/<set>__seed<N>.npz), so that reward components and physical
metrics can be recomputed without re-running the policies:
  reward, energy_term, qos_penalty, switch_penalty, cooldown_penalty,
  power_watts, power_watts_steady, switching_energy_wh, is_safe, q_score,
  usr (selected_upf == USR), action, actual_load_gbps.

Usage:
  python evaluate_all.py                 # all controller sets present on disk
  python evaluate_all.py --only primary  # only this study's retrained sets
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "experiments"
NEW = EXP / "wcnc2027_ablation"
SEEDS = [1, 7, 13, 23, 42, 64, 77, 99]
K = 10

# Legacy MAPPO checkpoints behind reports/phase-7/multiseed_summary_v04twin.json
# (latest-mtime dir per seed, as selected by evaluate_multiseed.discover_checkpoints;
# reproduced to 0.1 reward units in the feasibility audit).
LEGACY_MAPPO_V04 = {
    1: "mappo_seed1_20260517T110908", 7: "mappo_seed7_20260517T003433",
    13: "mappo_seed13_20260517T003433", 23: "mappo_seed23_20260517T110908",
    42: "mappo_seed42_20260526T143637", 64: "mappo_seed64_20260517T110908",
    77: "mappo_seed77_20260517T110908", 99: "mappo_seed99_20260517T003433",
}
LEGACY_IPPO_V04 = {s: next(EXP.glob(f"ppo_single_site_ensemble_seed{s}_*")).name for s in SEEDS}
LEGACY_CENTRAL_V04 = {s: f"ppo_multi_site_seed{s}_20260517T004954" for s in (7, 13, 42, 99)}


def controller_sets() -> dict[str, dict]:
    """{set_name: {"kind", "group", "ckpts": {seed: path}}} for everything on disk."""
    sets = {
        "MAPPO": dict(kind="mappo", group="primary",
                      ckpts={s: NEW / f"mappo_seed{s}" / "mappo_best.pt" for s in SEEDS}),
        "Shared-PPO": dict(kind="mappo", group="primary",
                           ckpts={s: NEW / f"shared_ppo_seed{s}" / "mappo_best.pt" for s in SEEDS}),
        "IPPO": dict(kind="ippo", group="primary",
                     ckpts={s: NEW / f"ippo_seed{s}" for s in SEEDS}),
        "Centralized PPO": dict(kind="central", group="primary",
                                ckpts={s: NEW / f"central_seed{s}" / "ppo_multi_site.zip" for s in SEEDS}),
        # --- legacy checkpoints, reported separately ---
        "legacy MAPPO (v0.4 re-scored set)": dict(kind="mappo", group="legacy",
            ckpts={s: EXP / d / "mappo_best.pt" for s, d in LEGACY_MAPPO_V04.items()}),
        "legacy MAPPO (mappo_nr, June)": dict(kind="mappo", group="legacy",
            ckpts={s: EXP / f"mappo_nr_seed{s}" / "mappo_best.pt" for s in SEEDS}),
        "legacy IPPO (ippo_nr, paper Table I)": dict(kind="ippo", group="legacy",
            ckpts={s: EXP / f"ippo_nr_seed{s}" for s in SEEDS}),
        "legacy IPPO (v0.4 re-scored set)": dict(kind="ippo", group="legacy",
            ckpts={s: EXP / d for s, d in LEGACY_IPPO_V04.items()}),
        "legacy Centralized PPO (central_nr, June+Aug)": dict(kind="central", group="legacy",
            ckpts={s: EXP / f"central_nr_seed{s}" / "ppo_multi_site.zip" for s in SEEDS}),
        "legacy Centralized PPO (v0.4 re-scored set)": dict(kind="central", group="legacy",
            ckpts={s: EXP / d / "ppo_multi_site.zip" for s, d in LEGACY_CENTRAL_V04.items()}),
    }
    for v in sets.values():
        ok = {}
        for s, p in v["ckpts"].items():
            if v["kind"] == "ippo":
                if all((p / f"cluster_{k}" / "ppo_single_site.zip").exists() for k in range(K)):
                    ok[s] = p
            elif p.exists():
                ok[s] = p
        v["ckpts"] = ok
    return sets


def _policy(kind: str, path: str):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "research" / "phase7"))
    import evaluate_multiseed as E  # noqa: E402
    from stable_baselines3 import PPO
    agents = [f"cluster_{k}" for k in range(K)]
    p = Path(path)
    if kind == "mappo":
        return E._mappo_policy(E._load_mappo_actor(p), agents)
    if kind == "central":
        return E._centralised_policy(PPO.load(p, device="cpu"))
    if kind == "ippo":
        return E._ensemble_policy(p)
    if kind == "always-DPDK":
        return E._const_policy(0)
    if kind.startswith("hysteresis"):
        from src.baselines.hysteresis import MultiAgentHysteresis
        band = float(kind.split("=")[1]) / 1000.0
        pol = MultiAgentHysteresis(agents=agents, t_up_gbps=0.081, t_down_gbps=max(0.0, 0.081 - band),
                                   cooldown_steps=1)
        pol.reset()
        return pol
    raise ValueError(kind)


def rollout_detailed(set_name: str, kind: str, seed: int, path: str) -> dict:
    import numpy as np
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    from src.envs.multi_agent_upf_env import MultiAgentUPFEnv

    t0 = time.time()
    fn = _policy(kind, path)
    env = MultiAgentUPFEnv(split="test")
    obs, _ = env.reset(seed=42)
    agents = list(env.possible_agents)
    T = env._N  # noqa: SLF001
    fields = ["reward", "energy_term", "qos_penalty", "switch_penalty", "cooldown_penalty",
              "power_watts", "power_watts_steady", "switching_energy_wh", "q_score", "actual_load_gbps"]
    arr = {f: np.zeros((T, K)) for f in fields}
    is_safe = np.zeros((T, K), bool)
    usr = np.zeros((T, K), bool)
    action = np.zeros((T, K), np.int8)
    t = 0
    while env.agents:
        act = fn(obs)
        obs, r, _te, _tr, info = env.step(act)
        for k, a in enumerate(agents):
            ck = info[a]
            arr["reward"][t, k] = r[a]
            for f in fields[1:]:
                arr[f][t, k] = float(ck[f])
            is_safe[t, k] = bool(ck["is_safe"])
            usr[t, k] = ck["selected_upf"] == "USR"
            action[t, k] = int(act[a])
        t += 1
    out = HERE / "rollouts" / f"{set_name.replace(' ', '_').replace('/', '-')}__seed{seed}.npz"
    np.savez_compressed(out, is_safe=is_safe, usr=usr, action=action, step_h=env.step_h, tau=env.tau,
                        steps=t, checkpoint=str(path), **arr)
    return dict(set=set_name, seed=seed, checkpoint=str(Path(path).relative_to(ROOT)) if str(path).startswith(str(ROOT)) else path,
                file=str(out.relative_to(HERE)), total_reward=float(arr["reward"].sum()), seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["primary", "legacy", "baselines", "all"], default="all")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    (HERE / "rollouts").mkdir(exist_ok=True)

    jobs = []
    sets = controller_sets()
    for name, v in sets.items():
        if args.only in ("all", v["group"]):
            for s, p in sorted(v["ckpts"].items()):
                jobs.append((name, v["kind"], s, str(p)))
    if args.only in ("all", "baselines"):
        jobs.append(("always-DPDK", "always-DPDK", 0, "-"))
        jobs.append(("Hysteresis (b=50 Mbps)", "hysteresis_band_mbps=50", 0, "-"))
    print(f"{len(jobs)} rollouts", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(rollout_detailed, *zip(*jobs)))
    manifest = HERE / "rollouts" / "manifest.json"
    old = json.loads(manifest.read_text()) if manifest.exists() else []
    keep = {(r["set"], r["seed"]): r for r in old}
    for r in results:
        keep[(r["set"], r["seed"])] = r
    manifest.write_text(json.dumps(sorted(keep.values(), key=lambda r: (r["set"], r["seed"])), indent=1))
    for r in results:
        print(f"{r['set']:48s} seed {r['seed']:>3}  R={r['total_reward']:>10.1f}  ({r['seconds']}s)")


if __name__ == "__main__":
    main()
