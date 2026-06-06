"""Phase 8 — Calibrate the shared fleet-power budget (`P_max`).

Rolls out two reference policies — all-DPDK and all-USR — across the
chosen split with the pool DISABLED, collects per-step fleet steady-state
power, and prints percentiles so the next training run can pick a
sensible value for ``pool.power_cap_w`` in ``configs/scenario_rl.yaml``.

The pool overhead at training time is the per-step sum
``sum_k composite.power_watts_steady`` — transition spikes are NOT
counted toward the cap (see ``single_site_upf_env.py``). This probe
reports that same scalar so the chosen ``P_max`` is directly comparable.

Usage
-----
    python research/phase8/probe_fleet_power.py
    python research/phase8/probe_fleet_power.py --split train --out-json results/p8_probe.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.envs.multi_site_upf_env import MultiSiteUPFEnv  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402

PERCENTILES = (1, 5, 25, 50, 75, 90, 95, 99)


def _rollout(env: MultiSiteUPFEnv, action_value: int, seed: int) -> np.ndarray:
    """Run one full episode under a fixed joint action; return fleet-power trace (W)."""
    env.reset(seed=seed)
    fixed = np.full(env.K, action_value, dtype=np.int64)
    powers: list[float] = []
    while True:
        _, _, term, trunc, info = env.step(fixed)
        powers.append(float(info["pool_power_w"]))
        if term or trunc:
            break
    return np.asarray(powers, dtype=np.float64)


def _summarise(trace: np.ndarray) -> dict[str, float]:
    stats: dict[str, float] = {
        "min": float(trace.min()),
        "max": float(trace.max()),
        "mean": float(trace.mean()),
        "n_steps": int(trace.size),
    }
    for p in PERCENTILES:
        stats[f"p{p}"] = float(np.percentile(trace, p))
    return stats


def _format_table(dpdk: dict[str, float], usr: dict[str, float]) -> str:
    rows = [
        ("min", "min"),
        ("p1", "p1"),
        ("p5", "p5"),
        ("p25", "p25"),
        ("p50", "p50 (median)"),
        ("p75", "p75"),
        ("p90", "p90"),
        ("p95", "p95"),
        ("p99", "p99"),
        ("max", "max"),
        ("mean", "mean"),
    ]
    lines = [
        f"  {'':<16}{'all-DPDK':>14}{'all-USR':>14}",
        "  " + "-" * 44,
    ]
    for key, label in rows:
        lines.append(f"  {label:<16}{dpdk[key]:>12.2f} W{usr[key]:>12.2f} W")
    return "\n".join(lines)


def _suggest(dpdk: dict[str, float], usr: dict[str, float]) -> list[tuple[str, float]]:
    # In this twin DPDK is polling => essentially flat steady-state power,
    # while USR is event-driven => load-dependent. The interesting sweep
    # range is therefore anchored on the all-USR distribution (the
    # policy-tunable case), with the DPDK ceiling as a reference floor.
    worst_case = max(dpdk["max"], usr["max"])
    return [
        (f"never-bind reference  (>= worst-case = {worst_case:.2f} W)", worst_case),
        ("very loose            (USR p99)", usr["p99"]),
        ("loose                 (USR p90)", usr["p90"]),
        ("moderate              (USR p75)", usr["p75"]),
        ("balanced              (USR p50)", usr["p50"]),
        ("tight                 (USR p25 -- frequently binds)", usr["p25"]),
        ("DPDK-only feasibility (all-DPDK max)", dpdk["max"]),
    ]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", type=str, default="train", choices=("train", "val", "test"))
    p.add_argument("--horizon-idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-json", type=Path, default=None,
        help="Optional path to write the full results as JSON.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # Force the pool OFF during calibration even if scenario_rl.yaml has
    # a cap set — we are profiling the underlying power, not measuring
    # behaviour under a penalty.
    cfg = copy.deepcopy(load_yaml(REPO_ROOT / "configs" / "scenario_rl.yaml"))
    cfg.setdefault("pool", {})
    cfg["pool"]["power_cap_w"] = None

    env = MultiSiteUPFEnv(
        scenario_cfg=cfg,
        split=args.split,
        horizon_idx=args.horizon_idx,
    )
    episode_len = int(env._N)  # noqa: SLF001
    print(f"Probing fleet steady-state power")
    print(f"  split={args.split!r}  K={env.K}  episode_length={episode_len}")
    print("-" * 60)

    dpdk_trace = _rollout(env, action_value=0, seed=args.seed)
    usr_trace = _rollout(env, action_value=1, seed=args.seed)
    dpdk_stats = _summarise(dpdk_trace)
    usr_stats = _summarise(usr_trace)

    print("Fleet steady-state power distribution (sum over K clusters):")
    print()
    print(_format_table(dpdk_stats, usr_stats))
    print()
    print("Suggested `pool.power_cap_w` candidates for the next MAPPO run:")
    for desc, val in _suggest(dpdk_stats, usr_stats):
        print(f"  {desc:<55} -> {val:>8.2f} W")
    print()
    print("Notes:")
    print("  - DPDK uses polling => fleet-power trace is nearly flat;")
    print("    USR is event-driven => trace scales with traffic load.")
    print("  - In this twin USR can exceed DPDK at peak load — so the")
    print("    pool penalty pushes the policy toward DPDK at high load")
    print("    (the opposite intuition from a naive instance-count cap).")
    print("  - A cap above worst-case never binds (pool effectively off).")
    print("  - All-DPDK is always feasible (constant power); a cap below")
    print("    all-DPDK max forces USR adoption at low load.")
    print("  - For a clean ablation, sweep `P_max` straddling all-USR")
    print("    p50 so the constraint genuinely matters but stays satisfiable.")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "split": args.split,
            "horizon_idx": args.horizon_idx,
            "seed": args.seed,
            "K": env.K,
            "episode_length": episode_len,
            "all_dpdk": dpdk_stats,
            "all_usr": usr_stats,
        }
        args.out_json.write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
