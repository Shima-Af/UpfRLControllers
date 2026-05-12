"""Phase 2 sanity check: train PPO on SingleSiteUPFEnv + compare baselines.

Usage:
    python scripts/train_ppo_single_site.py --total-timesteps 50000

Trains a small feed-forward PPO policy on one cluster, then rolls out four
policies on a fresh env instance and prints a comparison table:

    PPO (deterministic)
    random
    always DPDK
    always USR

Output directory layout (default `experiments/ppo_single_site_<seed>_<ts>`):
    ppo_single_site.zip       trained SB3 model
    tb/                       TensorBoard logs
    summary.txt               comparison table (also printed to stdout)
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.trainers.ppo_single_site import (  # noqa: E402
    constant_policy,
    ppo_policy,
    random_policy,
    rollout_episode,
    train_ppo_single_site,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cluster-idx", type=int, default=0)
    p.add_argument("--horizon-idx", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Run output dir (default: experiments/ppo_single_site_<seed>_<utc-ts>)",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the SB3 training progress bar.",
    )
    p.add_argument(
        "--max-eval-steps",
        type=int,
        default=None,
        help=(
            "Truncate each comparison rollout at this many env steps. "
            "Useful because the digital twin runs at ~5 steps/s, so a full "
            "1009-step episode per policy is ~3 minutes."
        ),
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=1024,
        help="PPO rollout length per update (default 1024).",
    )
    return p.parse_args()


def _format_row(name: str, m: dict) -> str:
    return (
        f"  {name:<14s} "
        f"total_r={m['total_reward']:>10.2f}  "
        f"mean_r={m['mean_reward']:>7.4f}  "
        f"energy_Wh={m['total_energy_wh']:>7.2f}  "
        f"switch_Wh={m['total_switch_wh']:>6.2f}  "
        f"unsafe={m['unsafe_rate']:.3f}  "
        f"DPDK={m['dpdk_rate']:.2f}  "
        f"USR={m['usr_rate']:.2f}  "
        f"flips={m['n_switches']:>3d}"
    )


def main() -> int:
    args = _parse_args()

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    if args.out_dir is None:
        args.out_dir = (
            REPO_ROOT / "experiments" / f"ppo_single_site_seed{args.seed}_{ts}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = args.out_dir / "tb"
    tb_dir.mkdir(exist_ok=True)

    print(f"Cluster idx:      {args.cluster_idx}")
    print(f"Horizon idx:      {args.horizon_idx}")
    print(f"Total timesteps:  {args.total_timesteps:,}")
    print(f"Seed:             {args.seed}")
    print(f"Output dir:       {args.out_dir}")
    print("-" * 70)
    print("Training PPO...")

    model = train_ppo_single_site(
        cluster_idx=args.cluster_idx,
        horizon_idx=args.horizon_idx,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        out_dir=args.out_dir,
        tensorboard_log=tb_dir,
        progress_bar=not args.no_progress,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=min(64, args.n_steps),
    )

    print("-" * 70)
    print("Rolling out one episode per policy on a fresh env instance...")

    policies = {
        "PPO":         ppo_policy(model, deterministic=True),
        "random":      random_policy(seed=args.seed),
        "always-DPDK": constant_policy(0),
        "always-USR":  constant_policy(1),
    }

    results = {}
    for name, pol in policies.items():
        results[name] = rollout_episode(
            cluster_idx=args.cluster_idx,
            horizon_idx=args.horizon_idx,
            policy=pol,
            seed=args.seed,
            max_steps=args.max_eval_steps,
        )

    table_lines = ["Comparison (single rollout, deterministic where applicable):"]
    for name, m in results.items():
        table_lines.append(_format_row(name, m))
    table = "\n".join(table_lines)
    print(table)

    # Best-by-total-reward callout.
    best_name = max(results, key=lambda k: results[k]["total_reward"])
    summary = (
        f"\nBest by total reward: {best_name}  "
        f"(reward={results[best_name]['total_reward']:.2f})"
    )
    print(summary)

    (args.out_dir / "summary.txt").write_text(table + summary + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
