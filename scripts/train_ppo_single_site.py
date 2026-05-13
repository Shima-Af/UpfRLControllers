"""Train PPO on SingleSiteUPFEnv under the paper-aligned reward.

Focus: verify PPO learns the intended energy-aware policy. The
comparison section reports the trained PPO alongside two reference
baselines (always-DPDK and the rule-based threshold) so we can see
both whether learning happened and how the trained policy stacks up
against the simplest reasonable controllers. Random / always-USR are
omitted from the headline — they exist mainly to sanity-check the
sign of the QoS penalty and were obscuring the comparison.

Usage:
    python scripts/train_ppo_single_site.py \\
        --total-timesteps 200000 --n-steps 1024 \\
        --ent-coef 0.15 --learning-rate 1e-4

Defaults follow Table 3 of COMCOM-S-26-00430.

Output directory layout (default `experiments/ppo_single_site_<seed>_<ts>`):
    best_model.zip            best deterministic eval return during training
    ppo_single_site.zip       same as best_model (stable name)
    ppo_single_site_final.zip final-step checkpoint
    evaluations.npz           EvalCallback eval history
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
    predicted_load_threshold_policy,
    rollout_episode,
    train_ppo_single_site,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cluster-idx", type=int, default=0)
    p.add_argument("--horizon-idx", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=200_000)
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
            "Default: full 1009-step episode."
        ),
    )
    p.add_argument(
        "--n-steps", type=int, default=1024,
        help="PPO rollout length per update (paper: 1024).",
    )
    p.add_argument(
        "--threshold-gbps",
        type=float,
        default=0.0905,
        help=(
            "Predicted-load threshold for the rule-based baseline "
            "(USR if predicted_load < threshold). Default 0.0905 Gbps "
            "matches the paper's profiling-derived DPDK<->USR boundary "
            "(90.47 Mbps)."
        ),
    )
    p.add_argument(
        "--ent-coef", type=float, default=0.15,
        help="PPO entropy coefficient (paper: 0.15).",
    )
    p.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="PPO learning rate (paper: 1e-4).",
    )
    p.add_argument(
        "--gamma", type=float, default=0.995,
        help="PPO discount factor (paper: 0.995).",
    )
    p.add_argument(
        "--gae-lambda", type=float, default=0.9,
        help="GAE lambda (paper: 0.9).",
    )
    return p.parse_args()


def _format_row(name: str, m: dict) -> str:
    return (
        f"  {name:<14s} "
        f"total_r={m['total_reward']:>10.2f}  "
        f"energy_Wh={m['total_energy_wh']:>7.2f}  "
        f"mean_SEC={m['mean_sec']:>8.5f}  "
        f"mean_Q={m['mean_q']:>5.3f}  "
        f"viol_rate={m['qos_violation_rate']:.3f}  "
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
    print(f"ent_coef:         {args.ent_coef}")
    print(f"learning_rate:    {args.learning_rate}")
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
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )

    print("-" * 70)
    print("Rolling out one episode per policy on a fresh env instance...")

    policies = {
        "PPO":         ppo_policy(model, deterministic=True),
        "always-DPDK": constant_policy(0),
        f"USR<{args.threshold_gbps:.3f}":
            predicted_load_threshold_policy(args.threshold_gbps),
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

    table_lines = [
        "Comparison (single rollout, PPO deterministic):",
    ]
    for name, m in results.items():
        table_lines.append(_format_row(name, m))
    table = "\n".join(table_lines)
    print(table)

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
