"""Train centralised PPO on MultiSiteUPFEnv (Phase 3).

Defaults: trains on the forecaster's training slice, selects best
checkpoint against the validation slice, refuses to touch ``test``
(use ``research/phase3/evaluate_test_split.py`` for that).

Usage:
    python scripts/train_ppo_multi_site.py \\
        --total-timesteps 200000 --n-steps 1024 \\
        --ent-coef 0.15 --learning-rate 1e-4

Output directory layout (``experiments/ppo_multi_site_<seed>_<ts>``):
    best_model.zip            best deterministic eval return during training
    ppo_multi_site.zip        same as best_model (stable filename)
    ppo_multi_site_final.zip  final-step checkpoint
    evaluations.npz           EvalCallback eval history
    tb/                       TensorBoard logs
    summary.txt               end-of-run comparison table
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.trainers.ppo_multi_site import (  # noqa: E402
    constant_multi_policy,
    ppo_multi_policy,
    predicted_load_threshold_multi_policy,
    rollout_multi_episode,
    train_ppo_multi_site,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--horizon-idx", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Run output dir (default: experiments/ppo_multi_site_<seed>_<utc-ts>)",
    )
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--n-steps", type=int, default=1024,
        help="PPO rollout length per update (paper: 1024).",
    )
    p.add_argument("--ent-coef", type=float, default=0.15)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.9)
    p.add_argument(
        "--threshold-gbps", type=float, default=0.05,
        help="Per-cluster threshold for the rule baseline.",
    )
    p.add_argument(
        "--train-split", type=str, default="train",
        choices=["train", "val", "test"],
    )
    p.add_argument(
        "--eval-split", type=str, default="val",
        choices=["train", "val", "test"],
        help="Forecaster slice for EvalCallback. 'test' is refused.",
    )
    p.add_argument(
        "--rollout-split", type=str, default="val",
        choices=["train", "val", "test"],
        help="End-of-run comparison table slice. 'test' is refused.",
    )
    return p.parse_args()


def _format_row(name: str, m: dict) -> str:
    return (
        f"  {name:<18s} "
        f"weighted_r={m['total_weighted_reward']:>10.2f}  "
        f"unweighted_r={m['total_unweighted_reward']:>10.2f}  "
        f"energy_Wh={m['total_energy_wh']:>8.2f}  "
        f"unsafe={m['agg_unsafe_rate'] * 100:>5.2f}%  "
        f"USR={m['agg_usr_rate'] * 100:>5.1f}%  "
        f"flips={m['agg_n_switches']:>4d}"
    )


def main() -> int:
    args = _parse_args()
    if args.eval_split == "test" or args.rollout_split == "test":
        raise SystemExit(
            "Refusing to use the test split during training or end-of-run "
            "rollout. Use research/phase3/evaluate_test_split.py for the test set."
        )

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    if args.out_dir is None:
        args.out_dir = (
            REPO_ROOT / "experiments" / f"ppo_multi_site_seed{args.seed}_{ts}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = args.out_dir / "tb"
    tb_dir.mkdir(exist_ok=True)

    print(f"Horizon idx:      {args.horizon_idx}")
    print(f"Total timesteps:  {args.total_timesteps:,}")
    print(f"Seed:             {args.seed}")
    print(f"ent_coef:         {args.ent_coef}")
    print(f"learning_rate:    {args.learning_rate}")
    print(f"Train split:      {args.train_split}")
    print(f"Eval split:       {args.eval_split}")
    print(f"Rollout split:    {args.rollout_split}")
    print(f"Output dir:       {args.out_dir}")
    print("-" * 70)
    print("Training PPO (centralised, MultiDiscrete([2]*K))...")

    model = train_ppo_multi_site(
        horizon_idx=args.horizon_idx,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        out_dir=args.out_dir,
        tensorboard_log=tb_dir,
        progress_bar=not args.no_progress,
        verbose=1,
        train_split=args.train_split,
        eval_split=args.eval_split,
        n_steps=args.n_steps,
        batch_size=min(64, args.n_steps),
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )

    print("-" * 70)
    print(f"Rolling out one episode per policy on split={args.rollout_split!r}...")

    K = 10  # forecaster currently produces K=10
    policies = {
        "PPO":              ppo_multi_policy(model, deterministic=True),
        "all-DPDK":         constant_multi_policy(0, K=K),
        f"threshold({args.threshold_gbps:.3f})":
            predicted_load_threshold_multi_policy(args.threshold_gbps, K=K),
    }

    results = {}
    for name, pol in policies.items():
        results[name] = rollout_multi_episode(
            policy=pol,
            horizon_idx=args.horizon_idx,
            seed=args.seed,
            split=args.rollout_split,
        )

    table_lines = ["Comparison (single rollout, PPO deterministic):"]
    for name, m in results.items():
        table_lines.append(_format_row(name, m))
    table = "\n".join(table_lines)
    print(table)

    best_name = max(results, key=lambda k: results[k]["total_weighted_reward"])
    summary = (
        f"\nBest by weighted total reward: {best_name}  "
        f"(weighted_reward={results[best_name]['total_weighted_reward']:.2f})"
    )
    print(summary)
    (args.out_dir / "summary.txt").write_text(table + summary + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
