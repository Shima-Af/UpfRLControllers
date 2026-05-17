"""Train MAPPO (Phase 6) on the multi-agent UPF env.

Defaults: train on `split="train"`, val-based best-checkpoint
selection, refuses to touch the test split (use
``research/phase6/evaluate_test_split.py`` for that).

Usage:
    python scripts/train_mappo.py --total-timesteps 200000

Output directory: ``experiments/mappo_seed<seed>_<utc-ts>/``
    mappo_best.pt    best-of-val checkpoint
    mappo_final.pt   final-step checkpoint
    eval_log.json    eval-time return history
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.envs.multi_agent_upf_env import MultiAgentUPFEnv  # noqa: E402
from src.trainers.mappo import MAPPO, MAPPOConfig  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.9)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.05)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--actor-hidden", type=int, default=64)
    p.add_argument("--critic-hidden", type=int, default=128)
    p.add_argument("--reward-scale", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--eval-freq", type=int, default=4096)
    p.add_argument(
        "--train-split", type=str, default="train",
        choices=["train", "val", "test"],
    )
    p.add_argument(
        "--eval-split", type=str, default="val",
        choices=["train", "val", "test"],
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Run output dir (default: experiments/mappo_seed<seed>_<utc-ts>)",
    )
    p.add_argument(
        "--no-progress", action="store_true",
        help="Disable the tqdm progress bar.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.eval_split == "test":
        raise SystemExit(
            "Refusing to use the test split for EvalCallback. Use "
            "research/phase6/evaluate_test_split.py for the test-set headline number."
        )

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    if args.out_dir is None:
        args.out_dir = (
            REPO_ROOT / "experiments" / f"mappo_seed{args.seed}_{ts}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Train split:     {args.train_split}")
    print(f"Eval split:      {args.eval_split}")
    print(f"Reward scale:    {args.reward_scale}")
    print(f"Device:          {args.device}")
    print(f"Output dir:      {args.out_dir}")
    print("-" * 70)

    env = MultiAgentUPFEnv(split=args.train_split)
    eval_env = MultiAgentUPFEnv(split=args.eval_split)

    cfg = MAPPOConfig(
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        actor_hidden=args.actor_hidden,
        critic_hidden=args.critic_hidden,
        reward_scale=args.reward_scale,
        device=args.device,
        seed=args.seed,
        eval_freq=args.eval_freq,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )

    trainer = MAPPO(env, cfg)
    print(
        f"Actor params:  {sum(p.numel() for p in trainer.actor.parameters()):>7d}"
    )
    print(
        f"Critic params: {sum(p.numel() for p in trainer.critic.parameters()):>7d}"
    )
    print("-" * 70)

    trainer.learn(
        out_dir=args.out_dir,
        eval_env=eval_env,
        verbose=1,
        progress_bar=not args.no_progress,
    )

    print("-" * 70)
    print(f"Best eval return: {trainer.best_eval_return:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
