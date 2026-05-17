"""End-to-end driver for Slice 0:
canonical Plan -> compile reward weights -> fine-tune -> replay -> report metrics.

Usage:
    # Smoke (no fine-tune, replays the base c0 checkpoint as-is):
    python -m intent.cli --plan balanced --skip-finetune

    # Real run (short fine-tune, then replay on val):
    python -m intent.cli --plan energy_greedy --finetune-steps 20000

The output is the per-plan metrics dict. There is no automated
accept/reject in Slice 0 — judging whether the fine-tune produced
the intended shift is a human's call, by reading the metrics. The
predicate verifier comes back in Slice 2, once the LLM can emit
thresholds grounded in observed performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from intent.compile.plan_to_weights import compile_weights  # noqa: E402
from intent.examples.canonical_plans import CANONICAL_PLANS  # noqa: E402
from intent.finetune.finetune_c0 import finetune  # noqa: E402
from intent.verify.twin_replay import replay  # noqa: E402

_DEFAULT_BASE = (
    "experiments/ppo_single_site_ensemble_seed1_20260517T110907/"
    "cluster_0/best_model.zip"
)


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--plan",
        required=True,
        choices=sorted(CANONICAL_PLANS),
        help="Name of a canonical Plan in intent/examples/canonical_plans.py",
    )
    p.add_argument(
        "--base-checkpoint",
        default=_DEFAULT_BASE,
        help="Path to the base c0 PPO checkpoint to fine-tune from.",
    )
    p.add_argument("--finetune-steps", type=int, default=20_000)
    p.add_argument("--cluster-idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--eval-split", default="val", choices=["train", "val", "test"]
    )
    p.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Replay the base checkpoint directly (faster smoke test).",
    )
    args = p.parse_args()

    plan = CANONICAL_PLANS[args.plan]
    weights = compile_weights(plan)

    _section(f"Plan: {args.plan}")
    print(json.dumps(plan.model_dump(), indent=2))
    _section("Compiled reward weights")
    print(json.dumps(weights, indent=2))

    if args.skip_finetune:
        ckpt = Path(args.base_checkpoint)
        _section(f"Replay (skip-finetune) — base: {ckpt}")
    else:
        _section(
            f"Fine-tune ({args.finetune_steps} steps from {args.base_checkpoint})"
        )
        ckpt = finetune(
            base_checkpoint=args.base_checkpoint,
            reward_weights=weights,
            cluster_idx=args.cluster_idx,
            timesteps=args.finetune_steps,
            seed=args.seed,
        )
        print(f"Fine-tuned checkpoint: {ckpt}")

    _section(f"Replay metrics on split={args.eval_split!r}")
    metrics = replay(
        ckpt,
        cluster_idx=args.cluster_idx,
        split=args.eval_split,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
