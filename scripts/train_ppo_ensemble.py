"""Train one single-site PPO per cluster — the Phase 3 ensemble baseline.

Runs ``train_ppo_single_site`` for each cluster index, writing the
best-of-val checkpoint to:

    experiments/ppo_single_site_ensemble_<ts>/cluster_<k>/ppo_single_site.zip

``scripts/evaluate_multi_site_test.py --ensemble-dir <that dir>`` then
stacks them into a joint policy for the head-to-head against the
centralised PPO.

Uses ``ProcessPoolExecutor`` to run ``--n-parallel`` clusters at once.
Each worker spawns its own SB3 process; SB3 grabs ~1 CPU core for
training plus a few for matmul, so 4-way parallel is comfortable on
this 32-core box and finishes in ~50 min instead of ~3.3 h sequential.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _train_one(
    cluster_idx: int,
    out_dir: str,
    total_timesteps: int,
    seed: int,
    n_steps: int,
    ent_coef: float,
    learning_rate: float,
) -> dict:
    # Heavy imports inside the worker process keep the parent light
    # and avoid pickling sklearn surrogate models across the fork.
    import warnings

    warnings.filterwarnings("ignore")
    from src.trainers.ppo_single_site import train_ppo_single_site

    out_path = Path(out_dir) / f"cluster_{cluster_idx}"
    out_path.mkdir(parents=True, exist_ok=True)
    train_ppo_single_site(
        cluster_idx=cluster_idx,
        horizon_idx=0,
        total_timesteps=total_timesteps,
        seed=seed + cluster_idx,
        out_dir=out_path,
        progress_bar=False,
        verbose=0,
        train_split="train",
        eval_split="val",
        n_steps=n_steps,
        batch_size=min(64, n_steps),
        ent_coef=ent_coef,
        learning_rate=learning_rate,
    )
    return {"cluster_idx": cluster_idx, "out_dir": str(out_path)}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--ent-coef", type=float, default=0.15)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--n-parallel", type=int, default=4)
    p.add_argument("--K", type=int, default=10)
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Root dir for the ensemble runs "
             "(default: experiments/ppo_single_site_ensemble_<utc-ts>)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    if args.out_dir is None:
        args.out_dir = (
            REPO_ROOT / "experiments" / f"ppo_single_site_ensemble_{ts}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training ensemble of {args.K} single-site PPOs")
    print(f"  total_timesteps per cluster: {args.total_timesteps:,}")
    print(f"  parallelism: {args.n_parallel}")
    print(f"  out dir: {args.out_dir}")
    print("-" * 60)

    # Cap CPU oversubscription — each SB3 worker may already use 4+ BLAS threads.
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")

    futures = {}
    with ProcessPoolExecutor(max_workers=args.n_parallel) as pool:
        for k in range(args.K):
            fut = pool.submit(
                _train_one,
                cluster_idx=k,
                out_dir=str(args.out_dir),
                total_timesteps=args.total_timesteps,
                seed=args.seed,
                n_steps=args.n_steps,
                ent_coef=args.ent_coef,
                learning_rate=args.learning_rate,
            )
            futures[fut] = k
        done = 0
        for fut in as_completed(futures):
            k = futures[fut]
            try:
                res = fut.result()
                done += 1
                print(
                    f"  [{done:2d}/{args.K}] cluster {k} done -> "
                    f"{Path(res['out_dir']).relative_to(REPO_ROOT)}"
                )
            except Exception as e:  # noqa: BLE001
                print(f"  cluster {k} FAILED: {e!r}")
                raise

    print("-" * 60)
    print(f"Ensemble dir: {args.out_dir}")
    print(
        "Next: python scripts/evaluate_multi_site_test.py "
        f"--ensemble-dir {args.out_dir.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
