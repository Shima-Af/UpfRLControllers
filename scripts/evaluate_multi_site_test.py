"""One-shot evaluation of multi-site PPO on the held-out test slice.

Mirrors ``scripts/evaluate_test_split.py`` but for Phase 3.

Compares:
    - multi-PPO (centralised, joint MultiDiscrete action)
    - multi-PPO-ensemble (10 single-site PPOs stacked, if checkpoints
      are available under experiments/ppo_single_site_*/)
    - all-DPDK
    - per-cluster threshold
    - all-USR
    - random

Outputs:
    reports/phase-3/test_split_summary.json   structured per-cluster KPIs
    stdout                                    aggregate comparison table
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.envs.multi_site_upf_env import MultiSiteUPFEnv  # noqa: E402
from src.trainers.ppo_multi_site import (  # noqa: E402
    constant_multi_policy,
    ppo_multi_policy,
    predicted_load_threshold_multi_policy,
    random_multi_policy,
    rollout_multi_episode,
)


def _latest(glob: str) -> Path | None:
    matches = sorted(
        (REPO_ROOT / "experiments").glob(glob),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _ensemble_policy(per_cluster_checkpoints: list[Path], K: int):
    """Stack K single-site PPOs into a joint policy.

    Each underlying PPO consumes its own 14-dim slice of the obs and
    returns a binary action; the wrapper concatenates them.
    """
    models = [PPO.load(p) for p in per_cluster_checkpoints]

    def _fn(obs: np.ndarray) -> np.ndarray:
        out = np.zeros(K, dtype=np.int64)
        for k in range(K):
            sub_obs = obs[k * 14: (k + 1) * 14]
            a, _ = models[k].predict(sub_obs, deterministic=True)
            out[k] = int(np.asarray(a).item())
        return out

    return _fn


def _row(label: str, m: dict) -> str:
    return (
        f"  {label:<22s} "
        f"weighted_r={m['total_weighted_reward']:>9.2f}  "
        f"unweighted_r={m['total_unweighted_reward']:>10.2f}  "
        f"energy_Wh={m['total_energy_wh']:>8.2f}  "
        f"unsafe={m['agg_unsafe_rate'] * 100:>5.2f}%  "
        f"USR={m['agg_usr_rate'] * 100:>5.1f}%  "
        f"flips={m['agg_n_switches']:>4d}"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--horizon-idx", type=int, default=0)
    p.add_argument("--threshold-gbps", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--multi-ckpt", type=Path, default=None,
        help="Path to centralised PPO checkpoint (.zip). "
             "Defaults to the most recent ppo_multi_site_*/ppo_multi_site.zip.",
    )
    p.add_argument(
        "--ensemble-dir", type=Path, default=None,
        help="Directory containing 10 single-site checkpoints named "
             "cluster_<idx>/ppo_single_site.zip. If omitted, the ensemble "
             "baseline is skipped.",
    )
    p.add_argument(
        "--out-json", type=Path,
        default=REPO_ROOT / "reports" / "phase-3" / "test_split_summary.json",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # Resolve centralised PPO checkpoint.
    multi_ckpt = args.multi_ckpt
    if multi_ckpt is None:
        latest_run = _latest("ppo_multi_site_*")
        if latest_run is not None:
            cand = latest_run / "ppo_multi_site.zip"
            if cand.exists():
                multi_ckpt = cand

    K = 10
    print(f"Evaluating on split='test' (horizon_idx={args.horizon_idx}, K={K})")
    print("-" * 88)

    policies: dict[str, callable] = {}
    if multi_ckpt is not None and multi_ckpt.exists():
        model = PPO.load(multi_ckpt)
        policies["multi-PPO"] = ppo_multi_policy(model, deterministic=True)
        print(f"  centralised checkpoint: {multi_ckpt.relative_to(REPO_ROOT)}")
    else:
        print("  centralised checkpoint: <none found, skipping multi-PPO>")

    if args.ensemble_dir is not None:
        args.ensemble_dir = args.ensemble_dir.resolve()
        if args.ensemble_dir.is_dir():
            ckpts = []
            for k in range(K):
                cand = args.ensemble_dir / f"cluster_{k}" / "ppo_single_site.zip"
                if not cand.exists():
                    print(
                        f"  ensemble missing cluster {k}, "
                        "skipping ensemble baseline"
                    )
                    ckpts = []
                    break
                ckpts.append(cand)
            if ckpts:
                policies["multi-PPO-ensemble"] = _ensemble_policy(ckpts, K=K)
                try:
                    rel = args.ensemble_dir.relative_to(REPO_ROOT)
                except ValueError:
                    rel = args.ensemble_dir
                print(f"  ensemble dir: {rel}")

    policies["all-DPDK"] = constant_multi_policy(0, K=K)
    policies[f"threshold({args.threshold_gbps:.3f})"] = (
        predicted_load_threshold_multi_policy(args.threshold_gbps, K=K)
    )
    policies["all-USR"] = constant_multi_policy(1, K=K)
    policies["random"] = random_multi_policy(seed=args.seed, K=K)

    summaries: dict[str, dict] = {}
    for name, pol in policies.items():
        m = rollout_multi_episode(
            policy=pol,
            horizon_idx=args.horizon_idx,
            seed=args.seed,
            split="test",
        )
        print(_row(name, m))
        summaries[name] = m

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "split": "test",
                "K": K,
                "horizon_idx": args.horizon_idx,
                "threshold_gbps": args.threshold_gbps,
                "seed": args.seed,
                "multi_ckpt": str(multi_ckpt) if multi_ckpt else None,
                "ensemble_dir":
                    str(args.ensemble_dir) if args.ensemble_dir else None,
                "summaries": summaries,
            },
            indent=2,
            default=str,
        )
    )
    print("-" * 88)
    print(f"Wrote {args.out_json.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
