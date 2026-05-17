"""Paired bootstrap test: MAPPO vs IPPO ensemble across training seeds.

Reads `reports/phase-7/multiseed_summary.json` and:
  1. Pairs MAPPO and IPPO test rewards on common seeds.
  2. Computes the per-seed delta (MAPPO - IPPO; positive => MAPPO wins).
  3. Resamples the seeds with replacement B times (default 10 000) and
     computes the bootstrap distribution of mean delta.
  4. Reports mean delta, 95% percentile CI, and one-sided p-value
     (fraction of bootstrap means <= 0).

Paired because each seed is one independent training run; we test
whether the seed-matched difference is consistently above zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "reports/phase-7/multiseed_summary.json",
    )
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    d = json.loads(args.summary.read_text())

    mappo_seeds = d["aggregated"]["MAPPO"]["per_seed"]
    ippo_seeds = d["aggregated"]["IPPO-ensemble"]["per_seed"]
    common = sorted(set(mappo_seeds) & set(ippo_seeds), key=int)
    if len(common) < 4:
        print(f"need at least 4 paired seeds, got {len(common)}", file=sys.stderr)
        return 1

    mappo_r = np.array(
        [mappo_seeds[s]["total_reward_unweighted"] for s in common]
    )
    ippo_r = np.array(
        [ippo_seeds[s]["total_reward_unweighted"] for s in common]
    )
    delta = mappo_r - ippo_r  # positive => MAPPO wins (less negative reward)

    rng = np.random.default_rng(args.seed)
    n = len(common)
    boot = np.empty(args.n_bootstrap)
    for b in range(args.n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[b] = delta[idx].mean()

    mean_delta = float(delta.mean())
    obs_std = float(delta.std(ddof=1))
    alpha = (1 - args.ci) / 2
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
    p_one_sided = float((boot <= 0).mean())  # H0: MAPPO no better than IPPO
    boot_mean = float(boot.mean())
    boot_std = float(boot.std())

    print(f"Paired bootstrap MAPPO vs IPPO ensemble (test split, v2 reward)")
    print(f"  paired seeds:           {common}  (n={n})")
    print(f"  per-seed delta MAPPO-IPPO:")
    for s, d_ in zip(common, delta):
        marker = "MAPPO wins" if d_ > 0 else "IPPO wins"
        print(f"    seed {s:>3}: {d_:>+8.1f}  ({marker})")
    print(f"  observed mean delta:    {mean_delta:+.2f}")
    print(f"  observed std (paired):  {obs_std:.2f}")
    print(f"  bootstrap mean (B={args.n_bootstrap}): {boot_mean:+.2f}")
    print(f"  bootstrap std:          {boot_std:.2f}")
    print(f"  {int(args.ci*100)}% CI of mean delta: [{lo:+.2f}, {hi:+.2f}]")
    print(f"  one-sided p (H0: delta<=0): {p_one_sided:.4f}")
    if lo > 0:
        verdict = f"MAPPO > IPPO significant at {int(args.ci*100)}% (CI excludes 0)"
    elif hi < 0:
        verdict = f"IPPO > MAPPO significant at {int(args.ci*100)}% (CI excludes 0)"
    else:
        verdict = (
            f"NOT statistically distinguishable at {int(args.ci*100)}% "
            f"(CI [{lo:+.1f}, {hi:+.1f}] crosses 0)"
        )
    print(f"  verdict: {verdict}")

    out_path = REPO_ROOT / "reports/phase-7/paired_bootstrap_mappo_vs_ippo.json"
    out_path.write_text(
        json.dumps(
            {
                "common_seeds": list(map(int, common)),
                "per_seed_delta": delta.tolist(),
                "mean_delta": mean_delta,
                "observed_std_paired": obs_std,
                "n_bootstrap": args.n_bootstrap,
                "ci_level": args.ci,
                "ci_low": float(lo),
                "ci_high": float(hi),
                "p_one_sided_mappo_ge_ippo": p_one_sided,
                "verdict": verdict,
            },
            indent=2,
        )
    )
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
