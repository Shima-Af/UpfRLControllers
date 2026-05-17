"""One-shot evaluation of the best PPO checkpoint on the held-out test slice.

This is the *only* script in the repo that touches ``split="test"`` for
PPO. The trainer and the EvalCallback are pinned to train/val so the
test slice stays untouched until the moment Phase 2 quotes its
headline number — preventing the in-sample evaluation we shipped
before the train/val/test arrays were available.

Outputs:
    reports/phase-2/test_split_summary.json   structured KPIs per policy
    stdout                                    human-readable comparison

Re-run after any retrain. The figure-generation script consumes the
same checkpoint but is parameterised by ``--split`` so it can render
val-set debug figures alongside the test-set ones.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.backend.app.policies import (  # noqa: E402
    POLICY_LABELS,
    PolicyRegistry,
    run_rollout,
)

POLICY_ORDER = ["ppo", "threshold", "always-dpdk", "random", "always-usr"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cluster-idx", type=int, default=0)
    p.add_argument("--horizon-idx", type=int, default=0)
    p.add_argument(
        "--threshold-gbps", type=float, default=0.05,
        help="Threshold for the rule-based baseline (USR if pred < threshold).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-json", type=Path,
        default=REPO_ROOT / "reports" / "phase-2" / "test_split_summary.json",
        help="Where to write the structured JSON summary.",
    )
    return p.parse_args()


def _row(label: str, s) -> str:  # noqa: ANN001 — RolloutSummary
    return (
        f"  {label:<18s} "
        f"total_r={s.total_reward:>10.2f}  "
        f"energy_Wh={s.total_energy_wh:>6.2f}  "
        f"unsafe={s.unsafe_rate * 100:>5.2f}%  "
        f"USR={s.usr_rate * 100:>5.1f}%  "
        f"flips={s.n_switches:>3d}"
    )


def main() -> int:
    args = _parse_args()
    registry = PolicyRegistry(repo_root=REPO_ROOT)

    print(
        f"Evaluating policies on split='test' "
        f"(cluster_idx={args.cluster_idx}, horizon_idx={args.horizon_idx})"
    )
    print("-" * 70)

    summaries: dict[str, dict] = {}
    for pid in POLICY_ORDER:
        info = next(p for p in registry.list_policies() if p.id == pid)
        if info.requires_model and not info.model_loaded:
            print(f"  skipping {pid}: no checkpoint")
            continue
        r = run_rollout(
            registry,
            policy_id=pid,
            cluster_idx=args.cluster_idx,
            horizon_idx=args.horizon_idx,
            seed=args.seed,
            threshold_gbps=args.threshold_gbps,
            max_steps=None,
            split="test",
        )
        print(_row(POLICY_LABELS[pid], r.summary))
        summaries[pid] = r.summary.model_dump()

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "split": "test",
                "cluster_idx": args.cluster_idx,
                "horizon_idx": args.horizon_idx,
                "threshold_gbps": args.threshold_gbps,
                "seed": args.seed,
                "checkpoint_path": str(registry._ppo_path),  # noqa: SLF001
                "summaries": summaries,
            },
            indent=2,
            default=str,
        )
    )
    print("-" * 70)
    print(f"Wrote {args.out_json.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
