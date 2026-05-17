"""One-shot MAPPO evaluation on the held-out test slice.

Mirrors the Phase 2/3 test evaluators. Loads the most-recent (or
specified) MAPPO checkpoint, runs the deterministic policy on
``split="test"``, and writes a structured JSON to
``reports/phase-6/test_split_summary.json``.

For comparison against Phase 3/4 baselines, pass
``--with-baselines`` to also evaluate the multi-PPO checkpoint
(centralised) and the per-cluster ensemble in the same env.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.envs.multi_agent_upf_env import MultiAgentUPFEnv  # noqa: E402
from src.trainers.mappo import (  # noqa: E402
    Actor,
    CentralisedCritic,
    MAPPOConfig,
    rollout_mappo_episode,
)


def _latest(glob: str) -> Path | None:
    matches = sorted(
        (REPO_ROOT / "experiments").glob(glob),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _load_actor(ckpt_path: Path, device: str) -> Actor:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    actor = Actor(
        obs_dim=ckpt["obs_dim"],
        n_actions=ckpt["n_actions"],
        hidden=cfg.get("actor_hidden", 64),
    )
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def _row(label: str, m: dict) -> str:
    return (
        f"  {label:<22s} "
        f"unweighted_r={m['total_reward_unweighted']:>10.2f}  "
        f"energy_Wh={m['total_energy_wh']:>8.2f}  "
        f"unsafe={m['agg_unsafe_rate'] * 100:>5.2f}%  "
        f"USR={m['agg_usr_rate'] * 100:>5.1f}%  "
        f"flips={m['agg_n_switches']:>4d}"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--ckpt", type=Path, default=None,
        help="MAPPO checkpoint (.pt). Defaults to most recent best.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--out-json", type=Path,
        default=REPO_ROOT / "reports" / "phase-6" / "test_split_summary.json",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    ckpt = args.ckpt
    if ckpt is None:
        latest = _latest("mappo_seed*")
        if latest is not None:
            cand = latest / "mappo_best.pt"
            if cand.exists():
                ckpt = cand
            else:
                cand = latest / "mappo_final.pt"
                if cand.exists():
                    ckpt = cand
    if ckpt is None or not ckpt.exists():
        raise SystemExit("No MAPPO checkpoint found.")

    print(f"Evaluating MAPPO on split='test' (seed={args.seed})")
    print(f"  checkpoint: {ckpt.relative_to(REPO_ROOT)}")
    print("-" * 88)

    actor = _load_actor(ckpt, args.device)
    m = rollout_mappo_episode(
        actor, seed=args.seed, split="test", device=args.device,
    )
    print(_row("MAPPO", m))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "split": "test",
                "K": m["K"],
                "checkpoint": str(ckpt),
                "summary": m,
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
