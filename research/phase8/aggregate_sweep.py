"""Phase 8 — Aggregate sweep results into a paper-ready CSV + plots.

Walks every ``experiments/mappo_<tag>_seed<n>_<ts>/`` directory, loads the
``mappo_best.pt`` (falling back to ``mappo_final.pt``), evaluates it on
the requested split under **the same scenario overlay the run was
trained with** (so reward/pool numbers are self-consistent), and produces:

  research/phase8/results/sweep_results_long.csv
      one row per (tag, seed) with all per-run metrics.

  research/phase8/results/sweep_results_long_summary.csv
      pandas groupby mean/std/count across seeds, indexed by (family, tag).

  research/phase8/results/sweep_<family>_summary.png
      one figure per sweep family with bar charts of the headline metrics
      (reward / energy / unsafe / switches / mean fleet power) over tag,
      error bars = std across seeds.

The matching overlay is resolved by tag: ``mappo_pool_balanced_15w_*``
maps to ``configs/sweeps/pool_balanced_15w.yaml``. If no overlay exists
for a tag (e.g., legacy checkpoints), the base scenario config is used.

Usage
-----
Default — process every mappo run dir:
    python research/phase8/aggregate_sweep.py

Process only the most recent overnight sweep:
    python research/phase8/aggregate_sweep.py \\
        --pattern 'experiments/mappo_*_seed*_20260527T*'

Pin a custom output location:
    python research/phase8/aggregate_sweep.py \\
        --out-csv results/my_sweep.csv \\
        --out-figures-dir results/
"""
from __future__ import annotations

import argparse
import copy
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.envs.multi_agent_upf_env import MultiAgentUPFEnv  # noqa: E402
from src.trainers.mappo import Actor  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402

CKPT_RE = re.compile(r"^mappo_(?P<tag>.+)_seed(?P<seed>\d+)_(?P<ts>\d{8}T\d{6})$")

FAMILY_PREFIXES = {
    "pool":      "pool_",
    "lambda_sw": "lambda_sw_",
    "cooldown":  "cooldown_",
    "budget":    "budget_",
}


def _family_of(tag: str) -> str:
    for fam, pref in FAMILY_PREFIXES.items():
        if tag.startswith(pref):
            return fam
    return "other"


def _deep_merge(base: dict, overlay: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _scenario_cfg_for_tag(tag: str) -> tuple[dict, bool]:
    """Return (config_dict, overlay_found)."""
    base = load_yaml(REPO_ROOT / "configs" / "scenario_rl.yaml")
    overlay_path = REPO_ROOT / "configs" / "sweeps" / f"{tag}.yaml"
    if overlay_path.exists():
        return _deep_merge(base, load_yaml(overlay_path)), True
    return base, False


def _load_actor(ckpt_path: Path) -> Actor:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    a = Actor(ckpt["obs_dim"], ckpt["n_actions"], cfg.get("actor_hidden", 64))
    a.load_state_dict(ckpt["actor"])
    a.eval()
    return a


@torch.no_grad()
def _act(actor: Actor, agents: list[str], obs_dict: dict) -> dict[str, int]:
    obs = np.stack([obs_dict[a] for a in agents]).astype(np.float32)
    action, _ = actor.act(torch.from_numpy(obs), deterministic=True)
    return {a: int(action[k].item()) for k, a in enumerate(agents)}


def evaluate(
    ckpt_path: Path,
    scenario_cfg: dict,
    split: str,
    seed: int,
) -> dict[str, float | int | None]:
    """One deterministic rollout of the MAPPO actor on the chosen split."""
    env = MultiAgentUPFEnv(split=split, scenario_cfg=scenario_cfg)
    obs, _ = env.reset(seed=seed)
    agents = list(env.possible_agents)
    K = env.K
    step_h = float(env.step_h)
    tau = float(env.tau)
    actor = _load_actor(ckpt_path)

    pool_cap_w = scenario_cfg.get("pool", {}).get("power_cap_w", None)

    per_c_reward = np.zeros(K, dtype=np.float64)
    per_c_energy = np.zeros(K, dtype=np.float64)
    per_c_unsafe = np.zeros(K, dtype=np.int64)
    per_c_qosv = np.zeros(K, dtype=np.int64)
    per_c_dpdk = np.zeros(K, dtype=np.int64)
    per_c_switch = np.zeros(K, dtype=np.int64)
    last_act: list[int | None] = [None] * K

    fleet_power_trace: list[float] = []
    pool_overrun_steps = 0
    pool_penalty_total = 0.0
    steps = 0

    while env.agents:
        action = _act(actor, agents, obs)
        obs, r, _term, _trunc, info = env.step(action)
        for k, a in enumerate(agents):
            per_c_reward[k] += r[a]
            ck = info[a]
            per_c_energy[k] += float(ck["power_watts_steady"]) * step_h
            if not ck["is_safe"]:
                per_c_unsafe[k] += 1
            if ck["q_score"] < tau:
                per_c_qosv[k] += 1
            if ck["selected_upf"] == "DPDK":
                per_c_dpdk[k] += 1
            ak = int(action[a])
            if last_act[k] is not None and last_act[k] != ak:
                per_c_switch[k] += 1
            last_act[k] = ak

        # Fleet-level pool stats — share is identical across agents under
        # the uniform-distribution penalty, so reading from agent 0 is fine.
        a0_info = info[agents[0]]
        if "pool_power_w" in a0_info:
            fleet_power_trace.append(float(a0_info["pool_power_w"]))
            if float(a0_info.get("pool_overrun_w", 0.0)) > 0:
                pool_overrun_steps += 1
                pool_penalty_total += float(a0_info.get("pool_penalty_share", 0.0)) * K
        else:
            # Pool disabled — derive fleet power from per-cluster steady.
            fleet_power_trace.append(
                sum(float(info[a]["power_watts_steady"]) for a in agents)
            )
        steps += 1

    n = max(1, steps)
    fleet_power = np.asarray(fleet_power_trace, dtype=np.float64)
    out: dict[str, float | int | None] = {
        "steps": steps,
        "K": K,
        "split": split,
        "eval_seed": seed,
        "total_reward_unweighted": float(per_c_reward.sum()),
        "total_energy_wh": float(per_c_energy.sum()),
        "agg_unsafe_rate": float(per_c_unsafe.sum() / (K * n)),
        "agg_qos_violation_rate": float(per_c_qosv.sum() / (K * n)),
        "agg_dpdk_rate": float(per_c_dpdk.sum() / (K * n)),
        "agg_n_switches": int(per_c_switch.sum()),
        "mean_fleet_power_w": float(fleet_power.mean()),
        "p95_fleet_power_w": float(np.percentile(fleet_power, 95)),
        "max_fleet_power_w": float(fleet_power.max()),
        "pool_cap_w": pool_cap_w,
        "pool_overrun_rate": float(pool_overrun_steps / n) if pool_cap_w is not None else None,
        "pool_penalty_total": float(pool_penalty_total) if pool_cap_w is not None else None,
    }
    return out


PLOT_METRICS: list[tuple[str, str]] = [
    ("total_reward_unweighted", "Total reward"),
    ("total_energy_wh",         "Total energy (Wh)"),
    ("agg_unsafe_rate",         "Unsafe fraction"),
    ("agg_n_switches",          "# switches"),
    ("mean_fleet_power_w",      "Mean fleet power (W)"),
]


def _plot_family(df_family: pd.DataFrame, family: str, out_path: Path) -> None:
    n = len(PLOT_METRICS)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2), dpi=120)
    if n == 1:
        axes = [axes]
    for ax, (col, label) in zip(axes, PLOT_METRICS):
        g = df_family.groupby("tag")[col].agg(["mean", "std", "count"]).sort_index()
        x = np.arange(len(g))
        ax.bar(x, g["mean"], yerr=g["std"].fillna(0.0), capsize=3,
               color="steelblue", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(g.index, rotation=40, ha="right", fontsize=7)
        ax.set_title(label, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Sweep family: {family}  (n_seeds shown as error bars)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--pattern", type=str,
        default="experiments/mappo_*_seed*_*",
        help="Glob for run dirs, relative to repo root.",
    )
    p.add_argument(
        "--split", type=str, default="test",
        choices=("train", "val", "test"),
        help="Split to evaluate on. 'test' for paper numbers.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Seed for env reset only (policy itself is deterministic).",
    )
    p.add_argument(
        "--out-csv", type=Path, default=None,
        help="Output CSV (default: research/phase8/results/sweep_results_long.csv).",
    )
    p.add_argument(
        "--out-figures-dir", type=Path, default=None,
        help="Where to write per-family PNGs (default: research/phase8/results/).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    dirs = sorted(REPO_ROOT.glob(args.pattern))
    if not dirs:
        print(f"No run dirs matched pattern {args.pattern!r} under {REPO_ROOT}")
        return 1
    print(f"Found {len(dirs)} candidate dirs")

    rows: list[dict] = []
    for d in dirs:
        m = CKPT_RE.match(d.name)
        if not m:
            print(f"  [SKIP] {d.name} — does not match mappo_<tag>_seed<n>_<ts>")
            continue
        tag = m.group("tag")
        seed = int(m.group("seed"))
        family = _family_of(tag)

        ckpt = d / "mappo_best.pt"
        if not ckpt.exists():
            ckpt = d / "mappo_final.pt"
        if not ckpt.exists():
            print(f"  [SKIP] {d.name} — no .pt found")
            continue

        cfg, overlay_found = _scenario_cfg_for_tag(tag)
        try:
            metrics = evaluate(ckpt, cfg, args.split, args.seed)
        except Exception as e:
            print(f"  [FAIL] {d.name} — {type(e).__name__}: {e}")
            continue
        row = {
            "tag": tag,
            "family": family,
            "train_seed": seed,
            "ckpt_path": str(ckpt.relative_to(REPO_ROOT)),
            "overlay_found": overlay_found,
            **metrics,
        }
        rows.append(row)
        print(
            f"  [OK]   {tag:<26} seed={seed:<3} "
            f"reward={metrics['total_reward_unweighted']:>+8.2f}  "
            f"unsafe={metrics['agg_unsafe_rate']:.2%}  "
            f"mean_p={metrics['mean_fleet_power_w']:>5.1f}W"
        )

    if not rows:
        print("No successful runs.")
        return 1

    df = pd.DataFrame(rows)
    out_csv = args.out_csv or (REPO_ROOT / "research" / "phase8" / "results" / "sweep_results_long.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    metric_cols = [
        c for c in df.columns if any(
            c.startswith(pref) for pref in
            ("total_", "agg_", "mean_", "p95_", "max_", "pool_")
        )
    ]
    summary = df.groupby(["family", "tag"])[metric_cols].agg(["mean", "std", "count"])
    summary_csv = out_csv.parent / (out_csv.stem + "_summary.csv")
    summary.to_csv(summary_csv)

    print()
    print(f"Long CSV:    {out_csv.relative_to(REPO_ROOT)}")
    print(f"Summary CSV: {summary_csv.relative_to(REPO_ROOT)}")
    print()
    print("=== Per-tag aggregates (mean across seeds) ===")
    headline = df.groupby(["family", "tag"]).agg(
        n_seeds=("train_seed", "count"),
        reward=("total_reward_unweighted", "mean"),
        energy_wh=("total_energy_wh", "mean"),
        unsafe=("agg_unsafe_rate", "mean"),
        switches=("agg_n_switches", "mean"),
        mean_power=("mean_fleet_power_w", "mean"),
    )
    print(headline.round(3).to_string())

    fig_dir = args.out_figures_dir or (REPO_ROOT / "research" / "phase8" / "results")
    fig_dir.mkdir(parents=True, exist_ok=True)
    print()
    for family in sorted(df["family"].unique()):
        sub = df[df["family"] == family]
        if sub.empty or len(sub["tag"].unique()) < 2:
            continue
        out_png = fig_dir / f"sweep_{family}_summary.png"
        _plot_family(sub, family, out_png)
        print(f"Figure:      {out_png.relative_to(REPO_ROOT)}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
