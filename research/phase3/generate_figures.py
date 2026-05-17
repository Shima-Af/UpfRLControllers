"""Generate Phase 3 supervisor-facing figures.

Consumes the latest centralised PPO checkpoint plus (optionally) a
per-cluster ensemble directory, runs all comparison policies on the
test slice, and writes PNGs under reports/phase-3/figures/.

Figures:
  fig1_cluster_loads.png        Per-cluster mean / p95 / max load on test.
  fig2_total_reward.png         Weighted total reward per policy.
  fig3_per_cluster_reward.png   Grouped bar: per-cluster reward across
                                centralised PPO vs ensemble vs all-DPDK.
  fig4_action_heatmap.png       Cluster x time action matrix for the
                                centralised PPO (red = USR, gray = DPDK).
  fig5_cumulative_reward.png    Weighted cumulative reward over the
                                episode, one curve per policy.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.envs.multi_site_upf_env import MultiSiteUPFEnv  # noqa: E402
from src.trainers.ppo_multi_site import (  # noqa: E402
    constant_multi_policy,
    ppo_multi_policy,
    predicted_load_threshold_multi_policy,
)

OUT_DIR = REPO_ROOT / "reports" / "phase-3" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLICY_COLOR = {
    "multi-PPO": "#0ea5e9",
    "multi-PPO-ensemble": "#22c55e",
    "all-DPDK": "#10b981",
    "threshold(0.050)": "#a855f7",
}
POLICY_ORDER = [
    "multi-PPO", "multi-PPO-ensemble", "threshold(0.050)", "all-DPDK"
]


def _setup_axes(ax, title, xlabel, ylabel) -> None:
    ax.set_title(title, fontsize=11, color="#1e293b", loc="left", pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color="#475569")
    ax.set_ylabel(ylabel, fontsize=9, color="#475569")
    ax.tick_params(labelsize=8, colors="#64748b")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#cbd5e1")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)


def _latest(glob: str) -> Path | None:
    matches = sorted(
        (REPO_ROOT / "experiments").glob(glob),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _ensemble_policy(ckpts: list[Path], K: int):
    models = [PPO.load(p) for p in ckpts]

    def _fn(obs: np.ndarray) -> np.ndarray:
        out = np.zeros(K, dtype=np.int64)
        for k in range(K):
            sub = obs[k * 14: (k + 1) * 14]
            a, _ = models[k].predict(sub, deterministic=True)
            out[k] = int(np.asarray(a).item())
        return out

    return _fn


def _rollout_collect(policy_fn, *, split: str, seed: int):
    """Run one episode and return time-indexed action/reward/load arrays."""
    env = MultiSiteUPFEnv(split=split)
    obs, _ = env.reset(seed=seed)
    K = env.K
    n_steps = env._N  # noqa: SLF001
    actions = np.zeros((n_steps, K), dtype=np.int8)
    rewards = np.zeros(n_steps, dtype=np.float64)
    cum = np.zeros(n_steps, dtype=np.float64)
    per_cluster_reward = np.zeros((n_steps, K), dtype=np.float64)
    loads = np.zeros((n_steps, K), dtype=np.float64)
    cum_r = 0.0
    t = 0
    while True:
        a = policy_fn(obs)
        obs, r, term, trunc, info = env.step(a)
        actions[t] = a
        rewards[t] = r
        cum_r += r
        cum[t] = cum_r
        per_cluster_reward[t] = info["per_cluster_reward"]
        loads[t] = info["per_cluster_load_gbps"]
        t += 1
        if term or trunc:
            break
    return {
        "K": K,
        "n_steps": t,
        "actions": actions[:t],
        "rewards": rewards[:t],
        "cum": cum[:t],
        "per_cluster_reward": per_cluster_reward[:t],
        "loads": loads[:t],
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--multi-ckpt", type=Path, default=None,
        help="Centralised PPO checkpoint. Defaults to most recent.",
    )
    p.add_argument(
        "--ensemble-dir", type=Path, default=None,
        help="Ensemble dir holding cluster_<k>/ppo_single_site.zip. "
             "Defaults to most recent.",
    )
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    print(f"Generating Phase 3 figures in {OUT_DIR} (split={args.split!r})")

    K = 10
    multi_ckpt = args.multi_ckpt
    if multi_ckpt is None:
        latest = _latest("ppo_multi_site_*")
        if latest is not None:
            multi_ckpt = latest / "ppo_multi_site.zip"
    if multi_ckpt is None or not multi_ckpt.exists():
        raise SystemExit(
            "No centralised PPO checkpoint found. Train one first."
        )

    ensemble_dir = args.ensemble_dir
    if ensemble_dir is None:
        ensemble_dir = _latest("ppo_single_site_ensemble_*")
    ensemble_ckpts: list[Path] = []
    if ensemble_dir is not None and ensemble_dir.is_dir():
        ok = True
        for k in range(K):
            cand = ensemble_dir / f"cluster_{k}" / "ppo_single_site.zip"
            if not cand.exists():
                ok = False
                break
            ensemble_ckpts.append(cand)
        if not ok:
            print("  ensemble incomplete, skipping ensemble policy")
            ensemble_ckpts = []

    centralised = PPO.load(multi_ckpt)

    policies = {
        "multi-PPO": ppo_multi_policy(centralised, deterministic=True),
        "threshold(0.050)":
            predicted_load_threshold_multi_policy(0.05, K=K),
        "all-DPDK": constant_multi_policy(0, K=K),
    }
    if ensemble_ckpts:
        policies["multi-PPO-ensemble"] = _ensemble_policy(ensemble_ckpts, K=K)

    print("Running rollouts...")
    rolls: dict[str, dict] = {}
    for name in POLICY_ORDER:
        if name not in policies:
            continue
        rolls[name] = _rollout_collect(
            policies[name], split=args.split, seed=args.seed
        )
        cum = rolls[name]["cum"][-1]
        print(f"  {name:<22s} cum_weighted_r={cum:8.2f}")

    n_steps = next(iter(rolls.values()))["n_steps"]
    t = np.arange(n_steps)
    sample = next(iter(rolls.values()))
    loads = sample["loads"]  # (T, K)

    # ------------------------------------------------------------------
    # fig1 — per-cluster load profile (mean / p95 / max)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 3.0), dpi=150)
    means = loads.mean(axis=0)
    p95 = np.percentile(loads, 95, axis=0)
    maxes = loads.max(axis=0)
    x = np.arange(K)
    w = 0.27
    ax.bar(x - w, means, width=w, color="#0ea5e9", label="mean")
    ax.bar(x, p95, width=w, color="#a855f7", label="p95")
    ax.bar(x + w, maxes, width=w, color="#f59e0b", label="max")
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{k}" for k in range(K)])
    _setup_axes(
        ax,
        f"Per-cluster offered load on the {args.split} slice ({n_steps} steps)",
        "cluster", "load (Gbps)",
    )
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_cluster_loads.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig1_cluster_loads.png")

    # ------------------------------------------------------------------
    # fig2 — weighted total reward bar chart
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 2.6), dpi=150)
    labels = [n for n in POLICY_ORDER if n in rolls]
    values = [rolls[n]["cum"][-1] for n in labels]
    colors = [POLICY_COLOR[n] for n in labels]
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=1.0)
    for bar, v in zip(bars, values, strict=True):
        ax.text(
            v + 5, bar.get_y() + bar.get_height() / 2,
            f"{v:+.2f}",
            color="white", fontsize=9, fontweight="bold",
            va="center", ha="left",
        )
    ax.invert_yaxis()
    _setup_axes(
        ax,
        f"Weighted total reward per policy ({args.split} slice, K={K})",
        "total weighted reward (closer to 0 is better)", "",
    )
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_total_reward.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig2_total_reward.png")

    # ------------------------------------------------------------------
    # fig3 — per-cluster reward, grouped bar
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 3.4), dpi=150)
    policies_in_fig = [n for n in POLICY_ORDER if n in rolls]
    n_p = len(policies_in_fig)
    w = 0.8 / n_p
    x = np.arange(K)
    for i, name in enumerate(policies_in_fig):
        per_c = rolls[name]["per_cluster_reward"].sum(axis=0)
        offset = (i - (n_p - 1) / 2) * w
        ax.bar(x + offset, per_c, width=w, color=POLICY_COLOR[name], label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{k}" for k in range(K)])
    _setup_axes(
        ax,
        f"Per-cluster total reward (unweighted, {args.split} slice)",
        "cluster", "summed reward over episode",
    )
    ax.legend(fontsize=8, frameon=False, ncol=n_p)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_per_cluster_reward.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig3_per_cluster_reward.png")

    # ------------------------------------------------------------------
    # fig4 — action heatmap for the centralised PPO
    # ------------------------------------------------------------------
    actions = rolls["multi-PPO"]["actions"]  # (T, K), 0=DPDK 1=USR
    fig, ax = plt.subplots(figsize=(9.0, 3.6), dpi=150)
    im = ax.imshow(
        actions.T,
        aspect="auto",
        cmap="RdYlGn_r",  # green=DPDK(0), red=USR(1)
        interpolation="nearest",
        vmin=0, vmax=1,
    )
    ax.set_yticks(np.arange(K))
    ax.set_yticklabels([f"c{k}" for k in range(K)])
    _setup_axes(
        ax,
        f"Centralised PPO action over time ({args.split} slice; "
        f"green = DPDK, red = USR)",
        "timestep", "cluster",
    )
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=[0, 1])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_action_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig4_action_heatmap.png")

    # ------------------------------------------------------------------
    # fig5 — cumulative weighted reward
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.0, 3.6), dpi=150)
    for name in policies_in_fig:
        ax.plot(
            t, rolls[name]["cum"],
            color=POLICY_COLOR[name], linewidth=1.4, label=name,
        )
    _setup_axes(
        ax,
        f"Cumulative weighted reward over the {args.split} episode",
        "timestep", "cumulative reward",
    )
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_cumulative_reward.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig5_cumulative_reward.png")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
