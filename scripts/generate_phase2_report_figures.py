"""Generate the static figures for reports/phase-2/.

Runs all five policies on cluster 0 (full episode, deterministic seed)
and emits four PNG figures into ``reports/phase-2/figures/``:

  fig1_load_profile.png      Cluster 0's actual load over the episode.
  fig2_total_reward.png      Per-policy total-reward bar chart.
  fig3_action_timelines.png  Load + PPO / Threshold / Always-DPDK action rows.
  fig4_cumulative_reward.png One curve per policy.

Re-run after retraining PPO or after any reward-shape change to keep
the supervisor report in sync with the code. Figures land in a
gitignore-friendly location (reports/phase-2/figures/ tracked in git).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.backend.app.policies import (  # noqa: E402
    POLICY_LABELS,
    PolicyRegistry,
    run_rollout,
)

OUT_DIR = REPO_ROOT / "reports" / "phase-2" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLICY_COLOR = {
    "ppo": "#0ea5e9",
    "random": "#94a3b8",
    "always-dpdk": "#10b981",
    "always-usr": "#f59e0b",
    "threshold": "#a855f7",
}

# Order in which to draw policies (best -> worst, roughly).
POLICY_ORDER = ["ppo", "threshold", "always-dpdk", "random", "always-usr"]


def _setup_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, color="#1e293b", loc="left", pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color="#475569")
    ax.set_ylabel(ylabel, fontsize=9, color="#475569")
    ax.tick_params(labelsize=8, colors="#64748b")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#cbd5e1")
    ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.5)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test"],
        help=(
            "Forecaster slice to evaluate on. The supervisor-facing "
            "headline numbers should use 'test' (default). 'val' is "
            "useful for debugging checkpoint selection."
        ),
    )
    p.add_argument("--cluster-idx", type=int, default=0)
    p.add_argument("--horizon-idx", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    print(f"Generating Phase 2 figures in {OUT_DIR} (split={args.split!r})")

    registry = PolicyRegistry(repo_root=REPO_ROOT)

    print(
        f"Running rollouts (cluster_idx={args.cluster_idx}, "
        f"horizon_idx={args.horizon_idx}, full episode, "
        f"split={args.split!r})..."
    )
    rollouts = {}
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
            seed=42,
            threshold_gbps=0.05,
            max_steps=None,
            split=args.split,
        )
        rollouts[pid] = r
        print(
            f"  {info.label:<18s} total_r={r.summary.total_reward:8.2f}  "
            f"energy_Wh={r.summary.total_energy_wh:6.2f}  "
            f"unsafe={r.summary.unsafe_rate:.3f}"
        )

    # ------------------------------------------------------------------
    # Figure 1 — cluster 0 load profile
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 2.6), dpi=150)
    any_rollout = next(iter(rollouts.values()))
    t = np.array([s.t for s in any_rollout.steps])
    load = np.array([s.actual_load_gbps for s in any_rollout.steps])
    ax.plot(t, load, color="#334155", linewidth=0.8)
    ax.axhline(0.05, color="#a855f7", linestyle="--", linewidth=0.8, alpha=0.7)
    # Place threshold annotation in a corner that doesn't overlap the data.
    ax.text(
        0.01, 0.93, "USR threshold = 0.05 Gbps",
        transform=ax.transAxes,
        color="#a855f7", fontsize=8, va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
    )
    _setup_axes(
        ax,
        f"Cluster {args.cluster_idx} — offered load over the "
        f"{args.split} episode ({len(t)} steps)",
        "timestep",
        "load (Gbps)",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_load_profile.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig1_load_profile.png")

    # ------------------------------------------------------------------
    # Figure 2 — total reward bar chart
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=150)
    pids = list(rollouts.keys())
    labels = [POLICY_LABELS[pid] for pid in pids]
    values = [rollouts[pid].summary.total_reward for pid in pids]
    colors = [POLICY_COLOR[pid] for pid in pids]
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=1.0)
    # Annotate values just inside each bar's left tip, extending into the bar.
    for bar, v in zip(bars, values, strict=True):
        ax.text(
            v + 8, bar.get_y() + bar.get_height() / 2,
            f"{v:+.2f}",
            color="white", fontsize=9, fontweight="bold",
            va="center", ha="left",
        )
    ax.invert_yaxis()  # best (least negative) at top
    _setup_axes(
        ax,
        f"Total reward per policy on cluster {args.cluster_idx} "
        f"({args.split} split, full episode)",
        "total reward (closer to 0 is better)",
        "",
    )
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_total_reward.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig2_total_reward.png")

    # ------------------------------------------------------------------
    # Figure 3 — action timelines for PPO, threshold, always-DPDK
    # ------------------------------------------------------------------
    panels = [("ppo", "PPO (trained)"),
              ("threshold", "Threshold (USR < 0.05 Gbps)"),
              ("always-dpdk", "Always DPDK")]
    panels = [p for p in panels if p[0] in rollouts]
    fig, axes = plt.subplots(
        nrows=len(panels) + 1, ncols=1,
        figsize=(8.5, 1.0 + 0.85 * (len(panels) + 1)),
        dpi=150, sharex=True,
    )
    # Top: load
    axes[0].plot(t, load, color="#334155", linewidth=0.8)
    axes[0].axhline(
        0.05, color="#a855f7", linestyle="--", linewidth=0.7, alpha=0.7,
    )
    _setup_axes(
        axes[0],
        "Offered load + per-policy action timeline (red = USR, gray = DPDK)",
        "",
        "Gbps",
    )
    # One row per policy: scatter of action values
    for ax, (pid, label) in zip(axes[1:], panels, strict=True):
        actions = np.array([s.action for s in rollouts[pid].steps])
        usr_mask = actions == 1
        dpdk_mask = actions == 0
        ax.scatter(
            t[dpdk_mask], np.zeros(dpdk_mask.sum()),
            s=2, color="#94a3b8", alpha=0.5,
        )
        ax.scatter(
            t[usr_mask], np.ones(usr_mask.sum()),
            s=2, color="#ef4444", alpha=0.8,
        )
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["DPDK", "USR"], fontsize=8)
        ax.set_ylim(-0.4, 1.4)
        # USR usage rate annotation
        rate = usr_mask.mean()
        ax.text(
            0.995, 0.5, f"{label} — USR {100*rate:.1f}%",
            transform=ax.transAxes, fontsize=9, color="#475569",
            ha="right", va="center",
        )
        _setup_axes(ax, "", "", "")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.4)
    axes[-1].set_xlabel("timestep", fontsize=9, color="#475569")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_action_timelines.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig3_action_timelines.png")

    # ------------------------------------------------------------------
    # Figure 4 — cumulative reward curves
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=150)
    for pid in POLICY_ORDER:
        if pid not in rollouts:
            continue
        cum = np.array([s.cumulative_reward for s in rollouts[pid].steps])
        ax.plot(
            t, cum,
            color=POLICY_COLOR[pid],
            linewidth=1.4,
            label=POLICY_LABELS[pid],
        )
    _setup_axes(
        ax,
        "Cumulative reward over the episode",
        "timestep",
        "cumulative reward",
    )
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_cumulative_reward.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig4_cumulative_reward.png")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
