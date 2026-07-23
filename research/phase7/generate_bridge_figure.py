"""D2 — bridge figure: independent single-site learners (IPPO ensemble) vs the
cooperative fleet controller (MAPPO), per cluster, on one axis.

Phase-2/3b single-site PPO and Phase-6/7 MAPPO otherwise live in separate
tables; this figure puts them side by side per MEC cluster so the "single-site
ceiling → cooperation" argument is visible in one plot. The win concentrates on
the mid-load, data-poor clusters (c5, c7, c8) where an independent learner
cannot see the switching boundary often enough to learn it, but MAPPO's shared
actor transfers the rule in from data-rich clusters.

Reads the post-fix authoritative summary (v0.4.0 twin). Output: PDF + PNG.

    python research/phase7/generate_bridge_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY = REPO_ROOT / "reports/phase-7/multiseed_summary_v04twin.json"
OUT_DIR = REPO_ROOT / "reports/phase-7/figures"

MAPPO_RED = "#ef4444"
IPPO_GREEN = "#22c55e"
INK = "#1e293b"
MUTE = "#64748b"
GRID = "#e2e8f0"
HILITE = "#f8d34e"  # transfer-win clusters


def _per_cluster_mean(summary: dict, controller: str) -> np.ndarray:
    """Mean per-cluster reward across seeds for one controller."""
    ps = summary["aggregated"][controller]["per_seed"]
    mat = np.array([s["per_cluster_total_reward"] for s in ps.values()])
    return mat.mean(axis=0)


def main() -> int:
    if not SUMMARY.exists():
        raise SystemExit(f"missing {SUMMARY} — run evaluate_multiseed.py first")
    summary = json.loads(SUMMARY.read_text())

    ippo = _per_cluster_mean(summary, "IPPO-ensemble")
    mappo = _per_cluster_mean(summary, "MAPPO")
    K = len(mappo)
    x = np.arange(K)
    w = 0.38

    # clusters where MAPPO meaningfully beats the independent learner
    delta = mappo - ippo
    transfer = {k for k in range(K) if delta[k] > 15.0}

    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=150)

    for k in transfer:
        ax.axvspan(k - 0.5, k + 0.5, color=HILITE, alpha=0.18, zorder=0)

    ax.bar(x - w / 2, ippo, w, label="IPPO ensemble (independent per-site)",
           color=IPPO_GREEN, edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(x + w / 2, mappo, w, label="MAPPO (cooperative, shared actor + central critic)",
           color=MAPPO_RED, edgecolor="white", linewidth=0.8, zorder=3)

    for k in transfer:
        top = max(ippo[k], mappo[k])
        ax.annotate(f"+{delta[k]:.0f}", (k, top), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7.5,
                    color=INK, fontweight="bold")

    ax.set_title("Per-cluster reward: independent single-site learning vs cooperative MAPPO",
                 fontsize=11, color=INK, loc="left", pad=8)
    ax.set_xlabel("MEC cluster (ordered as evaluated)", fontsize=9, color=MUTE)
    ax.set_ylabel("mean test reward over 8 seeds\n(higher = better)", fontsize=9, color=MUTE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{k}" for k in range(K)], fontsize=8)
    ax.tick_params(labelsize=8, colors=MUTE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cbd5e1")
    ax.legend(fontsize=8, frameon=False, loc="lower left",
              bbox_to_anchor=(0.20, 0.03))

    # caption-style note of the shaded band meaning
    ax.text(0.0, -0.28, "Shaded clusters: MAPPO > ensemble by >15 reward units — "
            "data-poor sites that inherit the operating rule from the fleet.",
            transform=ax.transAxes, fontsize=7.5, color=MUTE)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_bridge_ippo_vs_mappo.{ext}", bbox_inches="tight")
    print(f"wrote {OUT_DIR}/fig_bridge_ippo_vs_mappo.{{pdf,png}}")
    print(f"transfer-win clusters (Δ>15): {sorted(transfer)}  Δ={np.round(delta,1)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
