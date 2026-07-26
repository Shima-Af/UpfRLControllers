"""Regenerate the controller chapter's figures under the v0.4 twin, with text
rendered in the paper's Latin Modern font (matplotlib usetex + lmodern).

Covers the four data-driven figures (from the authoritative v0.4 summary
multiseed_summary_v04twin.json):
  fig_p7_reward_errorbars.png  — fleet reward mean±std per controller
  fig_bridge_ippo_vs_mappo.pdf — per-cluster IPPO vs MAPPO (transfer story)
  fig_p7_safety_vs_reward.png  — safety (unsafe %) vs reward, per seed
  fig_p2_singlesite_reward.png — single-site (cluster 0): PPO vs DPDK vs threshold

The two rollout-based heatmaps are produced by generate_chapter_heatmaps.py.

    python research/phase7/generate_chapter_figures.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY = REPO_ROOT / "reports/phase-7/multiseed_summary_v04twin.json"
OUT = REPO_ROOT / "reports/chapter-controllers/figures"

# House palette (kept for continuity across the thesis figures)
RED, GREEN, BLUE = "#ef4444", "#22c55e", "#0ea5e9"
PURPLE, EMER, PINK = "#a855f7", "#10b981", "#ec4899"
INK, MUTE, GRID, HILITE = "#1e293b", "#64748b", "#e2e8f0", "#f8d34e"


def _style() -> None:
    """Match the chapters' Latin Modern body font via usetex."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{lmodern}",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
    })


def _clean_axes(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cbd5e1")
    ax.tick_params(colors=MUTE)


def _load():
    return json.loads(SUMMARY.read_text())["aggregated"]


def _seed_rewards(agg, ctrl):
    return [s["total_reward_unweighted"] for s in agg[ctrl]["per_seed"].values()]


def _c0(agg, ctrl, field="per_cluster_total_reward"):
    return st.mean(s[field][0] for s in agg[ctrl]["per_seed"].values())


# Display labels (usetex-safe: no raw underscores)
LABELS = {
    "MAPPO": "MAPPO",
    "IPPO-ensemble": "IPPO ensemble",
    "centralised-PPO": "Centralised PPO",
    "always-DPDK": "Always DPDK",
    "threshold(derived=81.0Mbps)": "Threshold (81 Mbps)",
    "hysteresis(t_up=81,t_down=0,cd=1)": "Hysteresis (auto band)",
    "hysteresis(t_up=81,t_down=61,cd=1)": "Hysteresis (band 20)",
}
COLORS = {
    "MAPPO": RED, "IPPO-ensemble": GREEN, "centralised-PPO": BLUE,
    "always-DPDK": EMER, "threshold(derived=81.0Mbps)": PINK,
    "hysteresis(t_up=81,t_down=0,cd=1)": "#8b5cf6",
    "hysteresis(t_up=81,t_down=61,cd=1)": PURPLE,
}


def fig_reward_errorbars(agg) -> None:
    order = ["MAPPO", "IPPO-ensemble", "hysteresis(t_up=81,t_down=61,cd=1)",
             "always-DPDK", "threshold(derived=81.0Mbps)", "centralised-PPO"]
    means, stds, labels, colors = [], [], [], []
    for c in order:
        rs = _seed_rewards(agg, c)
        means.append(st.mean(rs)); stds.append(st.pstdev(rs))
        labels.append(LABELS[c]); colors.append(COLORS[c])
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    ax.barh(y, means, xerr=stds, color=colors, edgecolor="white",
            error_kw={"ecolor": INK, "capsize": 3, "linewidth": 1.0})
    for yi, m, s in zip(y, means, stds):
        ax.text(m, yi, f"  {m:,.0f}", va="center", ha="right" if m < -2000 else "left",
                color="white" if m < -2000 else INK, fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel(r"test-period total reward (mean $\pm$ std over seeds; higher is better)")
    ax.set_title(r"Controller reward under the corrected (v0.4) twin")
    ax.axvline(0, color="#cbd5e1", linewidth=0.8)
    _clean_axes(ax)
    fig.tight_layout(); fig.savefig(OUT / "fig_p7_reward_errorbars.png", bbox_inches="tight")
    plt.close(fig)


def fig_safety_vs_reward(agg) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for c in ("MAPPO", "IPPO-ensemble", "centralised-PPO"):
        rs = _seed_rewards(agg, c)
        us = [s["agg_unsafe_rate"] * 100 for s in agg[c]["per_seed"].values()]
        ax.scatter(rs, us, s=70, color=COLORS[c], edgecolors="white",
                   linewidths=1.0, label=LABELS[c], zorder=3)
    for c in ("always-DPDK", "threshold(derived=81.0Mbps)",
              "hysteresis(t_up=81,t_down=61,cd=1)"):
        r = st.mean(_seed_rewards(agg, c))
        u = st.mean(s["agg_unsafe_rate"] * 100 for s in agg[c]["per_seed"].values())
        ax.scatter([r], [u], s=90, marker="D", color=COLORS[c],
                   edgecolors="white", linewidths=1.0, label=LABELS[c], zorder=3)
    ax.set_xlabel(r"test-period total reward (higher is better)")
    ax.set_ylabel(r"QoS-violation rate (\%; lower is better)")
    ax.set_title(r"Safety versus reward --- top-left is ideal")
    ax.grid(True, color=GRID, linewidth=0.7); ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.legend(frameon=False, loc="upper left", ncol=1)
    fig.tight_layout(); fig.savefig(OUT / "fig_p7_safety_vs_reward.png", bbox_inches="tight")
    plt.close(fig)


def fig_singlesite(agg) -> None:
    rows = [("Single-site PPO", _c0(agg, "IPPO-ensemble"), RED),
            ("Always DPDK", _c0(agg, "always-DPDK"), EMER),
            ("Threshold (81 Mbps)", _c0(agg, "threshold(derived=81.0Mbps)"), PINK)]
    labels = [r[0] for r in rows]; vals = [r[1] for r in rows]; cols = [r[2] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.bar(x, vals, 0.6, color=cols, edgecolor="white")
    for xi, v in zip(x, vals):
        ax.text(xi, v, f"{v:,.0f}", ha="center", va="top", color="white", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r"cluster-0 total reward (higher is better)")
    ax.set_title(r"Single-site control on a low-load cluster (v0.4)")
    _clean_axes(ax)
    fig.tight_layout(); fig.savefig(OUT / "fig_p2_singlesite_reward.png", bbox_inches="tight")
    plt.close(fig)


def fig_bridge(agg) -> None:
    ippo = np.array([np.mean([s["per_cluster_total_reward"][k]
                    for s in agg["IPPO-ensemble"]["per_seed"].values()]) for k in range(10)])
    mappo = np.array([np.mean([s["per_cluster_total_reward"][k]
                     for s in agg["MAPPO"]["per_seed"].values()]) for k in range(10)])
    delta = mappo - ippo
    transfer = {k for k in range(10) if delta[k] > 15.0}
    x = np.arange(10); w = 0.38
    fig, ax = plt.subplots(figsize=(9.0, 3.7))
    for k in transfer:
        ax.axvspan(k - 0.5, k + 0.5, color=HILITE, alpha=0.18, zorder=0)
    ax.bar(x - w / 2, ippo, w, label="IPPO ensemble (independent per-site)",
           color=GREEN, edgecolor="white", zorder=3)
    ax.bar(x + w / 2, mappo, w, label="MAPPO (cooperative; shared actor, central critic)",
           color=RED, edgecolor="white", zorder=3)
    for k in transfer:
        ax.annotate(f"$+${delta[k]:.0f}", (k, max(ippo[k], mappo[k])),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=7.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([f"c{k}" for k in range(10)])
    ax.set_xlabel(r"MEC cluster")
    ax.set_ylabel(r"mean test reward over 8 seeds")
    ax.set_title(r"Per-cluster reward: independent single-site learning vs cooperative MAPPO")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.02))
    ax.text(0.0, -0.26, r"Shaded: MAPPO $>$ ensemble by $>15$ reward units --- "
            r"data-poor sites that inherit the fleet's operating rule.",
            transform=ax.transAxes, fontsize=7.5, color=MUTE)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_bridge_ippo_vs_mappo.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"bridge transfer clusters: {sorted(transfer)}")


def main() -> int:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    agg = _load()
    fig_reward_errorbars(agg)
    fig_safety_vs_reward(agg)
    fig_singlesite(agg)
    fig_bridge(agg)
    print(f"wrote 4 figures to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
