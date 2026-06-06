"""Phase 9 — Per-cluster MAPPO behavior characterization.

Joins the per-cluster KPIs from fleet_eval with the per-cluster traffic
statistics from heterogeneity_analysis, and produces a figure that shows
how the MAPPO policy adapts to each cluster's traffic profile.

Run heterogeneity_analysis.py and fleet_eval.py first.

Usage
-----
    python research/phase9/per_cluster_behavior.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "research" / "phase9" / "results"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    rd = args.results_dir

    het = pd.read_csv(rd / f"heterogeneity_{args.split}.csv")
    pc = pd.read_csv(rd / f"fleet_eval_{args.split}_per_cluster.csv")

    mappo = pc[pc.policy == "MAPPO"].copy()
    df = mappo.merge(het, on="cluster", how="left")
    df["mean_load_mbps"] = df["mean_gbps"] * 1000.0
    df["p95_load_mbps"] = df["p95_gbps"] * 1000.0
    df["usr_rate_pct"] = 100.0 - df["dpdk_rate_pct"]

    print("=== Per-cluster MAPPO behavior (joined with cluster heterogeneity) ===")
    cols = [
        "cluster", "mean_load_mbps", "p95_load_mbps", "peak_hour",
        "dpdk_rate_pct", "usr_rate_pct", "n_switches",
        "delay_compliance_pct", "p95_delay_us",
    ]
    print(df[cols].round(2).to_string(index=False))

    out_csv = rd / f"per_cluster_behavior_{args.split}.csv"
    df.to_csv(out_csv, index=False)

    # Figure: 4-panel per-cluster behavior.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=130)

    # 1. USR rate vs cluster mean load.
    ax = axes[0, 0]
    sc = ax.scatter(df.mean_load_mbps, df.usr_rate_pct, s=70,
                    c=df.cluster, cmap="viridis", edgecolor="black")
    for _, r in df.iterrows():
        ax.annotate(f"c{int(r.cluster)}", (r.mean_load_mbps, r.usr_rate_pct),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Cluster mean load (Mbps)")
    ax.set_ylabel("MAPPO USR rate (%)")
    ax.set_title("USR usage decreases as cluster load grows")
    ax.grid(True, alpha=0.3)

    # 2. Switches vs load variability.
    ax = axes[0, 1]
    ax.scatter(df.coeff_of_variation, df.n_switches, s=70,
               c=df.cluster, cmap="viridis", edgecolor="black")
    for _, r in df.iterrows():
        ax.annotate(f"c{int(r.cluster)}", (r.coeff_of_variation, r.n_switches),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Cluster traffic CoV (std/mean)")
    ax.set_ylabel("# mode switches over episode")
    ax.set_title("Switches scale with load variability")
    ax.grid(True, alpha=0.3)

    # 3. Bar plot: per-cluster DPDK vs USR shares.
    ax = axes[1, 0]
    x = np.arange(len(df))
    df_sorted = df.sort_values("mean_load_mbps").reset_index(drop=True)
    ax.bar(x, df_sorted.dpdk_rate_pct, color="#3b6db5",
           edgecolor="black", label="DPDK%")
    ax.bar(x, df_sorted.usr_rate_pct,
           bottom=df_sorted.dpdk_rate_pct, color="#e07b2f",
           edgecolor="black", label="USR%")
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{int(c)}" for c in df_sorted.cluster], fontsize=8)
    ax.set_ylabel("Mode share (%)")
    ax.set_title("Per-cluster mode share (clusters sorted by load)")
    ax.legend(loc="lower right")
    ax.set_ylim([0, 100])

    # 4. QoS metric per cluster.
    ax = axes[1, 1]
    ax.bar(x, df_sorted.delay_compliance_pct, color="#3a9a4c",
           edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{int(c)}" for c in df_sorted.cluster], fontsize=8)
    ax.set_ylabel("Delay budget compliance (%)")
    ax.set_title("Per-cluster QoS compliance")
    ax.axhline(100.0, color="black", linestyle="--", alpha=0.3)
    ax.set_ylim([95, 100.5])

    fig.tight_layout()
    fig_path = rd / f"per_cluster_behavior_{args.split}.png"
    fig.savefig(fig_path)
    plt.close(fig)

    print()
    print(f"CSV:    {out_csv}")
    print(f"Figure: {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
