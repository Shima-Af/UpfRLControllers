"""Phase 9 — Cluster heterogeneity characterization.

Loads the per-cluster traffic traces (NetMob, K=10) for the test split,
computes per-cluster statistics (mean, p95, peak hour, peak-to-trough,
diurnal shape), and produces a CSV + a multi-panel figure suitable for
the paper's "fleet operates over heterogeneous MEC sites" section.

Usage
-----
    python research/phase9/heterogeneity_analysis.py
    python research/phase9/heterogeneity_analysis.py --split test --out-csv ...
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
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_yaml  # noqa: E402

STEPS_PER_HOUR = 4  # 15-min control steps


def _load_traces(split: str) -> tuple[np.ndarray, float]:
    """Return (targets[T, H, K], alpha) for the requested split."""
    paths_cfg = load_yaml(REPO_ROOT / "configs" / "digital_twin_paths.yaml")
    scenario_cfg = load_yaml(REPO_ROOT / "configs" / "scenario_rl.yaml")
    tf = paths_cfg.get("traffic_forecaster", {})
    rel = tf.get(f"targets_{split}", f"data/external/traffic_forecaster/targets_{split}.npy")
    targets = np.load(REPO_ROOT / rel)
    alpha = float(
        tf.get("alpha")
        or scenario_cfg.get("traffic", {}).get("alpha")
        or 1.0
    )
    return targets * alpha, alpha


def _per_cluster_stats(loads_gbps: np.ndarray) -> pd.DataFrame:
    """loads_gbps: shape (T, K). Returns one row per cluster."""
    T, K = loads_gbps.shape
    rows = []
    for k in range(K):
        x = loads_gbps[:, k]
        # Diurnal shape: average load by hour-of-day (assuming step=15min).
        hours = (np.arange(T) // STEPS_PER_HOUR) % 24
        hourly = np.array([x[hours == h].mean() for h in range(24)])
        peak_hour = int(np.argmax(hourly))
        trough_hour = int(np.argmin(hourly))
        rows.append({
            "cluster": k,
            "mean_gbps": float(x.mean()),
            "p50_gbps": float(np.percentile(x, 50)),
            "p95_gbps": float(np.percentile(x, 95)),
            "max_gbps": float(x.max()),
            "min_gbps": float(x.min()),
            "peak_hour": peak_hour,
            "trough_hour": trough_hour,
            "peak_to_trough_ratio": float(hourly.max() / max(hourly.min(), 1e-9)),
            "coeff_of_variation": float(x.std() / max(x.mean(), 1e-9)),
        })
    return pd.DataFrame(rows)


def _plot_heterogeneity(loads_gbps: np.ndarray, out_path: Path) -> None:
    T, K = loads_gbps.shape
    hours = (np.arange(T) // STEPS_PER_HOUR) % 24
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=130)

    # Left: diurnal profile per cluster (mean by hour-of-day).
    ax = axes[0]
    for k in range(K):
        hourly = np.array([loads_gbps[hours == h, k].mean() for h in range(24)])
        ax.plot(range(24), hourly * 1000.0, alpha=0.75, lw=1.0, label=f"c{k}")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean load (Mbps)")
    ax.set_title("Per-cluster diurnal profile")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=2, loc="upper left")

    # Middle: load CDF per cluster.
    ax = axes[1]
    for k in range(K):
        sorted_x = np.sort(loads_gbps[:, k]) * 1000.0
        cdf = np.linspace(0, 1, len(sorted_x))
        ax.plot(sorted_x, cdf, lw=1.0, alpha=0.75, label=f"c{k}")
    ax.set_xlabel("Load (Mbps)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Per-cluster load CDF")
    ax.grid(True, alpha=0.3)

    # Right: scatter of mean vs p95 load (one point per cluster).
    ax = axes[2]
    means = loads_gbps.mean(axis=0) * 1000.0
    p95s = np.percentile(loads_gbps, 95, axis=0) * 1000.0
    ax.scatter(means, p95s, s=60, c=range(K), cmap="viridis", edgecolor="black")
    for k in range(K):
        ax.annotate(f"c{k}", (means[k], p95s[k]), fontsize=8, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("Mean load (Mbps)")
    ax.set_ylabel("p95 load (Mbps)")
    ax.set_title("Mean vs peak load per cluster")
    ax.grid(True, alpha=0.3)
    ax.plot([0, max(p95s)], [0, max(p95s)], "k--", alpha=0.3, lw=0.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", type=str, default="test",
                   choices=("train", "val", "test"))
    p.add_argument("--horizon-idx", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    targets, alpha = _load_traces(args.split)
    loads = targets[:, args.horizon_idx, :]  # (T, K) in Gbps

    out_dir = args.out_dir or (REPO_ROOT / "research" / "phase9" / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Split: {args.split}, alpha={alpha}, shape={loads.shape}")
    df = _per_cluster_stats(loads)

    csv_path = out_dir / f"heterogeneity_{args.split}.csv"
    df.to_csv(csv_path, index=False)

    fig_path = out_dir / f"heterogeneity_{args.split}.png"
    _plot_heterogeneity(loads, fig_path)

    print()
    print("=== Per-cluster statistics ===")
    print(df.round(4).to_string(index=False))
    print()
    print(f"CSV:    {csv_path.relative_to(REPO_ROOT)}")
    print(f"Figure: {fig_path.relative_to(REPO_ROOT)}")

    # Fleet-level diversity summary.
    print()
    print("=== Fleet-level diversity ===")
    print(f"Mean-load range (Mbps): [{df.mean_gbps.min()*1000:.1f}, "
          f"{df.mean_gbps.max()*1000:.1f}]  "
          f"ratio={df.mean_gbps.max()/df.mean_gbps.min():.2f}x")
    print(f"p95-load range  (Mbps): [{df.p95_gbps.min()*1000:.1f}, "
          f"{df.p95_gbps.max()*1000:.1f}]  "
          f"ratio={df.p95_gbps.max()/df.p95_gbps.min():.2f}x")
    print(f"Peak-hour spread (h):  {df.peak_hour.min()}-{df.peak_hour.max()}  "
          f"(distinct peaks: {df.peak_hour.nunique()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
