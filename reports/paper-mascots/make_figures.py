"""Generate the two results figures for the MASCOTS 2026 paper.

fig_bootstrap.pdf  (paper Fig. 3)
    (a) Per-seed paired test reward, MAPPO vs IPPO ensemble (8 seeds).
    (b) Paired-bootstrap distribution (10 000 resamples) of the mean
        seed-matched MAPPO - IPPO delta, with the 95 % CI shaded.

fig_per_cluster.pdf  (paper Fig. 4)
    Per-cluster mean test reward (8 seeds) for always-DPDK, the IPPO
    ensemble, and MAPPO, with clusters sorted by mean offered load.
    Shows that MAPPO's advantage over IPPO concentrates on the
    low- and mid-load clusters.

All numbers are read from the Phase-7 evaluation artefacts; nothing is
hand-entered. Run:  python reports/paper-mascots/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE7 = REPO_ROOT / "reports" / "phase-7"
HET_CSV = REPO_ROOT / "research" / "phase9" / "results" / "heterogeneity_test.csv"
OUT = Path(__file__).resolve().parent / "figures"

# IEEE-friendly defaults: Times-like serif (STIX matches the IEEEtran
# Times body font), bold titles, modest sizes, vector output.
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
})

C_MAPPO = "#b3242b"   # red
C_IPPO = "#2f5c9e"    # blue
C_DPDK = "#7a7a7a"    # grey


def _load() -> dict:
    summ = json.loads((PHASE7 / "multiseed_summary_n8.json").read_text())
    boot = json.loads((PHASE7 / "paired_bootstrap_mappo_vs_ippo.json").read_text())
    return {"summary": summ, "boot": boot}


def _per_seed_rewards(summary: dict, key: str, seeds: list[str]) -> np.ndarray:
    ps = summary["aggregated"][key]["per_seed"]
    return np.array([ps[s]["total_reward_unweighted"] for s in seeds])


def _per_cluster_mean(summary: dict, key: str) -> np.ndarray:
    """Mean per-cluster reward across all seeds for a controller."""
    ps = summary["aggregated"][key]["per_seed"]
    mat = np.array([ps[s]["per_cluster_total_reward"] for s in ps])  # (n_seeds, K)
    return mat.mean(axis=0)


def fig_bootstrap(data: dict) -> None:
    summary, boot = data["summary"], data["boot"]
    seeds = [str(s) for s in boot["common_seeds"]]
    mappo = _per_seed_rewards(summary, "MAPPO", seeds)
    ippo = _per_seed_rewards(summary, "IPPO-ensemble", seeds)
    deltas = np.array(boot["per_seed_delta"])

    # Re-run the paired bootstrap for the histogram shape; report the
    # canonical CI persisted in the Phase-7 artefact so the figure and
    # the paper text agree exactly.
    rng = np.random.default_rng(0)
    n = len(deltas)
    resamples = rng.integers(0, n, size=(boot["n_bootstrap"], n))
    boot_means = deltas[resamples].mean(axis=1)
    ci_lo, ci_hi = boot["ci_low"], boot["ci_high"]

    # Stacked vertically (2x1) so each panel fills the single-column width
    # instead of being squeezed side-by-side.
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.8), constrained_layout=True)

    # (a) per-seed paired scatter
    ax = axes[0]
    x = np.arange(len(seeds))
    ax.scatter(x - 0.11, mappo, marker="o", s=34, color=C_MAPPO,
               label="MAPPO", zorder=3)
    ax.scatter(x + 0.11, ippo, marker="s", s=34, color=C_IPPO,
               label="IPPO ensemble", zorder=3)
    for k, (m, i) in enumerate(zip(mappo, ippo)):
        col = C_MAPPO if m > i else C_IPPO
        ax.plot([k - 0.11, k + 0.11], [m, i], color=col, alpha=0.5,
                lw=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_xlabel("training seed")
    ax.set_ylabel("test reward")
    ax.set_title("(a) per-seed test reward (8 seeds)", pad=18)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, borderaxespad=0.0, columnspacing=1.4)
    ix13 = seeds.index("13")
    ax.annotate("only IPPO win", xy=(ix13 + 0.11, ippo[ix13]),
                xytext=(ix13 - 1.9, ippo[ix13] - 165), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.5))

    # (b) bootstrap distribution
    ax = axes[1]
    ax.hist(boot_means, bins=45, color="#b8b8b8", edgecolor="white",
            linewidth=0.3)
    ax.axvspan(ci_lo, ci_hi, color=C_MAPPO, alpha=0.16,
               label=f"95% CI [{ci_lo:+.0f}, {ci_hi:+.0f}]")
    ax.axvline(boot["mean_delta"], color=C_MAPPO, lw=1.3,
               label=f"mean $\\Delta$ = {boot['mean_delta']:+.0f}")
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel(r"paired $\Delta$ = MAPPO $-$ IPPO (reward units)")
    ax.set_ylabel("bootstrap count")
    p = boot["p_one_sided_mappo_ge_ippo"]
    ax.set_title(f"(b) paired bootstrap, $p$ = {p:.3f}", pad=18)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, borderaxespad=0.0, columnspacing=1.4)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_bootstrap.pdf")
    fig.savefig(OUT / "fig_bootstrap.png", dpi=200)
    plt.close(fig)
    print("wrote", (OUT / "fig_bootstrap.pdf").relative_to(REPO_ROOT))


def fig_per_cluster(data: dict) -> None:
    summary = data["summary"]
    mappo = _per_cluster_mean(summary, "MAPPO")
    ippo = _per_cluster_mean(summary, "IPPO-ensemble")
    dpdk = np.array(summary["aggregated"]["always-DPDK"]
                    ["per_seed"]["0"]["per_cluster_total_reward"])
    K = len(mappo)

    # Per-cluster mean load (Gbps) for sorting + axis labels.
    loads = {}
    for line in HET_CSV.read_text().splitlines()[1:]:
        parts = line.split(",")
        loads[int(float(parts[0]))] = float(parts[1])
    order = sorted(range(K), key=lambda c: loads.get(c, 0.0))

    fig, ax = plt.subplots(figsize=(7.2, 3.1), constrained_layout=True)
    x = np.arange(K)
    w = 0.27
    ax.bar(x - w, dpdk[order], w, color=C_DPDK, label="always-DPDK",
           edgecolor="black", linewidth=0.3)
    ax.bar(x, ippo[order], w, color=C_IPPO, label="IPPO ensemble",
           edgecolor="black", linewidth=0.3)
    ax.bar(x + w, mappo[order], w, color=C_MAPPO, label="MAPPO",
           edgecolor="black", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"c{c}\n{loads[c]*1000:.0f}" for c in order])
    ax.set_xlabel("cluster (sorted by mean offered load, Mbps)")
    ax.set_ylabel("mean test reward")
    ax.set_title("Per-cluster mean test reward (8 seeds)", pad=18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              frameon=False, borderaxespad=0.0, columnspacing=1.4)
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    fig.savefig(OUT / "fig_per_cluster.pdf")
    fig.savefig(OUT / "fig_per_cluster.png", dpi=200)
    plt.close(fig)
    print("wrote", (OUT / "fig_per_cluster.pdf").relative_to(REPO_ROOT))


def main() -> int:
    data = _load()
    fig_bootstrap(data)
    fig_per_cluster(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
