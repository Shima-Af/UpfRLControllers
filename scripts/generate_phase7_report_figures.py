"""Generate Phase 7 supervisor-facing figures.

Consumes ``reports/phase-7/multiseed_summary.json`` and produces:

  fig1_reward_with_errorbars.png   Bar chart of mean test reward
                                   with std error bars across seeds.
  fig2_per_seed_scatter.png        Per-seed scatter for the three
                                   trained controllers — shows the
                                   centralised-PPO variance directly.
  fig3_per_cluster_compare.png     Per-cluster reward across MAPPO
                                   vs ensemble vs always-DPDK
                                   (mean across seeds for the trained
                                   controllers).
  fig4_safety_vs_reward.png        Scatter: unsafe % vs reward across
                                   policies, with seed clouds.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "reports" / "phase-7" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Stable colours per controller family. Baselines that include
# tunable parameters in their label match by prefix.
COLOR_PREFIX = [
    ("MAPPO",            "#ef4444"),
    ("IPPO-ensemble",    "#22c55e"),
    ("centralised-PPO",  "#0ea5e9"),
    ("hysteresis(t_up=81,t_down=61", "#a855f7"),   # tuned-band
    ("hysteresis(t_up=81,t_down=0",  "#8b5cf6"),   # auto-band (degenerate)
    ("threshold(derived", "#ec4899"),
    ("always-DPDK",      "#10b981"),
]


def _color(name: str) -> str:
    for prefix, c in COLOR_PREFIX:
        if name.startswith(prefix):
            return c
    return "#475569"


# Display order — best-to-worst-ish, baselines grouped.
ORDER_PRIORITY = [
    "MAPPO",
    "IPPO-ensemble",
    "hysteresis(t_up=81,t_down=61",  # tuned-band
    "always-DPDK",
    "hysteresis(t_up=81,t_down=0",   # auto-band (= DPDK)
    "threshold(derived",
    "centralised-PPO",
]


def _order_names(present: list[str]) -> list[str]:
    """Sort present controller names by ORDER_PRIORITY prefixes."""
    out: list[str] = []
    for prefix in ORDER_PRIORITY:
        for n in present:
            if n.startswith(prefix) and n not in out:
                out.append(n)
    # Any controllers not matched fall to the end in original order.
    for n in present:
        if n not in out:
            out.append(n)
    return out


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


def main() -> int:
    j = json.loads(
        (REPO_ROOT / "reports" / "phase-7" / "multiseed_summary.json")
        .read_text()
    )
    agg = j["aggregated"]
    K = j["K"]
    print(f"Loaded multiseed summary, controllers: {list(agg.keys())}")

    # ------------------------------------------------------------------
    # fig1 — reward with error bars
    # ------------------------------------------------------------------
    present = [n for n in agg if agg[n]["n_seeds"] > 0]
    names = _order_names(present)
    means = [agg[n]["reward_mean"] for n in names]
    stds = [agg[n]["reward_std"] for n in names]
    colors = [_color(n) for n in names]
    fig, ax = plt.subplots(figsize=(9.0, 3.5), dpi=150)
    bars = ax.barh(
        names, means, xerr=stds,
        color=colors, edgecolor="white", linewidth=1.0,
        error_kw={"ecolor": "#1e293b", "capsize": 4, "linewidth": 1.2},
    )
    for bar, v, s, n in zip(bars, means, stds, names, strict=True):
        n_seeds = agg[n]["n_seeds"]
        annotation = (
            f"{v:+.0f}  ±{s:.0f}  (n={n_seeds})" if n_seeds > 1
            else f"{v:+.0f}"
        )
        ax.text(
            v + (max(stds) * 0.1 + 50),
            bar.get_y() + bar.get_height() / 2,
            annotation,
            color="white", fontsize=8.5, fontweight="bold",
            va="center", ha="left",
        )
    ax.invert_yaxis()
    _setup_axes(
        ax,
        "Test-set total reward per controller (mean ± std across seeds)",
        "unweighted reward (closer to 0 is better)", "",
    )
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_reward_with_errorbars.png",
                bbox_inches="tight")
    plt.close(fig)
    print("  fig1_reward_with_errorbars.png")

    # ------------------------------------------------------------------
    # fig2 — per-seed scatter for trained controllers
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 3.2), dpi=150)
    trained = ["MAPPO", "IPPO-ensemble", "centralised-PPO"]
    y_positions = np.arange(len(trained))
    for i, name in enumerate(trained):
        if name not in agg or agg[name]["n_seeds"] == 0:
            continue
        seeds = agg[name]["seeds"]
        rewards = [
            agg[name]["per_seed"][str(s)]["total_reward_unweighted"]
            for s in seeds
        ]
        y = [i] * len(rewards)
        ax.scatter(rewards, y, color=_color(name), s=80, alpha=0.75,
                   edgecolors="white", linewidths=1.0, zorder=3)
        # Mean line
        m = agg[name]["reward_mean"]
        ax.plot([m, m], [i - 0.18, i + 0.18],
                color=_color(name), linewidth=2.5, zorder=2)
        # Label each point with its seed
        for r, s in zip(rewards, seeds, strict=True):
            ax.annotate(
                f"s{s}", (r, i), fontsize=7, color="#475569",
                xytext=(0, -12), textcoords="offset points", ha="center",
            )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(trained)
    ax.invert_yaxis()
    _setup_axes(
        ax,
        "Per-seed test reward — each dot is one trained controller, "
        "vertical bar = mean",
        "unweighted reward (closer to 0 is better)", "",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_per_seed_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig2_per_seed_scatter.png")

    # ------------------------------------------------------------------
    # fig3 — per-cluster reward, mean across seeds
    # ------------------------------------------------------------------
    def per_cluster_mean(name: str) -> np.ndarray:
        if name not in agg or agg[name]["n_seeds"] == 0:
            return np.zeros(K)
        rows = []
        for s in agg[name]["seeds"]:
            rows.append(
                agg[name]["per_seed"][str(s)]["per_cluster_total_reward"]
            )
        return np.array(rows).mean(axis=0)

    def per_cluster_std(name: str) -> np.ndarray:
        if name not in agg or agg[name]["n_seeds"] <= 1:
            return np.zeros(K)
        rows = []
        for s in agg[name]["seeds"]:
            rows.append(
                agg[name]["per_seed"][str(s)]["per_cluster_total_reward"]
            )
        return np.array(rows).std(axis=0, ddof=1)

    show_prefixes = [
        "MAPPO", "IPPO-ensemble",
        "hysteresis(t_up=81,t_down=61",  # tuned hysteresis
        "always-DPDK",
    ]
    show = []
    for prefix in show_prefixes:
        for n in agg:
            if n.startswith(prefix) and agg[n]["n_seeds"] > 0:
                show.append(n)
                break
    fig, ax = plt.subplots(figsize=(9.5, 3.4), dpi=150)
    n_p = len(show)
    w = 0.8 / n_p
    x = np.arange(K)
    for i, name in enumerate(show):
        m = per_cluster_mean(name)
        s = per_cluster_std(name)
        offset = (i - (n_p - 1) / 2) * w
        # Shorten long hysteresis label for the legend.
        label = name if not name.startswith("hysteresis") else (
            f"hysteresis(band=20Mbps,cd=1)"
        )
        ax.bar(
            x + offset, m, width=w, yerr=s,
            color=_color(name), label=label,
            error_kw={"ecolor": "#1e293b", "capsize": 3, "linewidth": 0.8},
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{k}" for k in range(K)])
    _setup_axes(
        ax,
        "Per-cluster test reward (mean ± std across seeds where applicable)",
        "cluster", "summed reward over episode",
    )
    ax.legend(fontsize=8, frameon=False, ncol=n_p)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_per_cluster_compare.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig3_per_cluster_compare.png")

    # ------------------------------------------------------------------
    # fig4 — safety vs reward scatter (per-seed clouds)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    for name in _order_names([n for n in agg if agg[n]["n_seeds"] > 0]):
        for s in agg[name]["seeds"]:
            row = agg[name]["per_seed"][str(s)]
            r = row["total_reward_unweighted"]
            u = row["agg_unsafe_rate"] * 100
            short = name
            if name.startswith("hysteresis(t_up=81,t_down=61"):
                short = "hysteresis (band=20)"
            elif name.startswith("hysteresis(t_up=81,t_down=0"):
                short = "hysteresis (auto band)"
            elif name.startswith("threshold(derived"):
                short = "threshold (derived 81Mbps)"
            ax.scatter(
                u, r, color=_color(name), s=70,
                alpha=0.85, edgecolors="white", linewidths=1.0,
                label=short if s == agg[name]["seeds"][0] else None,
                zorder=3,
            )
    _setup_axes(
        ax,
        "Safety vs reward trade-off — top-left corner is the ideal",
        "QoS unsafe rate (%)", "test reward (closer to 0 is better)",
    )
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_safety_vs_reward.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig4_safety_vs_reward.png")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
