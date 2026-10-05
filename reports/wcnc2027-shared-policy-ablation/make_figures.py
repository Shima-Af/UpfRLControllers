"""Publication figures (IEEE, STIX serif to match the paper). PDF + PNG.

fig_seed_rewards        per-seed test reward; seed-matched lines; centralized PPO on its own axis
fig_paired_differences  per-seed differences + mean and paired-bootstrap 95 % CI for the ablation contrasts
fig_energy_qos          per-seed energy vs QoS-violation rate
fig_per_cluster         per-cluster reward difference vs IPPO (MAPPO, Shared-PPO), clusters by mean load
fig_reward_components   mean cost components per controller
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "savefig.dpi": 600, "pdf.fonttype": 42,
})
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
COL = {"MAPPO": "#2a78d6", "Shared-PPO": "#eb6834", "IPPO": "#1baf7a", "Centralized PPO": "#52514e"}
MRK = {"MAPPO": "o", "Shared-PPO": "s", "IPPO": "^", "Centralized PPO": "D"}
PRIMARY = ["MAPPO", "Shared-PPO", "IPPO", "Centralized PPO"]
SEEDS = [1, 7, 13, 23, 42, 64, 77, 99]


def style(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.4, zorder=0)
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(HERE / "all_seed_metrics.csv")
    fleet = df[df.cluster.astype(str) == "all"].copy()
    pc = pd.read_csv(HERE / "paired_comparisons.csv")
    agg = pd.read_csv(HERE / "aggregate_metrics.csv")
    base = {c: fleet[fleet.controller == c].reward.mean() for c in ("Hysteresis (b=50 Mbps)", "always-DPDK")
            if (fleet.controller == c).any()}
    present = [c for c in PRIMARY if (fleet.controller == c).any()]

    # 1 ── per-seed rewards ────────────────────────────────────────────────
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 2.6), gridspec_kw=dict(width_ratios=[1.9, 1.25], wspace=0.3))
    trio = [c for c in ("MAPPO", "Shared-PPO", "IPPO") if c in present]
    xs = {c: i for i, c in enumerate(trio)}
    wide = fleet[fleet.controller.isin(trio)].pivot(index="seed", columns="controller", values="reward").sort_index()
    wide = wide.dropna()
    jit = dict(zip(wide.index, np.linspace(-0.09, 0.09, len(wide.index))))    # one offset per seed, shared by points and lines
    floor = base.get("always-DPDK", -np.inf)
    normal = wide[wide >= floor]
    lo = np.nanmin(normal.to_numpy()) - 70
    hi = np.nanmax(normal.to_numpy()) + 40
    clip = lambda v: max(v, lo + 12)
    for s_ in wide.index:
        ax.plot([xs[c] + jit[s_] for c in trio], [clip(wide.loc[s_, c]) for c in trio], color="#cfcdc7", lw=0.7, zorder=1)
    for c in trio:
        for s_ in wide.index:
            v = wide.loc[s_, c]
            if v >= lo:
                ax.scatter(xs[c] + jit[s_], v, s=14, marker=MRK[c], color=COL[c], edgecolor="white", lw=0.4, zorder=3)
            else:   # off-scale run: arrow at the lower edge, value printed
                ax.annotate("", xy=(xs[c] + jit[s_], lo), xytext=(xs[c] + jit[s_], lo + 45),
                            arrowprops=dict(arrowstyle="-|>", color=COL[c], lw=1.0), zorder=3)
                ax.text(xs[c] + jit[s_] + 0.05, lo + 20, f"seed {s_}: {v:,.0f}", fontsize=6.3, color=INK2, va="center")
        a = agg[(agg.controller == c) & (agg.metric == "reward")].iloc[0]
        med = a["median"]
        ax.scatter(xs[c] + 0.28, med, marker=MRK[c], s=30, facecolor="white", edgecolor=COL[c], lw=1.2, zorder=4)
    ax.set_ylim(lo, hi)
    ax.set_xticks(range(len(trio)), trio)
    ax.set_xlim(-0.4, len(trio) - 0.3)
    ax.set_ylabel("Test reward (higher is better)", color=INK2)
    ax.set_title("(a) Seed-matched runs (lines join equal seeds); open marker = median", color=INK, loc="left")
    style(ax)
    order4 = [c for c in PRIMARY if c in present]
    for i, c in enumerate(order4):
        g = fleet[fleet.controller == c].sort_values("seed")
        bx.scatter(i + np.linspace(-0.15, 0.15, len(g)), g.reward, s=12, marker=MRK[c], color=COL[c],
                   edgecolor="white", lw=0.4, zorder=3)
        a = agg[(agg.controller == c) & (agg.metric == "reward")].iloc[0]
        bx.errorbar(i + 0.32, a["mean"], yerr=[[a["mean"] - a["t_ci_lo"]], [a["t_ci_hi"] - a["mean"]]], fmt=MRK[c],
                    ms=4, color=COL[c], mfc="white", mew=1.0, capsize=1.8, lw=0.9, zorder=4)
    for (name, v), ls in zip(base.items(), [(0, (4, 2)), (0, (1, 1.5))]):
        bx.axhline(v, color=INK2, lw=0.7, ls=ls, zorder=0, label=name.replace(" Mbps", ""))
    bx.set_xticks(range(len(order4)), [c.replace("Centralized PPO", "Central.\nPPO").replace("Shared-PPO", "Shared-\nPPO") for c in order4])
    bx.set_xlim(-0.5, len(order4) - 0.3)
    bx.legend(frameon=False, loc="lower left", fontsize=6.3, handlelength=1.8, borderaxespad=0.1)
    bx.set_title("(b) All controllers, full scale; mean and 95% t-CI", color=INK, loc="left")
    style(bx)
    save(fig, "fig_seed_rewards")

    # 2 ── paired differences ──────────────────────────────────────────────
    rows = [("Delta_crit", "MAPPO $-$ Shared-PPO\n(centralized critic)"),
            ("Delta_share", "Shared-PPO $-$ IPPO\n(sharing + implementation)"),
            ("G ", "MAPPO $-$ IPPO\n(total)")]
    sub = [(pc[(pc.metric == "reward") & pc.contrast.str.startswith(k)], lab) for k, lab in rows]
    sub = [(q.iloc[0], lab) for q, lab in sub if not q.empty]
    if sub:
        XL, XR = -200.0, 400.0
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        for i, (r, lab) in enumerate(sub):
            y = len(sub) - 1 - i
            d = np.array(json.loads(r.per_seed_diff_A_minus_B))
            yy = np.full(len(d), y) + np.linspace(-0.12, 0.12, len(d))
            inside = (d >= XL) & (d <= XR)
            ax.scatter(d[inside], yy[inside], s=10, color=MUTED, zorder=2, lw=0)
            for v, yv in zip(d[~inside], yy[~inside]):
                ax.annotate("", xy=(XR if v > XR else XL, yv), xytext=((XR - 28) if v > XR else (XL + 28), yv),
                            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8))
                ax.text(XR - 32, yv, f"{v:+,.0f}", fontsize=6.0, color=INK2, ha="right", va="center")
            lo_, hi_ = max(r.boot_ci_lo, XL), min(r.boot_ci_hi, XR)
            ax.plot([lo_, hi_], [y + 0.3] * 2, color=INK, lw=1.4, solid_capstyle="butt", zorder=3)
            mean_in = XL <= r.mean_diff <= XR
            if r.boot_ci_hi > XR:
                ax.annotate("", xy=(XR, y + 0.3), xytext=(XR - 25, y + 0.3), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
                txt = f"CI to {r.boot_ci_hi:+,.0f}" + ("" if mean_in else f"; mean {r.mean_diff:+,.0f}")
                ax.text(XR - 5, y + 0.4, txt, fontsize=6.0, color=INK2, ha="right", va="bottom")
            if mean_in:
                ax.scatter([r.mean_diff], [y + 0.3], s=22, color=INK, zorder=4, label="mean, 95% bootstrap CI" if i == 0 else None)
            ax.scatter([r.median_diff], [y + 0.3], s=26, marker="D", facecolor="white", edgecolor=INK, lw=0.9, zorder=5,
                       label="median" if i == 0 else None)
            ax.text(XL + 8, y + 0.3, f"p={r.perm_p_two_sided_exact:.3f}", fontsize=6.3, color=INK2, va="center", ha="left")
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_xlim(XL, XR)
        ax.set_yticks(range(len(sub)), [lab for _, lab in sub][::-1])
        ax.set_xlabel("Per-seed test-reward difference\n(positive = first controller better; gray = seeds)", color=INK2)
        ax.set_ylim(-0.5, len(sub) - 0.1)
        ax.legend(frameon=False, loc="lower right", fontsize=6.3, bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.1)
        style(ax, grid_axis="x")
        save(fig, "fig_paired_differences")

    # 3 ── energy vs QoS violation (full range + zoom) ────────────────────
    fig, (ax, zx) = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw=dict(wspace=0.25))
    for a_, zoom in ((ax, False), (zx, True)):
        for c in present:
            g = fleet[fleet.controller == c]
            a_.scatter(g.energy_wh, 100 * g.qos_violation_rate, s=16 if not zoom else 22, marker=MRK[c], color=COL[c],
                       edgecolor="white", lw=0.4, label=c, zorder=3, alpha=0.95)
        for name, mk in (("Hysteresis (b=50 Mbps)", "P"), ("always-DPDK", "X")):
            g = fleet[fleet.controller == name]
            if not g.empty:
                a_.scatter(g.energy_wh, 100 * g.qos_violation_rate, s=30, marker=mk, color=INK, zorder=3,
                           label=name.replace(" Mbps", ""))
        a_.set_xlabel("Energy on test slice [Wh] (lower is better)", color=INK2)
        style(a_, grid_axis="both")
    ax.set_ylabel("QoS-violation rate [%]", color=INK2)
    ax.set_title("(a) All runs", color=INK, loc="left")
    trio_runs = fleet[fleet.controller.isin(["MAPPO", "Shared-PPO", "IPPO"]) & (fleet.reward >= base.get("always-DPDK", -np.inf))]
    x0, x1 = trio_runs.energy_wh.min() - 6, max(trio_runs.energy_wh.max(), base_e := fleet[fleet.controller == "always-DPDK"].energy_wh.max()) + 6
    zx.set_xlim(x0, x1)
    y1 = max(100 * trio_runs.qos_violation_rate.max(), 100 * fleet[fleet.controller.str.startswith("Hysteresis")].qos_violation_rate.max()) + 0.08
    zx.set_ylim(0.25, y1)
    zx.set_title("(b) Zoom: runs not worse than always-DPDK", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.3, borderaxespad=0.2, fontsize=6.5)
    save(fig, "fig_energy_qos")

    # 4 ── per-cluster ablation contrasts (median, IQR over seeds) ──────────
    cl = df[df.cluster.astype(str) != "all"].copy()
    cl["cluster"] = cl.cluster.astype(int)
    if {"MAPPO", "Shared-PPO", "IPPO"} <= set(cl.controller):
        loads = cl[cl.controller == "IPPO"].groupby("cluster").mean_load_gbps.mean()
        order = loads.sort_values().index.tolist()
        w = cl[cl.controller.isin(["MAPPO", "Shared-PPO", "IPPO"])].pivot_table(index=["seed", "cluster"], columns="controller", values="reward")
        contrasts_c = [("MAPPO", "Shared-PPO", "MAPPO $-$ Shared-PPO (critic)", "#0b0b0b"),
                       ("Shared-PPO", "IPPO", "Shared-PPO $-$ IPPO (sharing + impl.)", "#8a8984"),
                       ("MAPPO", "IPPO", "MAPPO $-$ IPPO (total)", "#cfcdc7")]
        fig, ax = plt.subplots(figsize=(7.16, 2.3))
        x = np.arange(len(order))
        width = 0.26
        rows_out = []
        for j, (a, b, lab, colr) in enumerate(contrasts_c):
            diff = (w[a] - w[b]).unstack("cluster")[order]
            med = diff.median()
            q1, q3 = diff.quantile(0.25), diff.quantile(0.75)
            xx = x + (j - 1) * width
            ax.bar(xx, med, width=width * 0.92, color=colr, edgecolor=INK if colr == "#cfcdc7" else colr, lw=0.4, label=lab, zorder=2)
            ax.errorbar(xx, med, yerr=[med - q1, q3 - med], fmt="none", ecolor=INK2, lw=0.7, capsize=1.5, zorder=3)
            for k in order:
                rows_out.append(dict(contrast=lab, cluster=k, median=float(med[k]), q1=float(q1[k]), q3=float(q3[k]),
                                     mean=float(diff[k].mean()), min=float(diff[k].min()), max=float(diff[k].max())))
        pd.DataFrame(rows_out).to_csv(HERE / "per_cluster_contrasts.csv", index=False, float_format="%.6g")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xticks(x, [f"c{k}\n{loads[k]:.2f}" for k in order])
        ax.set_xlabel("Cluster (mean test load, Gbps)", color=INK2)
        ax.set_ylabel("Per-cluster reward difference\n(median, IQR over 8 seeds)", color=INK2)
        ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=6.5, borderaxespad=0.2)
        style(ax)
        save(fig, "fig_per_cluster")

    # 5 ── reward components ───────────────────────────────────────────────
    comps = [("cost_energy_term", r"$\alpha\,$SEC"), ("cost_qos_penalty", "QoS penalty"),
             ("cost_switch_penalty", "switching"), ("cost_cooldown_penalty", "cooldown")]
    shades = ["#3987e5", "#86b6ef", "#1c5cab", "#0d366b"]
    names = present + [c for c in ("Hysteresis (b=50 Mbps)", "always-DPDK") if (fleet.controller == c).any()]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    left = np.zeros(len(names))
    for (m, lab), colr in zip(comps, shades):
        v = np.array([fleet[fleet.controller == n][m].mean() for n in names])
        ax.barh(range(len(names)), v, left=left, color=colr, edgecolor="white", lw=0.6, label=lab, zorder=2)
        left += v
    ax.set_yticks(range(len(names)), [n.replace(" Mbps", "") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Mean total cost over test slice (= $-$reward)", color=INK2)
    ax.legend(frameon=False, fontsize=6.5, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2, handlelength=1.2, columnspacing=0.9)
    style(ax, grid_axis="x")
    save(fig, "fig_reward_components")


if __name__ == "__main__":
    main()
