"""Figures from existing saved records only (no policy or environment is run).

fig_mechanism          (a) USR share by offered-load band, every seed, both controllers
                       (b) per-step SEC saving of USR vs expected penalty of a jump-forced exit, by band
                       (c) per-seed decomposition of Shared-PPO minus MAPPO cost
fig_checkpoint_selection
                       (a) validation return at each evaluation (the only saved learning record), best marked
                       (b) selected step vs USR share at 50–81 Mbps, 16 runs, seed-paired
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "pdf.fonttype": 42})
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
COL = {"MAPPO": "#2a78d6", "Shared-PPO": "#eb6834"}
MRK = {"MAPPO": "o", "Shared-PPO": "s"}
SEEDS = [1, 7, 13, 23, 42, 64, 77, 99]
BANDS = ["<10", "10–25", "25–50", "50–81", "81–149", "149–300"]


def style(ax, grid="y"):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, width=0.5, length=2.5)
    if grid:
        ax.grid(True, axis=grid, color=GRID, lw=0.4, zorder=0)


def save(fig, name):
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(HERE / "figures" / f"{name}.png", bbox_inches="tight", facecolor="white", dpi=220)
    plt.close(fig)


def fig_mechanism():
    lb = pd.read_csv(HERE / "load_bin_summary.csv")
    hv = pd.read_csv(HERE / "hold_vs_onset_risk_by_band.csv")
    dec = pd.read_csv(HERE / "decomposition_by_seed.csv")

    fig = plt.figure(figsize=(7.16, 4.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1.05], hspace=0.62, wspace=0.28)
    ax, bx, cx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    # (a) USR share by load band
    style(ax)
    xs = np.arange(len(BANDS))
    ax.axvspan(xs[3] - 0.5, xs[3] + 0.5, color="#f1f0ec", zorder=0, lw=0)
    for j, (c, col) in enumerate((("MAPPO", "usr_share_mappo"), ("Shared-PPO", "usr_share_shared"))):
        off = -0.13 if j == 0 else 0.13
        for i, b in enumerate(BANDS):
            v = 100 * lb[lb.load_bin == b].set_index("seed").loc[SEEDS, col].to_numpy()
            ax.scatter(np.full(8, i + off) + np.linspace(-0.05, 0.05, 8), v, s=7, color=COL[c], alpha=0.45, lw=0, zorder=2)
            ax.scatter(i + off, v.mean(), marker=MRK[c], s=26, facecolor=COL[c], edgecolor="white", lw=0.6, zorder=3,
                       label=c if i == 0 else None)
    m50 = lb[lb.load_bin == "50–81"].set_index("seed")
    ax.annotate(f"MAPPO {100 * m50.usr_share_mappo.mean():.0f}% vs\nShared-PPO {100 * m50.usr_share_shared.mean():.0f}%\n"
                f"(Shared-PPO higher in {int((m50.usr_share_shared > m50.usr_share_mappo).sum())}/8 seeds)",
                (2.87, 100 * m50.usr_share_mappo.mean()), xytext=(0.05, 40), fontsize=6.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax.set_xticks(xs)
    ax.set_xticklabels(BANDS)
    ax.set_xlabel("offered load at the decision step (Mbps)", color=INK2)
    ax.set_ylabel("USR selected (% of cluster-steps)", color=INK2)
    ax.set_ylim(-4, 104)
    ax.set_title("(a) Where each controller uses the user-space UPF", loc="left", color=INK)
    ax.legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 1.0))

    # (b) saving vs expected exit penalty by band
    style(bx)
    rows = hv[hv.band.isin(["10–25", "25–50", "50–81", "81–149"])]
    x = np.arange(len(rows))
    bx.bar(x - 0.19, rows.mean_sec_saving_per_usr_step, width=0.36, color="#1baf7a", edgecolor="white", lw=0.8,
           label="SEC term saved per USR step", zorder=2)
    bx.bar(x + 0.19, rows.expected_exit_qos_penalty, width=0.36, color="#e34948", edgecolor="white", lw=0.8,
           label="expected jump-exit QoS penalty", zorder=2)
    for i, r in enumerate(rows.itertuples()):
        bx.text(i, max(r.mean_sec_saving_per_usr_step, r.expected_exit_qos_penalty) + 0.25,
                f"p={r.p_onset_next:.3f}\nnet {r.net_expected_per_step_hold:+.1f}", ha="center", va="bottom", fontsize=6.2, color=INK2)
    bx.set_xticks(x)
    bx.set_xticklabels(rows.band)
    bx.set_ylim(0, 12.5)
    bx.set_xlabel("offered load at t (Mbps)", color=INK2)
    bx.set_ylabel("reward units per step", color=INK2)
    bx.set_title("(b) One-step value of holding USR (test traffic)", loc="left", color=INK)
    bx.legend(loc="upper left", frameon=False, fontsize=6.3)

    # (c) per-seed decomposition of Shared-PPO - MAPPO cost
    style(cx)
    cats = [
        ("SEC saved on USR holds only Shared-PPO makes", "#1baf7a",
         lambda d: d[d.category == "Shared-PPO USR / MAPPO DPDK"].delta_energy_term.sum()),
        ("SEC saved on USR holds only MAPPO makes", "#eda100",
         lambda d: d[d.category == "MAPPO USR / Shared-PPO DPDK"].delta_energy_term.sum()),
        ("QoS penalty: jump-forced exits only Shared-PPO is exposed to", "#e34948",
         lambda d: d[d.usr_risk_class == "Shared-PPO exits USR, MAPPO already in DPDK"].delta_qos_penalty.sum()),
        ("QoS penalty: jump-forced exits only MAPPO is exposed to", "#4a3aa7",
         lambda d: d[d.usr_risk_class == "MAPPO exits USR, Shared-PPO already in DPDK"].delta_qos_penalty.sum()),
    ]
    rows = []
    for s in [*SEEDS, "mean"]:
        d = dec if s == "mean" else dec[dec.seed == s]
        div = 8 if s == "mean" else 1
        vals = [f(d) / div for _, _, f in cats]
        total = -(d.delta_reward.sum()) / div
        rest = total - sum(vals)
        rows.append((s, vals, rest, total))
    xs = np.arange(len(rows)) + np.array([0] * 8 + [0.6])
    for i, (s, vals, rest, total) in enumerate(rows):
        pos, neg = 0.0, 0.0
        for (lab, colr, _), v in zip(cats + [("switching, cooldown and other", MUTED, None)], vals + [rest]):
            bottom = pos if v >= 0 else neg
            cx.bar(xs[i], v, bottom=bottom, width=0.62, color=colr, edgecolor="white", lw=0.8, zorder=2)
            if v >= 0:
                pos += v
            else:
                neg += v
        cx.scatter(xs[i], total, marker="D", s=22, facecolor="white", edgecolor=INK, lw=0.9, zorder=4)
        cx.text(xs[i], max(pos, total) + 12, f"{total:+.0f}", ha="center", va="bottom", fontsize=6.3, color=INK)
    cx.axhline(0, color=INK2, lw=0.6)
    cx.set_ylim(-240, 470)
    cx.set_xticks(xs)
    cx.set_xticklabels([f"seed {s}" for s, *_ in rows[:-1]] + ["mean of 8"])
    cx.set_ylabel("Shared-PPO − MAPPO cost\n(reward units; > 0: Shared-PPO worse)", color=INK2)
    cx.set_title("(c) What the cost difference is made of, per seed (all 10,090 cluster-steps of each test episode)",
                 loc="left", color=INK)
    handles = [Patch(color=c, label=lab) for lab, c, _ in cats] + [Patch(color=MUTED, label="switching, cooldown and other"),
               Line2D([], [], marker="D", ls="", mfc="white", mec=INK, label="net difference")]
    cx.legend(handles=handles, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=6.4)
    save(fig, "fig_mechanism")


def fig_selection():
    sel = pd.read_csv(HERE / "checkpoint_selection.csv")
    ex = pd.read_csv(HERE / "exposure_by_run.csv")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 2.7), gridspec_kw=dict(width_ratios=[1.45, 1]))
    style(ax)
    for r in sel.itertuples():
        st = np.array(json.loads(r.val_steps)) / 1000
        v = -np.array(json.loads(r.val_returns))
        ax.plot(st, v, color=COL[r.controller], lw=0.6, alpha=0.45, zorder=2)
        ax.scatter(r.best_step / 1000, -r.best_val_return, marker=MRK[r.controller], s=18, facecolor=COL[r.controller],
                   edgecolor="white", lw=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("training step (thousands; evaluation every 4,096 steps)", color=INK2)
    ax.set_ylabel("validation cost = −return (log)", color=INK2)
    ax.set_title("(a) Validation return during training; marker = selected checkpoint", loc="left", color=INK)
    ax.text(200, 4.4e7, r"$\approx 4.9\times10^{7}$: level most runs return to later in training", ha="right", va="top", fontsize=6.2, color=INK2)
    ax.legend(handles=[Line2D([], [], color=COL[c], marker=MRK[c], lw=0.8, ms=4, label=c) for c in COL],
              frameon=False, loc="center right")

    style(bx, grid="both")
    w = ex.pivot(index="seed", columns="controller")
    for s in SEEDS:
        bx.plot([w.loc[s, ("best_step", "MAPPO")] / 1000, w.loc[s, ("best_step", "Shared-PPO")] / 1000],
                [100 * w.loc[s, ("usr_share_50_81", "MAPPO")], 100 * w.loc[s, ("usr_share_50_81", "Shared-PPO")]],
                color="#cfcdc7", lw=0.7, zorder=1)
    for c in COL:
        q = ex[ex.controller == c]
        bx.scatter(q.best_step / 1000, 100 * q.usr_share_50_81, marker=MRK[c], s=22, facecolor=COL[c], edgecolor="white",
                   lw=0.5, zorder=3, label=c)
    for s in SEEDS:
        q = ex[(ex.controller == "Shared-PPO") & (ex.seed == s)].iloc[0]
        bx.annotate(str(s), (q.best_step / 1000, 100 * q.usr_share_50_81), textcoords="offset points", xytext=(4, -2),
                    fontsize=5.8, color=INK2)
    bx.set_xlabel("selected checkpoint step (thousands)", color=INK2)
    bx.set_ylabel("USR share at 50–81 Mbps (%)", color=INK2)
    bx.set_title("(b) Selection step vs USR use; lines join seeds", loc="left", color=INK)
    bx.legend(frameon=False, loc="upper right")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_checkpoint_selection")


if __name__ == "__main__":
    fig_mechanism()
    fig_selection()
    print("ok")
