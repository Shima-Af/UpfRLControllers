"""Step 5 — publication figures (PDF + PNG) for the calibration-robustness study.

fig1_calibration_robustness   reward / energy / QoS-violation per controller, every world, seed range
fig2_energy_qos_scatter       controller positions in (Wh, QoSv) across worlds
fig3_ranking_heatmap          reward / energy / QoS ranks, controllers x worlds
fig4_difference_from_pinned   MAPPO, IPPO, hysteresis: world minus pinned, Experiment A and B
fig5_decision_trace           48 h trace chosen by the traffic-only rule of analysis_plan.md
(+ S2 variants of fig1-fig4 with suffix _S2)
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import rb_common as C  # noqa: E402
from evaluate_worlds import safe  # noqa: E402
from s04_analyze import BASE_IDS, RANKED, controller_sets  # noqa: E402

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "pdf.fonttype": 42, "savefig.dpi": 300})
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
COL = {"MAPPO": "#2a78d6", "IPPO": "#1baf7a", "Centralized PPO": "#4a3aa7", "Hysteresis (fixed)": "#eda100",
       "Always-DPDK": "#e34948"}
MRK = {"MAPPO": "o", "IPPO": "s", "Centralized PPO": "^", "Hysteresis (fixed)": "D", "Always-DPDK": "v"}
SHORT = {"MAPPO": "MAPPO", "IPPO": "IPPO", "Centralized PPO": "Centralized PPO", "Hysteresis (fixed)": "Hysteresis (fixed)",
         "Always-DPDK": "Always-DPDK"}
W = [w["world"] for w in C.worlds()]
XPOS = {}
_x = 0.0
for w in W:            # gaps between world groups
    if w in ("LOSO-dep-f0", "LOLO-dep-f0", "LOSO-nes-f0", "LOLO-nes-f0"):
        _x += 0.8
    XPOS[w] = _x
    _x += 1.0
GROUPS = [("pinned", ["pinned"]), ("LOSO deployed", [f"LOSO-dep-f{k}" for k in range(5)]),
          ("LOLO deployed", [f"LOLO-dep-f{k}" for k in range(5)]), ("LOSO nested", [f"LOSO-nes-f{k}" for k in range(5)]),
          ("LOLO nested", [f"LOLO-nes-f{k}" for k in range(5)])]
OUT = C.HERE / "figures"


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


def world_axis(ax, labels=True):
    ax.set_xlim(-0.6, max(XPOS.values()) + 0.6)
    ticks = [XPOS[w] for w in W]
    ax.set_xticks(ticks)
    if labels:
        ax.set_xticklabels(["P" if w == "pinned" else w[-1] for w in W], fontsize=6)
    else:
        ax.set_xticklabels([])
    for name, ws in GROUPS[1:]:
        x0 = XPOS[ws[0]] - 0.5
        ax.axvline(x0 - 0.4, color=GRID, lw=0.8, zorder=0)
    # shade secondary (nested) worlds
    ax.axvspan(XPOS["LOSO-nes-f0"] - 0.9, max(XPOS.values()) + 0.6, color="#f1f0ec", zorder=0, lw=0)


SHORTG = {"pinned": "P", "LOSO deployed": "LOSO-dep", "LOLO deployed": "LOLO-dep", "LOSO nested": "LOSO-nes",
          "LOLO nested": "LOLO-nes"}


def group_labels(ax, y=1.02, short=False, pinned=True):
    for name, ws in GROUPS:
        if name == "pinned" and not pinned:
            continue
        xm = np.mean([XPOS[w] for w in ws])
        ax.text(xm, y, SHORTG[name] if short else name, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=5.6 if short else 6.3, color=INK2)


PRIMARY_OR_REF = [w["world"] for w in C.worlds(("reference", "primary"))]


def clip_to(ax, values, pad=0.08, include_zero=False):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if include_zero:
        v = np.append(v, 0.0)
    lo, hi = v.min(), v.max()
    span = hi - lo if hi > lo else max(abs(hi), 1.0) * 0.1
    ax.set_ylim(lo - pad * span, hi + pad * span)


def offscale(ax, x, v, color, marker_fmt="{:,.0f}", label=True):
    """Draw an edge arrow + value for a point outside the y-limits; return True if drawn."""
    lo, hi = ax.get_ylim()
    if lo <= v <= hi:
        return False
    top = v > hi
    y = hi if top else lo
    ax.scatter(x, y, marker="^" if top else "v", s=18, facecolor="white", edgecolor=color, lw=0.8, zorder=5, clip_on=False)
    if not label:
        return True
    ax.annotate(marker_fmt.format(v), (x, y), textcoords="offset points", xytext=(0, -6 if top else 6),
                ha="center", va="top" if top else "bottom", fontsize=4.6, color=INK2, rotation=90,
                annotation_clip=False)
    return True


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white", dpi=220)
    plt.close(fig)


def fig1(summ: pd.DataFrame, s: str, suffix=""):
    mets = [("reward", "Reward (higher is better)", 1.0), ("energy_wh", "Energy (Wh)", 1.0),
            ("qos_violation_rate", "QoS-violation rate (%)", 100.0)]
    fig, axes = plt.subplots(len(RANKED), 3, figsize=(7.16, 8.2), sharex=True)
    for i, c in enumerate(RANKED):
        g = summ[(summ.set == s) & (summ.controller == c) & (summ.experiment == "A")].set_index("world")
        n = int(g.n_seeds.iloc[0])
        for j, (m, lab, sc) in enumerate(mets):
            ax = axes[i, j]
            style(ax)
            world_axis(ax, labels=(i == len(RANKED) - 1))
            pin = g.loc["pinned", f"{m}_mean"] * sc
            ax.axhline(pin, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
            ref = g.loc[PRIMARY_OR_REF]
            clip_to(ax, np.concatenate([ref[f"{m}_min"], ref[f"{m}_max"]]) * sc)
            fmt = "{:,.2f}" if m == "qos_violation_rate" else "{:,.0f}"
            for w in W:
                x = XPOS[w]
                if n > 1:
                    ax.plot([x, x], [g.loc[w, f"{m}_min"] * sc, g.loc[w, f"{m}_max"] * sc], color=COL[c], alpha=0.35,
                            lw=1.6, solid_capstyle="butt", zorder=2)
                sec = C.world_by_name(w)["tier"] == "secondary"
                v = g.loc[w, f"{m}_mean"] * sc
                if not offscale(ax, x, v, COL[c], fmt):
                    ax.scatter(x, v, marker=MRK[c], s=16 if w != "pinned" else 30,
                               facecolor="white" if sec else COL[c], edgecolor=INK if w == "pinned" else COL[c],
                               lw=0.8, zorder=3)
            if i == 0:
                ax.set_title(lab, color=INK, pad=12)
                group_labels(ax, 1.0, short=True, pinned=False)
            if j == 0:
                ax.set_ylabel(f"{SHORT[c]}\n(n = {n})" if n > 1 else SHORT[c], color=INK)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    handles = [Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=4, label="seed mean, primary world (deployed refit)"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=4, label="seed mean, secondary world (nested refit)"),
               Line2D([], [], color=INK2, alpha=0.35, lw=2, label="seed min–max within the world"),
               Line2D([], [], color=INK2, lw=0.6, ls=(0, (3, 2)), label="pinned twin value"),
               Line2D([], [], marker="v", ls="", mfc="white", mec=INK2, ms=4, label="seed mean off the axis (value printed)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.005), fontsize=6.3)
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=0.6, w_pad=0.8)
    fig.text(0.5, 0.03, "calibration world (P = pinned; digit = fold; shaded = secondary nested refits)", ha="center",
             color=INK2, fontsize=8)
    save(fig, f"fig1_calibration_robustness{suffix}")


def fig2(rs: pd.DataFrame, s: str, suffix=""):
    g = rs[(rs.set == s) & (rs.experiment == "A")]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 3.1), gridspec_kw=dict(width_ratios=[1, 1.25]))
    for a in (ax, bx):
        style(a, grid="both")
    for c in RANKED:
        for a in (ax, bx):
            pin = g[g.world == "pinned"].iloc[0]
            px, py = pin[f"mean_energy_wh__{c}"], 100 * pin[f"mean_qosv__{c}"]
            for _, r in g[g.tier != "reference"].iterrows():
                x, y = r[f"mean_energy_wh__{c}"], 100 * r[f"mean_qosv__{c}"]
                sec = r.tier == "secondary"
                if sec and a is bx:
                    continue
                if not sec:
                    a.plot([px, x], [py, y], color=COL[c], lw=0.4, alpha=0.45, zorder=1)
                a.scatter(x, y, marker=MRK[c], s=14, facecolor="white" if sec else COL[c], edgecolor=COL[c], lw=0.7, zorder=2)
            a.scatter(px, py, marker=MRK[c], s=46, facecolor=COL[c], edgecolor=INK, lw=0.9, zorder=4)
    ax.set_xlabel("Energy (Wh, lower is better)", color=INK2)
    ax.set_ylabel("QoS-violation rate (%, lower is better)", color=INK2)
    ax.set_title("(a) All controllers", loc="left", color=INK)
    # zoom: exclude centralized PPO
    zc = [c for c in RANKED if c != "Centralized PPO"]
    gp = g[g.tier != "secondary"]
    xs = np.concatenate([gp[f"mean_energy_wh__{c}"].to_numpy() for c in zc])
    ys = np.concatenate([100 * gp[f"mean_qosv__{c}"].to_numpy() for c in zc])
    bx.set_xlim(xs.min() - 5, xs.max() + 5)
    bx.set_ylim(0.2, ys.max() + 0.12)
    bx.set_title("(b) Zoom: pinned and primary worlds, without centralized PPO", loc="left", color=INK)
    bx.set_xlabel("Energy (Wh, lower is better)", color=INK2)
    pin = g[g.world == "pinned"].iloc[0]
    for c in RANKED:
        px, py = pin[f"mean_energy_wh__{c}"], 100 * pin[f"mean_qosv__{c}"]
        if c == "Centralized PPO":
            ax.annotate(SHORT[c], (px, py), textcoords="offset points", xytext=(-10, 10), fontsize=7, color=INK, ha="right")
            continue
        txt = {"MAPPO": (px - 2, 0.30), "IPPO": (px + 10, 0.36), "Hysteresis (fixed)": (px + 14, 0.47),
               "Always-DPDK": (px - 16, 0.60)}[c]
        bx.annotate(SHORT[c], (px, py), xytext=txt, textcoords="data", fontsize=7, color=INK, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=1, shrinkB=4))
    handles = [Line2D([], [], marker=MRK[c], ls="", mfc=COL[c], mec=COL[c], ms=5, label=SHORT[c]) for c in RANKED]
    handles += [Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK, ms=6, label="pinned (black edge)"),
                Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=4, label="primary fold (line to pinned)"),
                Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=4, label="secondary fold (panel a only)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, f"fig2_energy_qos_scatter{suffix}")


def fig3(rs: pd.DataFrame, s: str, suffix=""):
    g = rs[(rs.set == s) & (rs.experiment == "A")].set_index("world")
    fig, axes = plt.subplots(3, 1, figsize=(7.16, 4.6), sharex=True)
    cmap = ListedColormap(["#0d366b", "#2a78d6", "#7fb0ea", "#c5dcf6", "#eef4fb"])
    for ax, (lab, title) in zip(axes, (("reward", "Rank by scalarized reward (1 = best)"),
                                       ("energy", "Rank by energy (1 = lowest Wh)"),
                                       ("qos", "Rank by QoS-violation rate (1 = lowest)"))):
        M = np.array([[g.loc[w, f"rank_{lab}__{c}"] for w in W] for c in RANKED], float)
        xs = [XPOS[w] for w in W]
        for i in range(len(RANKED)):
            for j, w in enumerate(W):
                v = int(M[i, j])
                ax.add_patch(plt.Rectangle((xs[j] - 0.46, i - 0.46), 0.92, 0.92, color=cmap(v - 1), lw=0))
                ax.text(xs[j], i, str(v), ha="center", va="center", fontsize=6.5, color="white" if v <= 2 else INK)
        ax.set_ylim(len(RANKED) - 0.5, -0.5)
        ax.set_xlim(-0.6, max(xs) + 0.6)
        ax.set_yticks(range(len(RANKED)))
        ax.set_yticklabels([SHORT[c] for c in RANKED], color=INK)
        ax.set_title(title, loc="left", color=INK, pad=11 if lab == "reward" else 3)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0, colors=INK2)
        if lab == "reward":
            group_labels(ax, 1.0)
    axes[-1].set_xticks([XPOS[w] for w in W])
    axes[-1].set_xticklabels(["P" if w == "pinned" else w[-1] for w in W], fontsize=6.5)
    axes[-1].set_xlabel("calibration world (P = pinned; digit = fold; columns 12–21 = secondary nested refits)", color=INK2)
    fig.tight_layout(h_pad=0.5)
    save(fig, f"fig3_ranking_heatmap{suffix}")


def fig4(ev: pd.DataFrame, rp: pd.DataFrame, sets, s: str = "S1", suffix: str = ""):
    ids = {"MAPPO": sets[s]["MAPPO"], "IPPO": sets[s]["IPPO"], "Hysteresis (fixed)": [BASE_IDS["Hysteresis (fixed)"]]}
    mets = [("reward", "Δ reward vs pinned", 1.0), ("energy_wh", "Δ energy vs pinned (Wh)", 1.0),
            ("qos_violation_rate", "Δ QoS-violation rate vs pinned (pp)", 100.0)]
    WW = [w for w in W if w != "pinned"]
    fig, axes = plt.subplots(3, 2, figsize=(7.16, 6.0), sharex=True, sharey="row")
    offs = {"MAPPO": -0.25, "IPPO": 0.0, "Hysteresis (fixed)": 0.25}
    for j, (df, title) in enumerate(((ev, "Experiment A — closed-policy zero-shot"), (rp, "Experiment B — fixed-action replay"))):
        for i, (m, lab, sc) in enumerate(mets):
            ax = axes[i, j]
            style(ax)
            world_axis(ax, labels=(i == 2))
            ax.axhline(0, color=INK2, lw=0.6)
            allid = [x for v_ in ids.values() for x in v_]
            prim = pd.concat([ev, rp])
            prim = prim[prim.ckpt_id.isin(allid) & prim.world.isin([w for w in PRIMARY_OR_REF if w != "pinned"])]
            clip_to(ax, prim[f"diff_from_pinned_{m}"].to_numpy() * sc, include_zero=True)
            fmt = "{:+,.2f}" if m == "qos_violation_rate" else "{:+,.0f}"
            for c, cid in ids.items():
                h = df[df.ckpt_id.isin(cid) & df.world.isin(WW)]
                for w, hw in h.groupby("world", observed=True):
                    x = XPOS[w] + offs[c]
                    v = hw[f"diff_from_pinned_{m}"].to_numpy() * sc
                    if len(v) > 1:
                        ax.scatter(np.full(len(v), x), v, s=4, color=COL[c], alpha=0.45, lw=0, zorder=2)
                    if not offscale(ax, x, v.mean(), COL[c], fmt, label=False):
                        ax.scatter(x, v.mean(), marker=MRK[c], s=16, facecolor=COL[c], edgecolor="white", lw=0.4, zorder=3)
            if i == 0:
                ax.set_title(title, color=INK, pad=12)
                group_labels(ax, 1.0, pinned=False)
            if j == 0:
                ax.set_ylabel(lab, color=INK2)
    for ax in axes[-1]:
        ax.set_xlabel("calibration world (digit = fold)", color=INK2)
    handles = [Line2D([], [], marker=MRK[c], ls="", mfc=COL[c], mec="white", ms=5, label=f"{SHORT[c]} (seed mean)") for c in ids]
    handles += [Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=2.5, alpha=0.5, label="individual seeds (" + {"S1": "manuscript checkpoints", "S2": "same-code retrained set",
                                                             "S3": "v0.4 re-scored set"}[s] + ")"),
                Line2D([], [], marker="v", ls="", mfc="white", mec=INK2, ms=4,
                       label="seed mean off the axis (secondary nested worlds only; values in the CSVs)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03), fontsize=6.3)
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=0.6)
    save(fig, f"fig4_difference_from_pinned{suffix}")


def fig5(sets, thr: pd.DataFrame):
    tr = np.load(C.ROLLOUTS / "main" / "traffic_test.npz")
    L, F = tr["load_gbps"] * 1000, tr["forecast_gbps"] * 1000
    T = L.shape[0]
    dec = float(thr.loc["pinned", "decision_mbps"])
    cl = int(np.argmin(np.abs(np.median(L, axis=0) - dec)))
    inreg = ((L[:, cl] >= 50) & (L[:, cl] <= 600)).astype(int)
    csum = np.convolve(inreg, np.ones(192, int), "valid")
    start = int(np.argmax(csum))                       # first maximum = earliest start on ties
    sl = slice(start, start + 192)
    prim = thr[thr.tier == "primary"].copy()
    prim["order"] = [W.index(w) for w in prim.index]
    lo = prim.sort_values(["qos_limit_mbps", "decision_mbps", "order"]).index[0]
    hi = prim.sort_values(["qos_limit_mbps", "decision_mbps", "order"], ascending=[False, False, True]).index[0]
    worlds = ["pinned", lo, hi]
    sel = dict(cluster=cl, median_load_mbps=float(np.median(L[:, cl])), window_start_step=start, window_steps=192,
               steps_in_region=int(csum[start]), worlds=worlds,
               world_qos_limits_mbps={w: float(thr.loc[w, "qos_limit_mbps"]) for w in worlds},
               controllers=["MAPPO seed 1 (S1)", "IPPO seed 1 (S1)", "Hysteresis (fixed)"])
    (OUT / "fig5_selection.json").write_text(json.dumps(sel, indent=1))
    ctrl = [("MAPPO", [i for i in sets["S1"]["MAPPO"] if "seed1_" in i or i.endswith("seed1")]),
            ("IPPO", [i for i in sets["S1"]["IPPO"] if i.endswith("seed1")]),
            ("Hysteresis (fixed)", [BASE_IDS["Hysteresis (fixed)"]])]
    for c, v in ctrl:
        assert len(v) == 1, (c, v)
    hours = (np.arange(T)[sl] - start) * C.STEP_MIN / 60
    from s04_analyze import world_usr_safe
    fig = plt.figure(figsize=(7.16, 6.0))
    gs = fig.add_gridspec(1 + len(worlds) + len(worlds) * len(ctrl), 1,
                          height_ratios=[3.2] + [0.35] * len(worlds) + [0.42] * (len(worlds) * len(ctrl)), hspace=0.25)
    ax = fig.add_subplot(gs[0])
    style(ax)
    ax.plot(hours, L[sl, cl], color=INK, lw=0.9, label="offered load (actual)")
    ax.plot(hours, F[sl, cl], color=MUTED, lw=0.8, ls=(0, (3, 1.5)), label="one-step forecast (policy input)")
    ax.axhspan(50, 600, color="#f1f0ec", zorder=0, lw=0)
    wstyle = {worlds[0]: "-", worlds[1]: (0, (4, 2)), worlds[2]: (0, (1, 1.5))}
    for w in worlds:
        ax.axhline(thr.loc[w, "qos_limit_mbps"], color="#e34948", lw=0.8, ls=wstyle[w])
        ax.axhline(thr.loc[w, "decision_mbps"], color="#2a78d6", lw=0.8, ls=wstyle[w])
    ax.set_yscale("log")
    ax.set_ylabel("Mbps", color=INK2)
    ax.set_xlim(hours[0], hours[-1] + 0.25)
    fig.suptitle(f"Cluster {cl} (median test load {np.median(L[:, cl]):.0f} Mbps, closest to the pinned 81 Mbps decision threshold); "
                 f"test steps {start}–{start + 191}; shaded band = 50–600 Mbps", x=0.12, ha="left", y=0.965, fontsize=8.5, color=INK)
    hand = [Line2D([], [], color=INK, lw=0.9, label="offered load"), Line2D([], [], color=MUTED, lw=0.8, ls=(0, (3, 1.5)), label="forecast"),
            Line2D([], [], color="#e34948", lw=0.8, label="world USR QoS limit"), Line2D([], [], color="#2a78d6", lw=0.8, label="world decision threshold")]
    hand += [Line2D([], [], color=INK2, lw=0.8, ls=wstyle[w],
                    label=f"{w} ({thr.loc[w, 'qos_limit_mbps']:.0f} / {thr.loc[w, 'decision_mbps']:.0f} Mbps)") for w in worlds]
    ax.legend(handles=hand, ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=6.3)
    from matplotlib.ticker import FixedLocator, FuncFormatter
    ax.yaxis.set_major_locator(FixedLocator([1, 10, 50, 100, 600, 1000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(labelbottom=False)
    row = 1
    for w in worlds:
        us = world_usr_safe(C.world_by_name(w), tr["load_gbps"])[sl, cl]
        a = fig.add_subplot(gs[row], sharex=ax)
        a.imshow((~us)[None, :].astype(float), aspect="auto", cmap=ListedColormap(["#eef4fb", "#e34948"]), vmin=0, vmax=1,
                 extent=(hours[0], hours[-1] + 0.25, 0, 1))
        a.set_yticks([0.5])
        a.set_yticklabels([f"USR unsafe? {w}"], fontsize=6.3, color=INK2)
        a.tick_params(labelbottom=False, length=0)
        for sp in a.spines.values():
            sp.set_visible(False)
        row += 1
    for c, v in ctrl:
        for w in worlds:
            d = np.load(C.ROLLOUTS / "main" / w / "A" / f"{safe(v[0])}.npz")
            usr = d["usr"][sl, cl]
            viol = ~d["is_safe"][sl, cl]
            a = fig.add_subplot(gs[row], sharex=ax)
            a.imshow(usr[None, :].astype(float), aspect="auto", cmap=ListedColormap(["#e4e3df", COL[c]]), vmin=0, vmax=1,
                     extent=(hours[0], hours[-1] + 0.25, 0, 1))
            if viol.any():
                a.scatter(hours[viol] + 0.125, np.full(viol.sum(), 0.5), marker="x", s=10, color=INK, lw=0.8, zorder=3)
            a.set_ylim(0, 1)
            a.set_yticks([0.5])
            lab = {"Hysteresis (fixed)": "Hyst."}.get(c, c)
            a.set_yticklabels([f"{lab} · {w}"], fontsize=6.3, color=INK)
            a.tick_params(length=0, labelbottom=(row == gs.nrows - 1))
            for sp in a.spines.values():
                sp.set_visible(False)
            row += 1
    a.set_xlabel("hours into window  (coloured = USR selected, grey = DPDK, × = step violating the world's QoS budget; Experiment A)",
                 color=INK2)
    save(fig, "fig5_decision_trace")
    return sel


def main():
    reg = C.checkpoint_registry()
    sets = controller_sets(reg)
    summ = pd.read_csv(C.HERE / "controller_world_summary.csv")
    rs = pd.read_csv(C.HERE / "rank_stability.csv")
    ev = pd.read_csv(C.HERE / "evaluation_results.csv")
    rp = pd.read_csv(C.HERE / "fixed_action_replay.csv")
    thr = pd.read_csv(C.HERE / "bundle_thresholds.csv").set_index("world")
    fig1(summ, "S1")
    fig1(summ, "S2", "_S2")
    fig2(rs, "S1")
    fig2(rs, "S2", "_S2")
    fig3(rs, "S1")
    fig3(rs, "S2", "_S2")
    fig4(ev, rp, sets)
    fig4(ev, rp, sets, "S2", "_S2")
    print(fig5(sets, thr))


if __name__ == "__main__":
    main()
