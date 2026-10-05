"""Step 4 — publication figures (IEEE two-column, full text width).

measured_vs_predicted.{pdf,png}  2x3: rows DPDK / OAI-USR, columns power,
    delay, loss. Every point is a held-out sample (LOSO nested CV, twin-style
    inputs). Identity line in every panel; QoS budget lines and the four
    budget-quadrant counts on delay and loss (zero-inflated loss on a symlog
    axis so the zeros stay visible). Colour = offered load (one-hue ramp).
residuals_vs_load.{pdf,png}  2x3: residual (predicted - measured) against
    offered load, per-level median and 5-95th percentile band.
Scatter layers are rasterised (600 dpi); axes and text stay vector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FixedLocator, NullFormatter

import common as C

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "savefig.dpi": 600, "pdf.fonttype": 42,
})
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
RAMP = LinearSegmentedColormap.from_list("load", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])   # validated ordinal ramp
NORM = LogNorm(vmin=1e-5, vmax=5.0)
BAND_FILL, MED, SPREAD = "#f0efec", "#1c5cab", "#cde2fb"
DELAY_BUDGET, LOSS_BUDGET = C.qos_budget()
COLS = {"power": ("power_watts", "Power (W)"),
        "delay": ("downlink_one_way_delay_distribution__weighted_mean_delay_us", r"Mean DL delay ($\mu$s)"),
        "loss": ("gtpu_packets_dn__packets_lost_delta", "Packet loss (pkts / 3 s)")}
SUBTITLE = {"power": "", "delay": r"dotted: 200-$\mu$s QoS budget", "loss": "dotted: 5-pkt QoS budget"}
ROWS = {"DPDK": "SD-Core DPDK", "USR": "OAI user-space (USR)"}


def style(ax):
    ax.grid(True, color=GRID, lw=0.4, zorder=0)
    ax.tick_params(colors=INK2, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)


def quadrant_counts(ax, x, y, thr):
    tn = int(np.sum((x <= thr) & (y <= thr)))
    fp = int(np.sum((x <= thr) & (y > thr)))
    fn = int(np.sum((x > thr) & (y <= thr)))
    tp = int(np.sum((x > thr) & (y > thr)))
    kw = dict(transform=ax.transAxes, fontsize=6.2, color=INK2, zorder=6,
              bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))
    # only the two error quadrants are labelled; the full matrix is in Table B
    ax.text(0.03, 0.97, f"false-unsafe {fp:,}", ha="left", va="top", **kw)
    ax.text(0.97, 0.03, f"false-safe {fn:,}", ha="right", va="bottom", **kw)


def budget_lines(ax, thr):
    ax.axvline(thr, color=INK2, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    ax.axhline(thr, color=INK2, lw=0.6, ls=(0, (1, 1.5)), zorder=1)


def measured_vs_predicted(d: pd.DataFrame):
    fig = plt.figure(figsize=(7.16, 4.35))
    gs = fig.add_gridspec(2, 3, wspace=0.36, hspace=0.40, left=0.085, right=0.905, top=0.905, bottom=0.095)
    for i, upf in enumerate(("DPDK", "USR")):
        du = d[d["upf"] == upf].sort_values("load_gbps")
        for j, out in enumerate(("power", "delay", "loss")):
            ax = fig.add_subplot(gs[i, j])
            style(ax)
            col, title = COLS[out]
            ok = du[col].notna().to_numpy()
            x, y = du[col].to_numpy()[ok], du[f"pred_{out}"].to_numpy()[ok]
            load = du["load_gbps"].to_numpy()[ok].clip(1e-5)
            sc = ax.scatter(x, y, c=load, cmap=RAMP, norm=NORM, s=3.5, lw=0, alpha=0.75, rasterized=True, zorder=3)
            if out == "power":
                lo, hi = (0.805, 0.862) if upf == "DPDK" else (-0.15, 4.8)
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
                mae, rmse = np.mean(np.abs(y - x)), np.sqrt(np.mean((y - x) ** 2))
                unit = "mW" if upf == "DPDK" else "W"
                f = 1000 if upf == "DPDK" else 1
                ax.text(0.03, 0.97, f"MAE {f * mae:.{1 if f == 1000 else 3}f} {unit}\nRMSE {f * rmse:.{1 if f == 1000 else 3}f} {unit}",
                        transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=INK2, linespacing=1.2)
                if upf == "DPDK":
                    ax.xaxis.set_major_locator(FixedLocator([0.81, 0.83, 0.85]))
                    ax.yaxis.set_major_locator(FixedLocator([0.81, 0.83, 0.85]))
            elif out == "delay":
                ax.set_xscale("log"); ax.set_yscale("log")
                lo, hi = 40, 4e4
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
                budget_lines(ax, DELAY_BUDGET)
                quadrant_counts(ax, x, y, DELAY_BUDGET)
            else:
                ax.set_xscale("symlog", linthresh=1, linscale=0.6)
                ax.set_yscale("symlog", linthresh=1, linscale=0.6)
                lo, hi = -0.4, 2e6
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
                ticks = [0, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6]
                for axis in (ax.xaxis, ax.yaxis):
                    axis.set_major_locator(FixedLocator(ticks))
                    axis.set_minor_formatter(NullFormatter())
                ax.set_xticklabels(["0", "1", "", "$10^2$", "", "$10^4$", "", "$10^6$"])
                ax.set_yticklabels(["0", "1", "", "$10^2$", "", "$10^4$", "", "$10^6$"])
                budget_lines(ax, LOSS_BUDGET)
                if upf == "DPDK" and np.all(x == 0) and np.all(y == 0):
                    ax.text(0.62, 0.30, f"all {len(x):,} held-out samples:\nmeasured = predicted = 0",
                            transform=ax.transAxes, ha="center", va="center", fontsize=6.8, color=INK2,
                            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.9))
                else:
                    quadrant_counts(ax, x, y, LOSS_BUDGET)
            ident = np.array([lo, hi])
            ax.plot(ident, ident, color=MUTED, lw=0.7, ls=(0, (4, 2)), zorder=2)
            if i == 0:
                ax.set_title(title, color=INK, pad=12)
                if SUBTITLE[out]:
                    ax.text(0.5, 1.02, SUBTITLE[out], transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5, color=INK2)
            if i == 1:
                ax.set_xlabel("Measured", color=INK2, labelpad=1)
            if j == 0:
                ax.set_ylabel(f"{ROWS[upf]}\nPredicted (held-out)", color=INK2, labelpad=2)
    cax = fig.add_axes([0.925, 0.095, 0.012, 0.905 - 0.095])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Offered DL load (Gbps)", color=INK2)
    cb.set_ticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5])
    cb.set_ticklabels([r"$\approx$0", "$10^{-4}$", "$10^{-3}$", "0.01", "0.1", "1", "5"])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6.5, colors=INK2)
    for ext in ("pdf", "png"):
        fig.savefig(C.OUT / f"measured_vs_predicted.{ext}", dpi=600)
    plt.close(fig)


def residuals_vs_load(d: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.9))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.92, bottom=0.12, wspace=0.36, hspace=0.38)
    band = C.near_boundary_band()
    for i, upf in enumerate(("DPDK", "USR")):
        du = d[d["upf"] == upf]
        for j, out in enumerate(("power", "delay", "loss")):
            ax = axes[i, j]
            style(ax)
            col, title = COLS[out]
            g = du[du[col].notna()].assign(res=lambda t: t[f"pred_{out}"] - t[col])
            x = g["load_gbps"].clip(lower=8e-6)
            if upf == "USR":
                ax.axvspan(*band, color=BAND_FILL, lw=0, zorder=0)
                ax.axvline(0.091, color=INK2, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
                ax.axvline(0.149, color=INK2, lw=0.6, ls=(0, (4, 2)), zorder=1)
            ax.scatter(x, g["res"], s=2, lw=0, color="#b4b2ab", alpha=0.35, rasterized=True, zorder=2)
            st = g.groupby("level_gbps").agg(x=("load_gbps", "median"), med=("res", "median"),
                                             p5=("res", lambda s: s.quantile(0.05)),
                                             p95=("res", lambda s: s.quantile(0.95))).reset_index()
            st["x"] = st["x"].clip(lower=8e-6)
            ax.fill_between(st["x"], st["p5"], st["p95"], color=SPREAD, lw=0, alpha=0.85, zorder=3, label="5–95th pct")
            ax.plot(st["x"], st["med"], color=MED, lw=1.0, marker="o", ms=2.2, zorder=4, label="per-level median")
            ax.axhline(0, color=INK2, lw=0.6, zorder=3)
            ax.set_xscale("log")
            ax.set_xlim(5e-6, 7)
            ax.xaxis.set_major_locator(FixedLocator([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]))
            ax.set_xticklabels([r"$\approx$0", "$10^{-4}$", "$10^{-3}$", "0.01", "0.1", "1"])
            ax.xaxis.set_minor_formatter(NullFormatter())
            if out == "delay":
                ax.set_yscale("symlog", linthresh=100, linscale=0.6)
                ax.yaxis.set_major_locator(FixedLocator([-1e4, -1e3, -1e2, 0, 1e2, 1e3, 1e4]))
                ax.set_yticklabels(["$-10^4$", "", "$-10^2$", "0", "$10^2$", "", "$10^4$"])
                ax.yaxis.set_minor_formatter(NullFormatter())
            elif out == "loss":
                if upf == "DPDK" and np.all(g["res"] == 0):
                    ax.set_ylim(-1, 1)
                    ax.yaxis.set_major_locator(FixedLocator([-1, 0, 1]))
                    ax.text(0.5, 0.70, "all residuals = 0", transform=ax.transAxes, ha="center", fontsize=6.8, color=INK2)
                else:
                    ax.set_yscale("symlog", linthresh=5, linscale=0.6)
                    ax.yaxis.set_major_locator(FixedLocator([-1e5, -1e3, -10, 0, 10, 1e3, 1e5]))
                    ax.set_yticklabels(["$-10^5$", "$-10^3$", "$-10$", "0", "10", "$10^3$", "$10^5$"])
                    ax.yaxis.set_minor_formatter(NullFormatter())
            if i == 0:
                ax.set_title(title.replace("(", "residual ("), color=INK, pad=4)
            if i == 1:
                ax.set_xlabel("Offered DL load (Gbps)", color=INK2, labelpad=1)
            if j == 0:
                ax.set_ylabel(f"{ROWS[upf]}\npredicted − measured", color=INK2, labelpad=2)
            if i == 1 and j == 0:
                ax.legend(loc="upper left", frameon=False, handlelength=1.4, borderaxespad=0.2)
            if i == 1 and j == 0:
                ax.text(0.091, 1.0, r"$\lambda_{be}$", transform=ax.get_xaxis_transform(), fontsize=6.5, color=INK2, ha="right", va="bottom")
                ax.text(0.149, 1.0, r"$\lambda_{qos}$", transform=ax.get_xaxis_transform(), fontsize=6.5, color=INK2, ha="left", va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(C.OUT / f"residuals_vs_load.{ext}", dpi=600)
    plt.close(fig)


def main():
    p = pd.read_csv(C.OUT / "heldout_predictions.csv.gz")
    d = p[(p["protocol"] == "LOSO") & (p["config"] == "nested")]
    assert (d.loc[d["upf"] == "USR", "pred_delay"] > 0).all() and (d.loc[d["upf"] == "DPDK", "pred_delay"] > 0).all()
    measured_vs_predicted(d)
    residuals_vs_load(d)


if __name__ == "__main__":
    main()
