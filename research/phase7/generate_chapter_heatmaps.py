"""Regenerate the controller chapter's two action-heatmap figures under the v0.4
twin, in the paper's Latin Modern font (usetex).

  fig_mappo_action_heatmap.png    — MAPPO per-site, per-step actions (test period)
  fig_p3_centralised_collapse.png — centralised PPO actions (the c0 collapse)

Reuses the rollout helpers from research/phase6/generate_figures.py so the
action trajectories are produced exactly as the phase reports did; only the twin
(v0.4, via the installed package) and the rendering font differ. The action
patterns are qualitative and unchanged by the switching-cost refinement; this
regeneration is for numerical currency and font consistency.

    python research/phase7/generate_chapter_heatmaps.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO  # noqa: E402
from research.phase6.generate_figures import (  # noqa: E402
    _load_actor, _mappo_policy_fn, _centralised_policy_fn,
    _rollout_collect, _latest,
)

OUT = REPO_ROOT / "reports/chapter-controllers/figures"
DPDK_GREEN, USR_RED = "#16a34a", "#dc2626"


def _style() -> None:
    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif",
        "text.latex.preamble": r"\usepackage{lmodern}",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 150,
    })


def _heatmap(actions: np.ndarray, title: str, out: Path) -> None:
    """actions: (T, K) of {0=DPDK, 1=USR}. Rendered clusters x time."""
    K = actions.shape[1]
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    ax.imshow(actions.T, aspect="auto", interpolation="nearest",
              cmap=ListedColormap([DPDK_GREEN, USR_RED]), vmin=0, vmax=1)
    ax.set_yticks(range(K)); ax.set_yticklabels([f"c{k}" for k in range(K)])
    ax.set_xlabel(r"15-minute step (test period)")
    ax.set_ylabel(r"MEC cluster")
    ax.set_title(title)
    ax.tick_params(colors="#64748b")
    # legend proxies
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=DPDK_GREEN, label="DPDK"),
                       Patch(color=USR_RED, label="USR")],
              frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, -0.38), fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    usr_frac = actions.mean() * 100
    print(f"  {out.name}: {actions.shape[0]} steps, USR {usr_frac:.1f}% of decisions")


def main() -> int:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)

    # MAPPO
    mp = _latest("mappo_seed*")
    ck = (mp / "mappo_best.pt") if (mp / "mappo_best.pt").exists() else (mp / "mappo_final.pt")
    actor = _load_actor(ck)
    env0 = _rollout_collect.__globals__  # not needed; kept explicit below
    from src.envs.multi_agent_upf_env import MultiAgentUPFEnv
    K = len(MultiAgentUPFEnv(split="test").possible_agents)
    roll = _rollout_collect(_mappo_policy_fn(actor, [f"cluster_{k}" for k in range(K)]),
                            split="test", seed=42)
    _heatmap(roll["actions"], r"MAPPO per-site actions --- intelligent low-load USR pattern",
             OUT / "fig_mappo_action_heatmap.png")

    # Centralised PPO
    cp = _latest("ppo_multi_site_*")
    model = PPO.load(cp / "ppo_multi_site.zip", device="cpu")
    roll_c = _rollout_collect(_centralised_policy_fn(model, K), split="test", seed=42)
    _heatmap(roll_c["actions"], r"Centralised PPO actions --- collapse to always-USR on the busiest cluster",
             OUT / "fig_p3_centralised_collapse.png")

    print(f"wrote 2 heatmaps to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
