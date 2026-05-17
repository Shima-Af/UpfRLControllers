"""Generate Phase 6 (MAPPO) supervisor-facing figures.

Compares MAPPO vs the best Phase 3/4 baselines on the test slice.
Same figure schema as Phase 3 with one addition: a training-curve
panel showing the val-set return over the course of training.

Outputs land in ``reports/phase-6/figures/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.envs.multi_agent_upf_env import MultiAgentUPFEnv  # noqa: E402
from src.envs.multi_site_upf_env import MultiSiteUPFEnv  # noqa: E402
from src.trainers.mappo import Actor  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "phase-6" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLICY_COLOR = {
    "MAPPO":              "#ef4444",
    "IPPO-ensemble":      "#22c55e",
    "centralised-PPO":    "#0ea5e9",
    "all-DPDK":           "#10b981",
}
POLICY_ORDER = [
    "MAPPO", "IPPO-ensemble", "centralised-PPO", "all-DPDK"
]


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


def _latest(glob: str) -> Path | None:
    matches = sorted(
        (REPO_ROOT / "experiments").glob(glob),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _load_actor(ckpt_path: Path) -> Actor:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    a = Actor(
        obs_dim=ckpt["obs_dim"],
        n_actions=ckpt["n_actions"],
        hidden=cfg.get("actor_hidden", 64),
    )
    a.load_state_dict(ckpt["actor"])
    a.eval()
    return a


def _mappo_policy_fn(actor: Actor, possible_agents: list[str]):
    @torch.no_grad()
    def _fn(obs_dict):
        obs = np.stack([obs_dict[a] for a in possible_agents]).astype(np.float32)
        action, _ = actor.act(torch.from_numpy(obs), deterministic=True)
        return {a: int(action[k].item()) for k, a in enumerate(possible_agents)}

    return _fn


def _ensemble_policy_fn(ckpts: list[Path]):
    models = [PPO.load(p) for p in ckpts]

    def _fn(obs_dict):
        out = {}
        for k, agent in enumerate(obs_dict.keys()):
            sub_obs = obs_dict[agent]
            a, _ = models[k].predict(sub_obs, deterministic=True)
            out[agent] = int(np.asarray(a).item())
        return out

    return _fn


def _centralised_policy_fn(model: PPO, K: int):
    def _fn(obs_dict):
        # Re-stack per-agent obs into the centralised env's flat layout.
        obs_flat = np.concatenate(
            [obs_dict[f"cluster_{k}"] for k in range(K)]
        ).astype(np.float32)
        a, _ = model.predict(obs_flat, deterministic=True)
        a = np.asarray(a).reshape(-1)
        return {f"cluster_{k}": int(a[k]) for k in range(K)}

    return _fn


def _const_policy_fn(action_id: int, K: int):
    def _fn(obs_dict):
        return {f"cluster_{k}": int(action_id) for k in range(K)}

    return _fn


def _rollout_collect(policy_fn, *, split: str, seed: int):
    env = MultiAgentUPFEnv(split=split)
    obs, _ = env.reset(seed=seed)
    K = env.K
    n_steps = env._N  # noqa: SLF001
    actions = np.zeros((n_steps, K), dtype=np.int8)
    rewards = np.zeros(n_steps, dtype=np.float64)
    cum = np.zeros(n_steps, dtype=np.float64)
    per_c_reward = np.zeros((n_steps, K), dtype=np.float64)
    loads = np.zeros((n_steps, K), dtype=np.float64)
    cum_r = 0.0
    t = 0
    while env.agents:
        act = policy_fn(obs)
        obs, r, _term, _trunc, info = env.step(act)
        for k, a in enumerate(env.possible_agents):
            actions[t, k] = act[a]
            per_c_reward[t, k] = r[a]
            loads[t, k] = float(info[a]["actual_load_gbps"])
        rewards[t] = sum(r.values())
        cum_r += rewards[t]
        cum[t] = cum_r
        t += 1
    return {
        "K": K, "n_steps": t,
        "actions": actions[:t], "rewards": rewards[:t], "cum": cum[:t],
        "per_cluster_reward": per_c_reward[:t], "loads": loads[:t],
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mappo-ckpt", type=Path, default=None)
    p.add_argument("--multi-ckpt", type=Path, default=None,
                   help="Phase 3 centralised PPO checkpoint")
    p.add_argument("--ensemble-dir", type=Path, default=None,
                   help="Phase 4 IPPO ensemble dir")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    print(f"Generating Phase 6 figures in {OUT_DIR} (split={args.split!r})")

    K = 10
    # Resolve checkpoints
    mappo_ckpt = args.mappo_ckpt
    if mappo_ckpt is None:
        latest = _latest("mappo_seed*")
        if latest is not None:
            cand = latest / "mappo_best.pt"
            mappo_ckpt = cand if cand.exists() else (latest / "mappo_final.pt")
    if mappo_ckpt is None or not mappo_ckpt.exists():
        raise SystemExit("No MAPPO checkpoint found.")

    multi_ckpt = args.multi_ckpt
    if multi_ckpt is None:
        latest = _latest("ppo_multi_site_*")
        if latest is not None:
            multi_ckpt = latest / "ppo_multi_site.zip"

    ensemble_dir = args.ensemble_dir
    if ensemble_dir is None:
        ensemble_dir = _latest("ppo_single_site_ensemble_*")
    ensemble_ckpts: list[Path] = []
    if ensemble_dir is not None and ensemble_dir.is_dir():
        for k in range(K):
            cand = ensemble_dir / f"cluster_{k}" / "ppo_single_site.zip"
            if cand.exists():
                ensemble_ckpts.append(cand)

    print(f"  MAPPO ckpt: {mappo_ckpt.relative_to(REPO_ROOT)}")
    if multi_ckpt:
        print(f"  centralised PPO ckpt: {multi_ckpt.relative_to(REPO_ROOT)}")
    if ensemble_dir and ensemble_dir.is_dir():
        print(f"  ensemble dir: {ensemble_dir.relative_to(REPO_ROOT)}")

    actor = _load_actor(mappo_ckpt)
    env_meta = MultiAgentUPFEnv(split=args.split)
    possible_agents = list(env_meta.possible_agents)
    del env_meta

    policies = {
        "MAPPO": _mappo_policy_fn(actor, possible_agents),
        "all-DPDK": _const_policy_fn(0, K),
    }
    if ensemble_ckpts and len(ensemble_ckpts) == K:
        policies["IPPO-ensemble"] = _ensemble_policy_fn(ensemble_ckpts)
    if multi_ckpt is not None and multi_ckpt.exists():
        policies["centralised-PPO"] = _centralised_policy_fn(
            PPO.load(multi_ckpt), K=K
        )

    print("Running rollouts...")
    rolls: dict[str, dict] = {}
    for name in POLICY_ORDER:
        if name not in policies:
            continue
        rolls[name] = _rollout_collect(
            policies[name], split=args.split, seed=args.seed
        )
        print(f"  {name:<22s} cum_unweighted_r={rolls[name]['cum'][-1]:>10.2f}")

    n_steps = next(iter(rolls.values()))["n_steps"]
    t = np.arange(n_steps)

    # ------------------------------------------------------------------
    # fig1 — total reward bar chart
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 2.6), dpi=150)
    labels = [n for n in POLICY_ORDER if n in rolls]
    values = [rolls[n]["cum"][-1] for n in labels]
    colors = [POLICY_COLOR[n] for n in labels]
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=1.0)
    for bar, v in zip(bars, values, strict=True):
        ax.text(
            v + max(50, abs(v) * 0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{v:+.2f}",
            color="white", fontsize=9, fontweight="bold",
            va="center", ha="left",
        )
    ax.invert_yaxis()
    _setup_axes(
        ax,
        f"Unweighted total reward per policy ({args.split} slice, K={K})",
        "total reward (closer to 0 is better)", "",
    )
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_total_reward.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig1_total_reward.png")

    # ------------------------------------------------------------------
    # fig2 — per-cluster reward grouped bar
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 3.4), dpi=150)
    pol_in_fig = [n for n in POLICY_ORDER if n in rolls]
    n_p = len(pol_in_fig)
    w = 0.8 / n_p
    x = np.arange(K)
    for i, name in enumerate(pol_in_fig):
        per_c = rolls[name]["per_cluster_reward"].sum(axis=0)
        offset = (i - (n_p - 1) / 2) * w
        ax.bar(x + offset, per_c, width=w, color=POLICY_COLOR[name], label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{k}" for k in range(K)])
    _setup_axes(
        ax,
        f"Per-cluster total reward ({args.split} slice)",
        "cluster", "summed reward over episode",
    )
    ax.legend(fontsize=8, frameon=False, ncol=n_p)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_per_cluster_reward.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig2_per_cluster_reward.png")

    # ------------------------------------------------------------------
    # fig3 — MAPPO action heatmap
    # ------------------------------------------------------------------
    actions = rolls["MAPPO"]["actions"]
    fig, ax = plt.subplots(figsize=(9.0, 3.6), dpi=150)
    im = ax.imshow(
        actions.T, aspect="auto", cmap="RdYlGn_r",
        interpolation="nearest", vmin=0, vmax=1,
    )
    ax.set_yticks(np.arange(K))
    ax.set_yticklabels([f"c{k}" for k in range(K)])
    _setup_axes(
        ax,
        f"MAPPO action over time ({args.split} slice; green = DPDK, red = USR)",
        "timestep", "cluster",
    )
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=[0, 1])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_action_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig3_action_heatmap.png")

    # ------------------------------------------------------------------
    # fig4 — cumulative reward
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.0, 3.6), dpi=150)
    for name in pol_in_fig:
        ax.plot(t, rolls[name]["cum"], color=POLICY_COLOR[name],
                linewidth=1.4, label=name)
    _setup_axes(
        ax, f"Cumulative reward over the {args.split} episode",
        "timestep", "cumulative reward",
    )
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_cumulative_reward.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig4_cumulative_reward.png")

    # ------------------------------------------------------------------
    # fig5 — MAPPO training curve (val return over training)
    # ------------------------------------------------------------------
    eval_log_path = mappo_ckpt.parent / "eval_log.json"
    if eval_log_path.exists():
        log = json.loads(eval_log_path.read_text())
        if log:
            steps = [e["step"] for e in log]
            rets = [e["eval_return"] for e in log]
            fig, ax = plt.subplots(figsize=(9.0, 3.0), dpi=150)
            ax.plot(steps, rets, color="#0ea5e9", linewidth=1.4)
            best = max(rets)
            ax.axhline(best, color="#ef4444", linestyle="--", linewidth=0.7,
                       alpha=0.7, label=f"best = {best:.2f}")
            _setup_axes(
                ax, "MAPPO val-set return during training",
                "training step", "deterministic eval return",
            )
            ax.legend(fontsize=8, frameon=False, loc="lower right")
            fig.tight_layout()
            fig.savefig(OUT_DIR / "fig5_training_curve.png",
                        bbox_inches="tight")
            plt.close(fig)
            print("  fig5_training_curve.png")
    else:
        print("  (no eval_log.json found — skipping fig5)")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
