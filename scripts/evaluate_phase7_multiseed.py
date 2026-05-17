"""Phase 7 — multi-seed test evaluation across all controllers.

Discovers all available checkpoints and runs each one once on the
test slice. Aggregates per-controller (mean ± std across seeds) for
the headline comparison table. The baseline controllers are
deterministic so they contribute one number each.

Output: ``reports/phase-7/multiseed_summary.json`` — structured
per-controller, per-seed, per-cluster KPIs.

The seed→checkpoint mapping is inferred from the directory naming:
    experiments/mappo_seed<N>_<ts>/mappo_best.pt
    experiments/ppo_multi_site_seed<N>_<ts>/ppo_multi_site.zip
    experiments/ppo_single_site_ensemble_<ts>/cluster_<k>/ppo_single_site.zip
       (seed must come from the train command used; passed by --seed-map)

For ensembles, since the dir name does not encode the seed, we map
each ensemble dir to a seed by mtime correlation with the multi-site
dirs we know about (or accept --ensemble-seed-map overrides).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.envs.multi_agent_upf_env import MultiAgentUPFEnv  # noqa: E402
from src.trainers.mappo import Actor  # noqa: E402
from src.baselines.hysteresis import MultiAgentHysteresis  # noqa: E402
from src.baselines.threshold_derivation import (  # noqa: E402
    derive_thresholds,
    load_forecast_mae_gbps,
)
from src.utils.config import load_yaml, project_root  # noqa: E402

K = 10


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def _seed_from_dir(name: str) -> int | None:
    m = re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else None


def discover_checkpoints() -> dict[str, dict[int, Path]]:
    """Return {controller: {seed: ckpt_path}}.

    Ensembles are matched to seeds by the order ensemble dirs were
    created on disk, paired against the seeds we trained
    (we trained ensembles in the order seed 7, 13, 99 after seed 42).
    """
    exp = REPO_ROOT / "experiments"
    out: dict[str, dict[int, Path]] = {
        "mappo": {}, "centralised": {}, "ensemble": {}
    }

    # MAPPO
    for d in sorted(exp.glob("mappo_seed*_*"), key=lambda p: p.stat().st_mtime):
        s = _seed_from_dir(d.name)
        if s is None:
            continue
        cand = d / "mappo_best.pt"
        if cand.exists():
            out["mappo"][s] = cand

    # Centralised PPO
    for d in sorted(
        exp.glob("ppo_multi_site_seed*_*"), key=lambda p: p.stat().st_mtime,
    ):
        s = _seed_from_dir(d.name)
        if s is None:
            continue
        cand = d / "ppo_multi_site.zip"
        if cand.exists():
            out["centralised"][s] = cand

    # Ensemble — prefer dirs that carry the seed in their name
    # (`ppo_single_site_ensemble_seed<N>_<ts>`). Older runs used a
    # seedless dir name and were matched by creation order; that
    # heuristic is kept as a fallback if no seed-tagged dirs exist.
    seed_tagged = sorted(
        exp.glob("ppo_single_site_ensemble_seed*_*"),
        key=lambda p: p.stat().st_mtime,
    )
    if seed_tagged:
        for d in seed_tagged:
            s = _seed_from_dir(d.name)
            if s is None:
                continue
            if all(
                (d / f"cluster_{k}" / "ppo_single_site.zip").exists()
                for k in range(K)
            ):
                out["ensemble"][s] = d
    else:
        ens_dirs = sorted(
            exp.glob("ppo_single_site_ensemble_*"),
            key=lambda p: p.stat().st_mtime,
        )
        seed_order = [42, 7, 13, 99]
        for i, d in enumerate(ens_dirs):
            if i >= len(seed_order):
                break
            if all(
                (d / f"cluster_{k}" / "ppo_single_site.zip").exists()
                for k in range(K)
            ):
                out["ensemble"][seed_order[i]] = d
    return out


# ---------------------------------------------------------------------------
# Policy factories (return dict-action policies for the PettingZoo env)
# ---------------------------------------------------------------------------


def _load_mappo_actor(path: Path) -> Actor:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    a = Actor(ckpt["obs_dim"], ckpt["n_actions"], cfg.get("actor_hidden", 64))
    a.load_state_dict(ckpt["actor"])
    a.eval()
    return a


def _mappo_policy(actor: Actor, agents: list[str]):
    @torch.no_grad()
    def _fn(obs_dict):
        obs = np.stack([obs_dict[a] for a in agents]).astype(np.float32)
        action, _ = actor.act(torch.from_numpy(obs), deterministic=True)
        return {a: int(action[k].item()) for k, a in enumerate(agents)}
    return _fn


def _centralised_policy(model: PPO):
    def _fn(obs_dict):
        obs_flat = np.concatenate(
            [obs_dict[f"cluster_{k}"] for k in range(K)]
        ).astype(np.float32)
        a, _ = model.predict(obs_flat, deterministic=True)
        a = np.asarray(a).reshape(-1)
        return {f"cluster_{k}": int(a[k]) for k in range(K)}
    return _fn


def _ensemble_policy(ensemble_dir: Path):
    models = [
        PPO.load(ensemble_dir / f"cluster_{k}" / "ppo_single_site.zip")
        for k in range(K)
    ]

    def _fn(obs_dict):
        out = {}
        for k in range(K):
            agent = f"cluster_{k}"
            a, _ = models[k].predict(obs_dict[agent], deterministic=True)
            out[agent] = int(np.asarray(a).item())
        return out
    return _fn


def _const_policy(action_id: int):
    def _fn(obs_dict):
        return {a: int(action_id) for a in obs_dict.keys()}
    return _fn


def _threshold_policy(threshold_gbps: float):
    """Per-agent stateless threshold: USR if forecast < threshold, else DPDK.

    Each agent's observation contains the 1-step forecast at index 9
    (current load + 8 history + forecast). No hysteresis, no cooldown
    (i.e. this is the ``ThresholdPolicy`` from UPF_NDT, applied per
    cluster).
    """
    def _fn(obs_dict):
        out = {}
        for agent, obs in obs_dict.items():
            forecast = float(obs[9])
            out[agent] = 1 if forecast < threshold_gbps else 0
        return out
    return _fn


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def rollout(policy_fn, seed: int) -> dict:
    env = MultiAgentUPFEnv(split="test")
    obs, _ = env.reset(seed=seed)
    agents = list(env.possible_agents)
    K_local = env.K
    step_h = env.step_h
    tau = env.tau

    per_c_reward = np.zeros(K_local, dtype=np.float64)
    per_c_energy = np.zeros(K_local, dtype=np.float64)
    per_c_unsafe = np.zeros(K_local, dtype=np.int64)
    per_c_qos = np.zeros(K_local, dtype=np.int64)
    per_c_dpdk = np.zeros(K_local, dtype=np.int64)
    per_c_usr = np.zeros(K_local, dtype=np.int64)
    per_c_switch = np.zeros(K_local, dtype=np.int64)
    last_act: list[int | None] = [None] * K_local
    steps = 0

    while env.agents:
        act = policy_fn(obs)
        obs, r, _t, _u, info = env.step(act)
        for k, a in enumerate(agents):
            per_c_reward[k] += r[a]
            ck = info[a]
            per_c_energy[k] += float(ck["power_watts"]) * step_h
            if not ck["is_safe"]:
                per_c_unsafe[k] += 1
            if ck["q_score"] < tau:
                per_c_qos[k] += 1
            if ck["selected_upf"] == "DPDK":
                per_c_dpdk[k] += 1
            else:
                per_c_usr[k] += 1
            ak = int(act[a])
            if last_act[k] is not None and last_act[k] != ak:
                per_c_switch[k] += 1
            last_act[k] = ak
        steps += 1

    n = max(1, steps)
    return {
        "steps": steps,
        "total_reward_unweighted": float(per_c_reward.sum()),
        "total_energy_wh": float(per_c_energy.sum()),
        "per_cluster_total_reward": per_c_reward.tolist(),
        "per_cluster_energy_wh": per_c_energy.tolist(),
        "per_cluster_unsafe_rate": (per_c_unsafe / n).tolist(),
        "per_cluster_qos_violation_rate": (per_c_qos / n).tolist(),
        "per_cluster_dpdk_rate": (per_c_dpdk / n).tolist(),
        "per_cluster_usr_rate": (per_c_usr / n).tolist(),
        "per_cluster_n_switches": per_c_switch.tolist(),
        "agg_unsafe_rate": float(per_c_unsafe.sum() / (K_local * n)),
        "agg_usr_rate": float(per_c_usr.sum() / (K_local * n)),
        "agg_n_switches": int(per_c_switch.sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--threshold-gbps", type=float, default=None,
        help="Single-threshold value (Gbps). If omitted, derived from the "
             "twin via min(energy_breakeven, qos_limit) - safety_margin.",
    )
    p.add_argument(
        "--safety-margin-mbps", type=float, default=10.0,
        help="Subtracted from the limiting threshold to keep a buffer.",
    )
    p.add_argument(
        "--hysteresis-band-mbps", type=float, default=None,
        help="Manual hysteresis band (Mbps). If omitted, defaults to "
             "2 * forecast MAE (paper-faithful). With our K=10 forecaster "
             "MAE is high enough that the auto band swamps the decision "
             "point; we report both the auto and a narrow tuned variant.",
    )
    p.add_argument(
        "--tuned-band-mbps", type=float, default=20.0,
        help="Narrow hysteresis band for the practical hysteresis variant.",
    )
    p.add_argument(
        "--hysteresis-cooldown", type=int, default=1,
        help="Hysteresis cooldown in steps (UPF_NDT default: 1).",
    )
    p.add_argument("--seed", type=int, default=42,
                   help="Seed used for env reset.")
    p.add_argument(
        "--out-json", type=Path,
        default=REPO_ROOT / "reports" / "phase-7" / "multiseed_summary.json",
    )
    return p.parse_args()


def _derive_baseline_thresholds(safety_margin_mbps: float) -> dict:
    """Run the twin-based threshold derivation and return the spec + meta.

    Builds a temporary DigitalTwin to invoke the surrogate sweep, then
    discards it. The same numbers should hold across env instances
    because the twin is loaded from the same config files.
    """
    from upf_digital_twin import DigitalTwin  # noqa: WPS433

    root = project_root()
    scenario = load_yaml("configs/scenario_rl.yaml")
    paths = load_yaml("configs/digital_twin_paths.yaml")
    twin = DigitalTwin(scenario_cfg=scenario, paths_cfg=paths, project_root=root)
    tf = paths.get("traffic_forecaster", {})
    summary_path = root / tf.get(
        "forecast_eval_summary",
        "data/external/traffic_forecaster/forecast_eval_summary.json",
    )
    alpha = float(
        tf.get("alpha")
        or scenario.get("traffic", {}).get("alpha")
        or 1.0
    )
    mae_gbps = load_forecast_mae_gbps(
        summary_path, alpha_gbps_per_norm=alpha, K=K
    )
    spec = derive_thresholds(
        twin,
        safety_margin_mbps=safety_margin_mbps,
        forecast_mae_gbps=mae_gbps,
    )
    return {
        "decision_gbps": spec.decision_gbps,
        "energy_breakeven_gbps": spec.energy_breakeven_gbps,
        "qos_limit_gbps": spec.qos_limit_gbps,
        "delay_limit_gbps": spec.delay_limit_gbps,
        "t_up_gbps": spec.t_up_gbps,
        "t_down_gbps_auto": spec.t_down_gbps,
        "hysteresis_band_gbps_auto": spec.hysteresis_band_gbps,
        "safety_margin_mbps": spec.safety_margin_mbps,
        "forecast_mae_gbps": spec.forecast_mae_gbps,
        "derived_from": spec.derived_from,
    }


def _aggregate(results: dict[int, dict]) -> dict:
    """Aggregate per-seed results into mean ± std across seeds."""
    seeds = sorted(results.keys())
    if not seeds:
        return {"n_seeds": 0}
    rewards = np.array([results[s]["total_reward_unweighted"] for s in seeds])
    energies = np.array([results[s]["total_energy_wh"] for s in seeds])
    unsafes = np.array([results[s]["agg_unsafe_rate"] for s in seeds])
    usrs = np.array([results[s]["agg_usr_rate"] for s in seeds])
    flips = np.array([results[s]["agg_n_switches"] for s in seeds])
    return {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std(ddof=1)) if len(seeds) > 1 else 0.0,
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "energy_mean": float(energies.mean()),
        "energy_std": float(energies.std(ddof=1)) if len(seeds) > 1 else 0.0,
        "unsafe_mean": float(unsafes.mean()),
        "unsafe_std": float(unsafes.std(ddof=1)) if len(seeds) > 1 else 0.0,
        "usr_mean": float(usrs.mean()),
        "flips_mean": float(flips.mean()),
        "per_seed": results,
    }


def main() -> int:
    args = _parse_args()
    print("Discovering checkpoints...")
    ckpts = discover_checkpoints()
    for c, sd in ckpts.items():
        print(f"  {c}: {len(sd)} seeds  ({sorted(sd.keys())})")

    # Build agent names once.
    env_meta = MultiAgentUPFEnv(split="test")
    agents = list(env_meta.possible_agents)
    del env_meta

    all_results: dict[str, dict[int, dict]] = {
        "MAPPO": {},
        "IPPO-ensemble": {},
        "centralised-PPO": {},
    }

    # Trained controllers
    for seed, path in ckpts["mappo"].items():
        print(f"  evaluating MAPPO seed={seed}")
        actor = _load_mappo_actor(path)
        all_results["MAPPO"][seed] = rollout(
            _mappo_policy(actor, agents), seed=args.seed
        )

    for seed, path in ckpts["centralised"].items():
        print(f"  evaluating centralised-PPO seed={seed}")
        model = PPO.load(path)
        all_results["centralised-PPO"][seed] = rollout(
            _centralised_policy(model), seed=args.seed
        )

    for seed, path in ckpts["ensemble"].items():
        print(f"  evaluating IPPO-ensemble seed={seed}")
        all_results["IPPO-ensemble"][seed] = rollout(
            _ensemble_policy(path), seed=args.seed
        )

    # ------------------------------------------------------------------
    # Threshold + hysteresis baselines — derived from the twin
    # ------------------------------------------------------------------
    derived = _derive_baseline_thresholds(args.safety_margin_mbps)
    print("Derived baseline operating points:")
    print(f"  energy break-even:    {derived['energy_breakeven_gbps']*1000:6.2f} Mbps")
    print(f"  QoS limit:            {derived['qos_limit_gbps']*1000:6.2f} Mbps")
    print(f"  decision threshold:   {derived['decision_gbps']*1000:6.2f} Mbps")
    print(f"  forecast MAE (gbps):  {derived['forecast_mae_gbps']}")
    print(f"  auto hysteresis band: {derived['hysteresis_band_gbps_auto']*1000:6.2f} Mbps")

    decision_gbps = (
        args.threshold_gbps if args.threshold_gbps is not None
        else derived["decision_gbps"]
    )

    auto_t_up = derived["t_up_gbps"]
    auto_t_down = derived["t_down_gbps_auto"]
    # The auto band on K=10 is so wide (2 * MAE_K=10 ≈ 220 Mbps) it
    # swamps the decision point and t_down collapses to 0 — hysteresis
    # then degenerates to always-DPDK after the first switch. We also
    # report a narrow tuned variant so the controller class is
    # represented at a sensible operating point.
    tuned_band_gbps = float(args.tuned_band_mbps) / 1000.0
    tuned_t_up = decision_gbps
    tuned_t_down = max(0.0, decision_gbps - tuned_band_gbps)

    threshold_label = f"threshold(derived={decision_gbps*1000:.1f}Mbps)"
    hyst_auto_label = (
        f"hysteresis(t_up={auto_t_up*1000:.0f},t_down={auto_t_down*1000:.0f},"
        f"cd={args.hysteresis_cooldown})"
    )
    hyst_tuned_label = (
        f"hysteresis(t_up={tuned_t_up*1000:.0f},t_down={tuned_t_down*1000:.0f},"
        f"cd={args.hysteresis_cooldown})"
    )

    baselines: dict[str, callable] = {
        "always-DPDK": _const_policy(0),
        threshold_label: _threshold_policy(decision_gbps),
        hyst_auto_label: MultiAgentHysteresis(
            agents=agents,
            t_up_gbps=auto_t_up,
            t_down_gbps=auto_t_down,
            cooldown_steps=args.hysteresis_cooldown,
        ),
        hyst_tuned_label: MultiAgentHysteresis(
            agents=agents,
            t_up_gbps=tuned_t_up,
            t_down_gbps=tuned_t_down,
            cooldown_steps=args.hysteresis_cooldown,
        ),
    }
    for name, fn in baselines.items():
        print(f"  evaluating baseline {name}")
        # Reset stateful baselines before each rollout (no-op for stateless).
        if hasattr(fn, "reset"):
            fn.reset()
        all_results[name] = {0: rollout(fn, seed=args.seed)}

    # Aggregate
    aggregated = {
        name: _aggregate(seeds) for name, seeds in all_results.items()
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "split": "test",
                "seed_env": args.seed,
                "threshold_decision_gbps": decision_gbps,
                "derived": derived,
                "tuned_hysteresis_band_mbps": float(args.tuned_band_mbps),
                "hysteresis_cooldown_steps": int(args.hysteresis_cooldown),
                "K": K,
                "aggregated": aggregated,
            },
            indent=2,
            default=str,
        )
    )

    # Pretty print
    print("\n" + "=" * 92)
    print(
        f"{'Controller':<24s}{'n_seeds':>9s}"
        f"{'reward_mean':>14s}{'reward_std':>13s}"
        f"{'energy_mean':>13s}{'unsafe%':>10s}"
    )
    print("-" * 92)
    for name, agg in aggregated.items():
        if agg["n_seeds"] == 0:
            continue
        print(
            f"{name:<24s}{agg['n_seeds']:>9d}"
            f"{agg['reward_mean']:>14.2f}"
            f"{agg['reward_std']:>13.2f}"
            f"{agg['energy_mean']:>13.2f}"
            f"{agg['unsafe_mean'] * 100:>9.2f}%"
        )
    print("=" * 92)
    print(f"\nWrote {args.out_json.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
