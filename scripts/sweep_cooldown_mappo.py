"""Sweep MAPPO over (cooldown_period, cooldown_cost) on a single seed.

Used by P1 #7 (paper-draft README) to show the headline result is not
load-bearing on the hand-picked cooldown defaults under the revised
physics-grounded reward.

Each invocation runs ONE (period, cost) combo so launches can be
parallelised. Results land under
`experiments/mappo_cd_p{period}_c{cost}_seed{seed}_<utc-ts>/` and the
evaluation summary is written to
`reports/phase-7/cooldown_sweep_p{period}_c{cost}_seed{seed}.json`.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.envs.multi_agent_upf_env import MultiAgentUPFEnv  # noqa: E402
from src.trainers.mappo import MAPPO, MAPPOConfig  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cooldown-period", type=int, required=True)
    p.add_argument("--cooldown-cost", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.9)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.05)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--actor-hidden", type=int, default=64)
    p.add_argument("--critic-hidden", type=int, default=128)
    p.add_argument("--reward-scale", type=float, default=0.01)
    p.add_argument("--eval-freq", type=int, default=10)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def rollout_test(actor, cfg_override: dict, device: str = "cpu") -> dict:
    """Deterministic test rollout under the swept scenario_cfg."""
    env = MultiAgentUPFEnv(split="test", scenario_cfg=cfg_override)
    obs_dict, _ = env.reset(seed=42)
    K = env.K
    step_h = env.step_h
    tau = env.tau
    dev = torch.device(device)
    actor = actor.to(dev)

    per_c_reward = np.zeros(K)
    per_c_energy = np.zeros(K)
    per_c_unsafe = np.zeros(K, dtype=int)
    per_c_usr = np.zeros(K, dtype=int)
    per_c_switch = np.zeros(K, dtype=int)
    last_act: list[int | None] = [None] * K
    steps = 0
    while env.agents:
        obs = np.stack(
            [obs_dict[a] for a in env.possible_agents]
        ).astype(np.float32)
        obs_t = torch.from_numpy(obs).to(dev)
        action, _ = actor.act(obs_t, deterministic=True)
        action_dict = {
            a: int(action[k].item()) for k, a in enumerate(env.possible_agents)
        }
        obs_dict, reward_dict, _t, _u, info_dict = env.step(action_dict)
        for k, a in enumerate(env.possible_agents):
            per_c_reward[k] += reward_dict[a]
            ck = info_dict[a]
            per_c_energy[k] += float(ck["power_watts"]) * step_h
            if not ck["is_safe"]:
                per_c_unsafe[k] += 1
            if ck["selected_upf"] == "USR":
                per_c_usr[k] += 1
            ak = int(action[k].item())
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
        "per_cluster_n_switches": per_c_switch.tolist(),
        "agg_unsafe_rate": float(per_c_unsafe.sum() / (K * n)),
        "agg_usr_rate": float(per_c_usr.sum() / (K * n)),
        "agg_n_switches": int(per_c_switch.sum()),
    }


def main() -> int:
    args = parse_args()

    base_cfg = load_yaml("configs/scenario_rl.yaml")
    cfg_override = copy.deepcopy(base_cfg)
    cfg_override["reward_weights"]["cooldown_period"] = args.cooldown_period
    cfg_override["reward_weights"]["cooldown_cost"] = args.cooldown_cost

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_dir = (
        REPO_ROOT
        / "experiments"
        / f"mappo_cd_p{args.cooldown_period}_c{args.cooldown_cost}_seed{args.seed}_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] period={args.cooldown_period} cost={args.cooldown_cost} seed={args.seed}")
    print(f"[sweep] out={out_dir}")

    env = MultiAgentUPFEnv(split="train", scenario_cfg=cfg_override)
    eval_env = MultiAgentUPFEnv(split="val", scenario_cfg=cfg_override)

    cfg = MAPPOConfig(
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        actor_hidden=args.actor_hidden,
        critic_hidden=args.critic_hidden,
        reward_scale=args.reward_scale,
        device=args.device,
        seed=args.seed,
        eval_freq=args.eval_freq,
        train_split="train",
        eval_split="val",
    )
    trainer = MAPPO(env, cfg)
    trainer.learn(
        out_dir=out_dir, eval_env=eval_env, verbose=1, progress_bar=False
    )

    # Reload best-of-val before test rollout
    best_path = out_dir / "mappo_best.pt"
    if best_path.exists():
        trainer.load(best_path)

    test_metrics = rollout_test(trainer.actor, cfg_override, device=args.device)
    out_summary = {
        "cooldown_period": args.cooldown_period,
        "cooldown_cost": args.cooldown_cost,
        "seed": args.seed,
        "split": "test",
        "best_val_return": float(trainer.best_eval_return),
        "experiment_dir": str(out_dir.relative_to(REPO_ROOT)),
        "test": test_metrics,
    }
    summary_path = (
        REPO_ROOT
        / "reports"
        / "phase-7"
        / f"cooldown_sweep_p{args.cooldown_period}_c{args.cooldown_cost}_seed{args.seed}.json"
    )
    summary_path.write_text(json.dumps(out_summary, indent=2))
    print(f"[sweep] wrote {summary_path.relative_to(REPO_ROOT)}")
    print(
        f"[sweep] test reward={test_metrics['total_reward_unweighted']:.2f}"
        f" energy_wh={test_metrics['total_energy_wh']:.2f}"
        f" n_switches={test_metrics['agg_n_switches']}"
        f" unsafe={test_metrics['agg_unsafe_rate']*100:.2f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
