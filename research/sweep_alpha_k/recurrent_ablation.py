#!/usr/bin/env python3
"""Recurrent vs feed-forward policy ablation (review #28).

The feed-forward controller reads an explicit eight-step load history from its
observation. The recurrent alternative reads only the current observation and
carries temporal state in an LSTM. Both are trained identically otherwise and
evaluated once each on the held-out test slice.
"""
from __future__ import annotations
import argparse, json, time, warnings, sys
from pathlib import Path
warnings.filterwarnings("ignore")

REPO = Path("/home/ubuntu/UpfRLControllers")
sys.path.insert(0, str(REPO))
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
from src.envs.single_site_upf_env import SingleSiteUPFEnv
from src.trainers.ppo_single_site import rollout_episode

HP = dict(learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10,
          gamma=0.995, gae_lambda=0.9, clip_range=0.2, ent_coef=0.01, vf_coef=0.5)


def make_env(cluster, split, seed):
    def _t():
        return Monitor(SingleSiteUPFEnv(cluster_idx=cluster, horizon_idx=0, split=split))
    return _t


def recurrent_policy(model):
    """Deterministic rollout fn carrying the LSTM state across the episode."""
    state = {"s": None, "start": np.ones((1,), dtype=bool)}
    def _fn(obs):
        a, state["s"] = model.predict(np.asarray(obs)[None, :], state=state["s"],
                                      episode_start=state["start"], deterministic=True)
        state["start"] = np.zeros((1,), dtype=bool)
        return int(a[0])
    return _fn


def ff_policy(model):
    def _fn(obs):
        a, _ = model.predict(np.asarray(obs), deterministic=True)
        return int(a)
    return _fn


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cluster", type=int, default=0)
    p.add_argument("--seeds", default="1,7,13,42,99")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--out", type=Path, default=REPO / "reports" / "phase-7" / "recurrent_ablation.json")
    a = p.parse_args()

    out = {"cluster": a.cluster, "total_timesteps": a.steps, "runs": []}
    for seed in [int(s) for s in a.seeds.split(",")]:
        for kind in ("feedforward", "recurrent"):
            env = DummyVecEnv([make_env(a.cluster, "train", seed)])
            t0 = time.time()
            if kind == "recurrent":
                m = RecurrentPPO("MlpLstmPolicy", env, seed=seed, device="cpu",
                                 verbose=0, **HP)
            else:
                m = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **HP)
            m.learn(total_timesteps=a.steps, progress_bar=False)
            train_s = time.time() - t0
            pol = recurrent_policy(m) if kind == "recurrent" else ff_policy(m)
            r = rollout_episode(cluster_idx=a.cluster, horizon_idx=0, policy=pol,
                                seed=0, split="test")
            out["runs"].append({"seed": seed, "kind": kind, "train_seconds": round(train_s, 1),
                                "total_reward": r["total_reward"],
                                "total_energy_wh": r["total_energy_wh"],
                                "unsafe_rate": r.get("unsafe_rate", 0.0)})
            print(f"[{kind:>12} seed {seed:>3}] reward={r['total_reward']:>9.1f} "
                  f"energy={r['total_energy_wh']:>7.1f} unsafe={r.get('unsafe_rate',0)*100:.2f}% "
                  f"train={train_s:.0f}s", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
