"""Short PPO fine-tune from an existing single-site c0 checkpoint,
under caller-supplied reward weights.

Slice 0 — the simplest fine-tune we can defend: load the base, swap
the reward via a patched scenario_cfg, call `model.learn(timesteps)`,
save. No lr schedule override, no curriculum, no callback gymnastics.
"""

from __future__ import annotations

import datetime as dt
from copy import deepcopy
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.single_site_upf_env import SingleSiteUPFEnv
from src.utils.config import load_yaml, project_root


def _make_env(
    scenario_cfg: dict,
    cluster_idx: int,
    split: str,
    seed: int,
):
    def _thunk():
        env = SingleSiteUPFEnv(
            cluster_idx=cluster_idx,
            horizon_idx=0,
            scenario_cfg=scenario_cfg,
            split=split,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _thunk


def finetune(
    base_checkpoint: str | Path,
    reward_weights: dict,
    *,
    cluster_idx: int = 0,
    timesteps: int = 20_000,
    seed: int = 42,
    train_split: str = "train",
    out_dir: str | Path | None = None,
    progress_bar: bool = False,
) -> Path:
    """Returns the path to the saved fine-tuned checkpoint."""
    scenario_cfg = deepcopy(load_yaml("configs/scenario_rl.yaml"))
    scenario_cfg.setdefault("reward_weights", {}).update(reward_weights)

    env = DummyVecEnv(
        [_make_env(scenario_cfg, cluster_idx, train_split, seed)]
    )
    model = PPO.load(str(base_checkpoint), env=env)
    model.learn(total_timesteps=timesteps, progress_bar=progress_bar)

    if out_dir is None:
        ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        out_dir = project_root() / "experiments" / f"intent_finetune_seed{seed}_{ts}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = out_dir / "model.zip"
    model.save(str(ckpt))
    return ckpt
