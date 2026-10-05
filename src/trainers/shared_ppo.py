"""Shared-PPO — MAPPO with the centralised critic replaced by a local critic.

Ablation for the WCNC 2027 revision. Isolates the contribution of MAPPO's
centralised critic from the contribution of sharing one actor across sites.

  Actor  : identical to MAPPO — one ``Actor`` (14 -> 64 -> 64 -> 2) shared by
           all K agents, fed each agent's own 14-dim observation.
  Critic : parameter-shared across agents like the actor, but fed only the
           corresponding agent's 14-dim local observation
           (14 -> 128 -> 128 -> 1), instead of MAPPO's 140-dim concatenated
           fleet state (140 -> 128 -> 128 -> K).

Everything else is MAPPO's code path, unchanged: the same PettingZoo env and
reward, rollout of n_steps env steps pooled over (T, K), per-agent GAE,
advantage normalisation over (T, K), minibatches of 256 agent-steps, 10
epochs, one Adam optimiser over actor + critic with the same learning rate,
eps, gradient clipping and loss weights, reward scaling, evaluation cadence and
best-of-validation checkpointing. Seeding is identical, so for a given seed the
actor starts from exactly the same weights as MAPPO's.

The four methods below are copies of ``MAPPO``'s with only the critic
input changed. ``src/trainers/mappo.py`` is not modified.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from src.envs.multi_agent_upf_env import MultiAgentUPFEnv
from src.trainers.mappo import MAPPO, MAPPOConfig, Actor, Rollout, _orthogonal_init


class LocalCritic(nn.Module):
    """Shared critic that maps one agent's local observation -> V(o_k)."""

    def __init__(self, obs_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)
        for layer in self.body:
            if isinstance(layer, nn.Linear):
                _orthogonal_init(layer)
        _orthogonal_init(self.head, gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(obs)).squeeze(-1)


class SharedPPO(MAPPO):
    """Shared actor + shared *local* critic; decentralised execution."""

    critic_type = "local"

    def __init__(self, env: MultiAgentUPFEnv, cfg: MAPPOConfig) -> None:  # noqa: D107
        self.env = env
        self.cfg = cfg
        self.K = env.K
        self.obs_dim = env._obs_dim_per  # noqa: SLF001
        self.state_dim = self.K * self.obs_dim
        self.n_actions = int(env.action_space(env.possible_agents[0]).n)
        self.device = torch.device(cfg.device)

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.actor = Actor(self.obs_dim, self.n_actions, cfg.actor_hidden).to(
            self.device
        )
        # --- the only architectural difference from MAPPO ---
        self.critic = LocalCritic(self.obs_dim, cfg.critic_hidden).to(self.device)
        self.opt = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.learning_rate,
            eps=1e-5,
        )

        self.buffer = Rollout(
            n_steps=cfg.n_steps,
            K=self.K,
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
        )

        self.global_step = 0
        self.best_eval_return = -np.inf

    # ------------------------------------------------------------------

    def collect_rollout(self, obs: np.ndarray, state: np.ndarray):
        self.buffer.reset_cursor()
        ep_return = 0.0
        n_done = 0

        for _ in range(self.cfg.n_steps):
            obs_t = torch.from_numpy(obs).to(self.device)            # (K, obs_dim)

            with torch.no_grad():
                action, log_prob = self.actor.act(obs_t, deterministic=False)
                value = self.critic(obs_t)                           # (K,)  local

            action_dict = {
                a: int(action[k].item())
                for k, a in enumerate(self.env.possible_agents)
            }
            next_obs_dict, reward_dict, term, trunc, _info = self.env.step(action_dict)
            reward_arr = np.array(
                [reward_dict[a] for a in self.env.possible_agents],
                dtype=np.float32,
            ) * self.cfg.reward_scale
            done = float(any(trunc.values()) or any(term.values()))

            self.buffer.add(
                obs=obs,
                state=state,
                action=action.cpu().numpy(),
                log_prob=log_prob.cpu().numpy(),
                reward=reward_arr,
                value=value.cpu().numpy(),
                done=done,
            )

            ep_return += float(reward_arr.sum()) / self.cfg.reward_scale
            self.global_step += 1

            if done:
                next_obs_dict, _ = self.env.reset(
                    seed=self.cfg.seed + self.global_step
                )
                n_done += 1

            obs = self._obs_dict_to_array(next_obs_dict)
            state = self.env.state()

        return obs, state, ep_return, {"episodes_in_rollout": float(n_done)}

    def update(self) -> dict[str, float]:
        T, K = self.cfg.n_steps, self.K
        obs = torch.from_numpy(self.buffer.obs.reshape(T * K, self.obs_dim)).to(
            self.device
        )
        actions = torch.from_numpy(self.buffer.actions.reshape(T * K)).to(self.device)
        old_log_probs = torch.from_numpy(self.buffer.log_probs.reshape(T * K)).to(
            self.device
        )
        advantages = torch.from_numpy(self.buffer.advantages.reshape(T * K)).to(
            self.device
        )
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False).clamp_min(1e-8)
        advantages = (advantages - adv_mean) / adv_std
        returns = torch.from_numpy(self.buffer.returns).to(self.device)  # (T, K)

        n_samples = T * K
        idx = np.arange(n_samples)
        t_idx = np.repeat(np.arange(T), K)
        k_idx = np.tile(np.arange(K), T)

        stats = {
            "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "approx_kl": 0.0, "clip_fraction": 0.0, "n_minibatches": 0.0,
        }

        for _ in range(self.cfg.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n_samples, self.cfg.minibatch_size):
                mb = idx[start: start + self.cfg.minibatch_size]
                mb_obs = obs[mb]
                mb_actions = actions[mb]
                mb_old_lp = old_log_probs[mb]
                mb_adv = advantages[mb]

                new_lp, entropy = self.actor.evaluate(mb_obs, mb_actions)

                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range
                ) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                # Local critic: V(o_k) for each (t, k) sample in the minibatch.
                vals_per_sample = self.critic(mb_obs)             # (B,)
                target_per_sample = returns[t_idx[mb], k_idx[mb]]
                value_loss = (vals_per_sample - target_per_sample).pow(2).mean()

                loss = (
                    policy_loss
                    + self.cfg.vf_coef * value_loss
                    + self.cfg.ent_coef * entropy_loss
                )

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.cfg.max_grad_norm,
                )
                self.opt.step()

                with torch.no_grad():
                    log_ratio = new_lp - mb_old_lp
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > self.cfg.clip_range)
                        .float()
                        .mean()
                    )

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(-entropy_loss.item())
                stats["approx_kl"] += float(approx_kl.item())
                stats["clip_fraction"] += float(clip_frac.item())
                stats["n_minibatches"] += 1.0

        n_mb = max(1.0, stats["n_minibatches"])
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
            stats[key] /= n_mb
        return stats

    def learn(
        self,
        out_dir: Path | None = None,
        eval_env: MultiAgentUPFEnv | None = None,
        verbose: int = 1,
        progress_bar: bool = True,
    ) -> None:
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        obs_dict, _info = self.env.reset(seed=self.cfg.seed)
        obs = self._obs_dict_to_array(obs_dict)
        state = self.env.state()

        n_updates = max(1, self.cfg.total_timesteps // self.cfg.n_steps)
        eval_log: list[dict[str, float]] = []

        t_start = time.time()
        pbar = tqdm(
            total=n_updates,
            desc="SharedPPO",
            disable=not progress_bar,
            unit="upd",
            dynamic_ncols=True,
        )

        for update in range(1, n_updates + 1):
            obs, state, rollout_return, rollout_stats = self.collect_rollout(obs, state)

            # Bootstrap value for GAE from each agent's local observation.
            with torch.no_grad():
                last_value = self.critic(
                    torch.from_numpy(obs).to(self.device)
                ).cpu().numpy()
            self.buffer.compute_gae(last_value, self.cfg.gamma, self.cfg.gae_lambda)

            train_stats = self.update()

            did_eval = False
            eval_ret = None
            if (
                eval_env is not None
                and out_dir is not None
                and (update * self.cfg.n_steps) % self.cfg.eval_freq < self.cfg.n_steps
            ):
                eval_ret = self.evaluate(eval_env, deterministic=True)
                eval_log.append({
                    "step": int(self.global_step),
                    "eval_return": float(eval_ret),
                })
                if eval_ret > self.best_eval_return:
                    self.best_eval_return = eval_ret
                    self.save(out_dir / "mappo_best.pt")
                did_eval = True

            postfix: dict[str, str] = {
                "step": f"{self.global_step:,}",
                "ret": f"{rollout_return:.0f}",
                "H": f"{train_stats['entropy']:.2f}",
                "kl": f"{train_stats['approx_kl']:.3f}",
            }
            if self.best_eval_return > -np.inf:
                postfix["best_eval"] = f"{self.best_eval_return:.0f}"
            pbar.set_postfix(postfix)
            pbar.update(1)

            tick = max(1, n_updates // 10)
            if verbose and (update == 1 or update == n_updates or update % tick == 0):
                pct = 100.0 * update / n_updates
                elapsed = time.time() - t_start
                eta_s = elapsed * (n_updates - update) / max(1, update)
                best_str = (
                    f"  best_eval={self.best_eval_return:>11.1f}"
                    if self.best_eval_return > -np.inf else ""
                )
                tqdm.write(
                    f"[{pct:5.1f}%] upd {update:>4d}/{n_updates}  "
                    f"step {self.global_step:>7d}/{self.cfg.total_timesteps}  "
                    f"ret={rollout_return:>+9.1f}  "
                    f"H={train_stats['entropy']:.2f}  "
                    f"KL={train_stats['approx_kl']:.3f}  "
                    f"elapsed={int(elapsed//60):d}:{int(elapsed%60):02d}  "
                    f"ETA={int(eta_s//60):d}:{int(eta_s%60):02d}"
                    f"{best_str}"
                )

            if verbose and did_eval:
                tqdm.write(
                    f"    [eval @ step {self.global_step:>7d}] "
                    f"return={eval_ret:>12.2f}  best={self.best_eval_return:>12.2f}"
                )

        pbar.close()
        elapsed = time.time() - t_start
        if verbose:
            print(f"Training finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

        if out_dir is not None:
            self.save(out_dir / "mappo_final.pt")
            (out_dir / "eval_log.json").write_text(json.dumps(eval_log, indent=2))

    def save(self, path: Path) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_type": self.critic_type,
                "config": self.cfg.__dict__,
                "K": self.K,
                "obs_dim": self.obs_dim,
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
            },
            path,
        )
