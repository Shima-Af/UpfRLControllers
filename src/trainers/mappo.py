"""Phase 6 — MAPPO with centralised training, decentralised execution.

From-scratch PyTorch implementation. Architecture:

  Actor (shared parameters across agents):
      input  : per-agent observation (14-dim)
      hidden : 64 -> 64 (ReLU)
      output : 2 logits (DPDK / USR)
      The same network is used for every agent — K=10 agents share
      one set of parameters. Each agent's own observation drives its
      own decision (decentralised execution).

  Critic (centralised, shared parameters with K heads):
      input  : global state (K * 14 = 140-dim concatenation of
               per-agent observations)
      hidden : 128 -> 128 (ReLU)
      output : K=10 value estimates (one per agent)
      Sees all agents' observations during training (centralised
      training); each head produces V(s_t) for one agent.

GAE per agent. PPO-clip surrogate loss. Advantage normalisation
across the (T, K) tensor. Agents share a single optimiser update —
gradients flow through the actor's per-agent passes summed up.

Training-loop variables follow the MAPPO paper notation (Yu et al.
2022). The PettingZoo Parallel env from
``src/envs/multi_agent_upf_env.py`` is the only thing this module
talks to — same train/val/test split semantics as Phases 2–4.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from tqdm.auto import tqdm

from src.envs.multi_agent_upf_env import MultiAgentUPFEnv

# ----------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------


def _orthogonal_init(layer: nn.Linear, gain: float = np.sqrt(2)) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class Actor(nn.Module):
    """Per-agent actor with shared parameters across all agents."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_actions)
        for layer in self.body:
            if isinstance(layer, nn.Linear):
                _orthogonal_init(layer)
        _orthogonal_init(self.head, gain=0.01)  # near-uniform init

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(obs))

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        if deterministic:
            action = logits.argmax(dim=-1)
            log_prob = torch.zeros_like(action, dtype=torch.float32)
            return action, log_prob
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(
        self, obs: torch.Tensor, action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return log-prob of given action and entropy of the dist."""
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy()


class CentralisedCritic(nn.Module):
    """Shared critic that maps global state -> K value estimates."""

    def __init__(self, state_dim: int, K: int, hidden: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.heads = nn.Linear(hidden, K)
        for layer in self.body:
            if isinstance(layer, nn.Linear):
                _orthogonal_init(layer)
        _orthogonal_init(self.heads, gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.heads(self.body(state))


# ----------------------------------------------------------------------
# Rollout buffer
# ----------------------------------------------------------------------


@dataclass
class Rollout:
    """Buffers a fixed-length on-policy rollout for K agents."""

    n_steps: int
    K: int
    obs_dim: int
    state_dim: int

    obs: np.ndarray = field(init=False)        # (T, K, obs_dim)
    state: np.ndarray = field(init=False)      # (T, state_dim)
    actions: np.ndarray = field(init=False)    # (T, K)  int
    log_probs: np.ndarray = field(init=False)  # (T, K)
    rewards: np.ndarray = field(init=False)    # (T, K)
    values: np.ndarray = field(init=False)     # (T, K)
    dones: np.ndarray = field(init=False)      # (T,)    episode ended after this step
    advantages: np.ndarray = field(init=False)  # (T, K)
    returns: np.ndarray = field(init=False)    # (T, K)

    def __post_init__(self) -> None:
        T, K = self.n_steps, self.K
        self.obs = np.zeros((T, K, self.obs_dim), dtype=np.float32)
        self.state = np.zeros((T, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((T, K), dtype=np.int64)
        self.log_probs = np.zeros((T, K), dtype=np.float32)
        self.rewards = np.zeros((T, K), dtype=np.float32)
        self.values = np.zeros((T, K), dtype=np.float32)
        self.dones = np.zeros(T, dtype=np.float32)
        self.advantages = np.zeros((T, K), dtype=np.float32)
        self.returns = np.zeros((T, K), dtype=np.float32)
        self._t = 0

    def add(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        value: np.ndarray,
        done: float,
    ) -> None:
        t = self._t
        self.obs[t] = obs
        self.state[t] = state
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.rewards[t] = reward
        self.values[t] = value
        self.dones[t] = done
        self._t += 1

    @property
    def full(self) -> bool:
        return self._t >= self.n_steps

    def reset_cursor(self) -> None:
        self._t = 0

    def compute_gae(
        self, last_value: np.ndarray, gamma: float, gae_lambda: float
    ) -> None:
        """Compute GAE per agent using bootstrapped final value."""
        T, K = self.n_steps, self.K
        adv = np.zeros((T, K), dtype=np.float32)
        gae = np.zeros(K, dtype=np.float32)
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
                next_nonterminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_nonterminal = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + gamma * next_value * next_nonterminal
                - self.values[t]
            )
            gae = delta + gamma * gae_lambda * next_nonterminal * gae
            adv[t] = gae
        self.advantages = adv
        self.returns = adv + self.values


# ----------------------------------------------------------------------
# MAPPO trainer
# ----------------------------------------------------------------------


@dataclass
class MAPPOConfig:
    total_timesteps: int = 200_000
    n_steps: int = 1024
    n_epochs: int = 10
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.9
    clip_range: float = 0.2
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    actor_hidden: int = 64
    critic_hidden: int = 128
    device: str = "cpu"
    seed: int = 42
    eval_freq: int = 4096
    eval_split: str = "val"
    train_split: str = "train"
    # Reward scaling. Per-step per-agent rewards in this env can spike to
    # ~ -50 at near-zero load (SEC = power / max(load, eps) explodes); the
    # cumulative return then sits in the -10k range. The critic MSE on
    # those returns dominates the policy loss by ~6 orders of magnitude
    # if rewards are unscaled. Scaling by 1/100 brings value/policy loss
    # back to the same order of magnitude. The optimal policy is invariant
    # under uniform reward scaling.
    reward_scale: float = 0.01


class MAPPO:
    """Centralised-training, decentralised-execution PPO over K agents.

    The actor is shared across all agents (one network handles every
    agent's per-step decision). The critic is shared and outputs K
    value estimates from the global state.
    """

    def __init__(self, env: MultiAgentUPFEnv, cfg: MAPPOConfig) -> None:
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
        self.critic = CentralisedCritic(self.state_dim, self.K, cfg.critic_hidden).to(
            self.device
        )
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
    # Core algorithm
    # ------------------------------------------------------------------

    def _obs_dict_to_array(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.stack(
            [obs_dict[a] for a in self.env.possible_agents], axis=0
        ).astype(np.float32)

    def collect_rollout(self, obs: np.ndarray, state: np.ndarray) -> tuple[
        np.ndarray, np.ndarray, float, dict[str, float]
    ]:
        """Step the env n_steps times, filling the buffer in place.

        Returns the final obs, state, episode return so far, and stats.
        """
        self.buffer.reset_cursor()
        ep_return = 0.0
        n_done = 0

        for _ in range(self.cfg.n_steps):
            obs_t = torch.from_numpy(obs).to(self.device)            # (K, obs_dim)
            state_t = torch.from_numpy(state).to(self.device).unsqueeze(0)  # (1, S)

            with torch.no_grad():
                action, log_prob = self.actor.act(obs_t, deterministic=False)
                value = self.critic(state_t).squeeze(0)              # (K,)

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

            # Track in unscaled units so log/eval numbers match the report.
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
        """Run n_epochs of PPO updates over the filled buffer."""
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
        # Normalise advantages across (T, K).
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False).clamp_min(1e-8)
        advantages = (advantages - adv_mean) / adv_std
        returns = torch.from_numpy(self.buffer.returns).to(self.device)  # (T, K)
        states = torch.from_numpy(self.buffer.state).to(self.device)     # (T, state_dim)

        n_samples = T * K
        idx = np.arange(n_samples)
        # Indices into (T, K) for matching critic outputs to actor batches.
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

                # Critic minibatch: take the unique time steps in `mb`,
                # forward all K values for each, and pull the per-agent
                # entries we need.
                mb_t = t_idx[mb]
                mb_k = k_idx[mb]
                # Forward critic for each unique state in this minibatch
                # (same time step appears K times, dedupe).
                unique_t, inv = np.unique(mb_t, return_inverse=True)
                states_unique = states[unique_t]                  # (U, state_dim)
                vals = self.critic(states_unique)                 # (U, K)
                # Map back: for each sample in mb, value = vals[inv[i], mb_k[i]]
                vals_per_sample = vals[inv, mb_k]                 # (B,)
                target_per_sample = returns[mb_t, mb_k]
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

    # ------------------------------------------------------------------
    # Top-level training loop
    # ------------------------------------------------------------------

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
            desc="MAPPO",
            disable=not progress_bar,
            unit="upd",
            dynamic_ncols=True,
        )

        for update in range(1, n_updates + 1):
            obs, state, rollout_return, rollout_stats = self.collect_rollout(obs, state)

            # Bootstrap value for GAE.
            with torch.no_grad():
                last_value = self.critic(
                    torch.from_numpy(state).to(self.device).unsqueeze(0)
                ).squeeze(0).cpu().numpy()
            self.buffer.compute_gae(last_value, self.cfg.gamma, self.cfg.gae_lambda)

            train_stats = self.update()

            # Evaluation + best-checkpoint selection.
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

            # Progress bar update: postfix shows rollout return, best eval,
            # entropy, and KL — the at-a-glance training health signals.
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

    @torch.no_grad()
    def evaluate(
        self, env: MultiAgentUPFEnv, deterministic: bool = True
    ) -> float:
        """Run one full episode under deterministic policy, return total reward."""
        obs_dict, _ = env.reset(seed=self.cfg.seed + 1)
        total = 0.0
        while env.agents:
            obs = self._obs_dict_to_array(obs_dict)
            obs_t = torch.from_numpy(obs).to(self.device)
            action, _ = self.actor.act(obs_t, deterministic=deterministic)
            action_dict = {
                a: int(action[k].item())
                for k, a in enumerate(env.possible_agents)
            }
            obs_dict, reward_dict, _term, _trunc, _info = env.step(action_dict)
            total += float(sum(reward_dict.values()))
        return total

    def save(self, path: Path) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "config": self.cfg.__dict__,
                "K": self.K,
                "obs_dim": self.obs_dim,
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
            },
            path,
        )

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])


# ----------------------------------------------------------------------
# Inference helper
# ----------------------------------------------------------------------


def mappo_policy(actor: Actor, possible_agents: list[str], device: torch.device):
    """Returns a callable that maps an obs-dict -> action-dict."""
    @torch.no_grad()
    def _fn(obs_dict: dict[str, np.ndarray]) -> dict[str, int]:
        obs = np.stack([obs_dict[a] for a in possible_agents]).astype(np.float32)
        obs_t = torch.from_numpy(obs).to(device)
        action, _ = actor.act(obs_t, deterministic=True)
        return {a: int(action[k].item()) for k, a in enumerate(possible_agents)}

    return _fn


def rollout_mappo_episode(
    actor: Actor,
    *,
    horizon_idx: int = 0,
    seed: int = 0,
    split: str = "test",
    device: str = "cpu",
) -> dict[str, Any]:
    """One full deterministic rollout, returning per-cluster KPIs."""
    env = MultiAgentUPFEnv(horizon_idx=horizon_idx, split=split)
    obs_dict, _info = env.reset(seed=seed)
    K = env.K
    step_h = env.step_h
    tau = env.tau
    dev = torch.device(device)
    actor = actor.to(dev)

    per_c_reward = np.zeros(K, dtype=np.float64)
    per_c_energy = np.zeros(K, dtype=np.float64)
    per_c_unsafe = np.zeros(K, dtype=np.int64)
    per_c_qos = np.zeros(K, dtype=np.int64)
    per_c_dpdk = np.zeros(K, dtype=np.int64)
    per_c_usr = np.zeros(K, dtype=np.int64)
    per_c_switch = np.zeros(K, dtype=np.int64)
    last_act: list[int | None] = [None] * K
    steps = 0

    while env.agents:
        obs = np.stack(
            [obs_dict[a] for a in env.possible_agents]
        ).astype(np.float32)
        obs_t = torch.from_numpy(obs).to(dev)
        action, _ = actor.act(obs_t, deterministic=True)
        action_dict = {
            a: int(action[k].item())
            for k, a in enumerate(env.possible_agents)
        }
        obs_dict, reward_dict, _t, _u, info_dict = env.step(action_dict)
        for k, a in enumerate(env.possible_agents):
            per_c_reward[k] += reward_dict[a]
            ck = info_dict[a]
            per_c_energy[k] += float(ck["power_watts"]) * step_h
            if not ck["is_safe"]:
                per_c_unsafe[k] += 1
            if ck["q_score"] < tau:
                per_c_qos[k] += 1
            if ck["selected_upf"] == "DPDK":
                per_c_dpdk[k] += 1
            else:
                per_c_usr[k] += 1
            ak = int(action[k].item())
            if last_act[k] is not None and last_act[k] != ak:
                per_c_switch[k] += 1
            last_act[k] = ak
        steps += 1

    n = max(1, steps)
    return {
        "K": K,
        "steps": steps,
        "split": split,
        "total_reward_unweighted": float(per_c_reward.sum()),
        "per_cluster_total_reward": per_c_reward.tolist(),
        "per_cluster_energy_wh": per_c_energy.tolist(),
        "per_cluster_unsafe_rate": (per_c_unsafe / n).tolist(),
        "per_cluster_qos_violation_rate": (per_c_qos / n).tolist(),
        "per_cluster_dpdk_rate": (per_c_dpdk / n).tolist(),
        "per_cluster_usr_rate": (per_c_usr / n).tolist(),
        "per_cluster_n_switches": per_c_switch.tolist(),
        "total_energy_wh": float(per_c_energy.sum()),
        "agg_unsafe_rate": float(per_c_unsafe.sum() / (K * n)),
        "agg_usr_rate": float(per_c_usr.sum() / (K * n)),
        "agg_n_switches": int(per_c_switch.sum()),
    }
