# Analysis plan — centralized-PPO seed completion and shared-actor/local-critic ablation

Written 2026-09-14T03:59+02:00, **before any full-length Shared-PPO run and before
any retrained controller was evaluated on the test slice**. The only Shared-PPO
output at this time is an 8,192-step smoke test (validation split only).
Code hashes at writing time:
`src/trainers/shared_ppo.py` sha256 `762d9725…`, `scripts/train_shared_ppo.py` sha256 `e00f3f02…`.

## Controllers (primary, same-code replication)

All four trained by this study with the repository code as of commit `71a1b2f`
plus the two new, untracked Shared-PPO files. They use the current reward
(`configs/scenario_rl.yaml`, L_SW revision of 2026-06-06), twin `upf-digital-twin`
v0.4.0 @ `95de456`, CPU, one BLAS/OpenMP thread per process, 200,000 environment
steps, and validation-slice best-checkpoint selection. Seeds: **1, 7, 13, 23, 42, 64, 77, 99**.

| Name | Script (defaults unchanged) | Actor | Critic |
|---|---|---|---|
| MAPPO | `scripts/train_mappo.py` | one shared 14→64→64→2 | shared, global 140-dim state → 10 values |
| Shared-PPO | `scripts/train_shared_ppo.py` | one shared 14→64→64→2 | shared, own 14-dim obs → 1 value |
| IPPO | `scripts/train_ppo_ensemble.py` | 10 separate (SB3) | 10 separate (SB3) |
| Centralized PPO | `scripts/train_ppo_multi_site.py` | one joint 140-dim → MultiDiscrete (SB3) | joint |

**Shared-PPO** is the single name used for the ablation: one actor and one critic,
both parameter-shared across the ten sites, the critic seeing only local information.

Existing checkpoints (paper-era and v0.4 re-scored sets) are evaluated with the
same evaluator and reported separately for continuity. They are **not** mixed into
the primary comparison, because some were trained under the pre-2026-06-06 reward
and/or the pre-v0.4 twin.

## Evaluation protocol

`MultiAgentUPFEnv(split="test")`, `reset(seed=42)` (the reset is deterministic),
the full 1,009-step episode, and argmax (deterministic) actions from the
best-of-validation checkpoint. The test slice is touched exactly once per
checkpoint, after all training has finished. No retuning.

## Metrics (per seed)

- **Reward:** the undiscounted sum over clusters and steps of the env reward (higher is better).
- **Energy (Wh):** Σ `power_watts` × step_h (steady + amortised switching spike).
- **QoS-violation rate:** share of cluster-steps with `is_safe == False`, as in Table I. The q_score < τ rate is also reported.
- **USR share.**
- **Switches per cluster per day:** action changes ÷ K ÷ (1,009 × 15 min / 24 h).
- **Reward components:** Σ energy_term, qos_penalty, switch_penalty, cooldown_penalty.
- **Per-cluster:** the same quantities for each cluster.

## Aggregates

Mean, SD (ddof = 1), median, 95 % CI of the mean (Student t, df = 7) and a
percentile bootstrap 95 % CI (10,000 resamples) over the 8 seeds.

## Paired comparisons (seed-matched)

Per-seed differences A − B, paired bootstrap 95 % CI of the mean difference
(10,000 resamples, RNG seed 20260914), and an **exact two-sided sign-flip
permutation test** (all 2⁸ = 256 sign assignments). Secondary tests: exact Wilcoxon
signed-rank (two-sided), Cohen's d_z = mean/SD, and Hedges-corrected g_z. A
Holm adjustment is reported across the four primary reward contrasts. **No
one-sided tests.**

Pairing caveat. The seed fixes the torch/NumPy RNG. For MAPPO vs Shared-PPO this
yields bitwise-identical actor initialisation, so the pairing is genuine. For
comparisons against the SB3 controllers the pairing is nominal.

Primary contrasts (reward): Δ_crit = MAPPO − Shared-PPO; Δ_share = Shared-PPO − IPPO;
G = MAPPO − IPPO; MAPPO − Centralized PPO.

## Decision rule for the four candidate conclusions (fixed before results)

Let M = 1 % of |mean MAPPO test reward| (materiality threshold). "Significant"
means paired-bootstrap 95 % CI excludes 0 **and** exact permutation p < 0.05
(two-sided, unadjusted).

1. **S1 — The centralized critic materially improves over a shared local critic** if
   Δ_crit > 0, Δ_crit is significant, and Δ_crit ≥ M.
2. Otherwise, **S4 — MAPPO does not outperform the ablation** if the mean Δ_crit ≤ 0.
3. Otherwise, **S2 — Parameter sharing explains most of MAPPO's advantage** if G > 0 and
   G is significant, Δ_share is significant, and Δ_share ≥ 0.5 · G.
4. Otherwise, **S3 — The ablation is inconclusive.**

If S4 holds and the S2 conditions also hold, both are reported (S4 primary).
Secondary metrics (energy, QoS-violation, USR, switches) are descriptive and do
not enter the decision.

## Known confounds to report regardless of outcome

- Shared-PPO vs IPPO differs in parameter sharing **and** in implementation and
  hyper-parameters (custom trainer: lr 3e-4, entropy 0.05, minibatch 256, reward
  ×0.01, critic 128-128; SB3 IPPO: lr 1e-4, entropy 0.15, minibatch 64, 64-64
  networks). Δ_share is therefore not a pure parameter-sharing effect.
- Centralized PPO is trained on the load-weighted joint reward of
  `MultiSiteUPFEnv` and evaluated on the unweighted per-cluster sum, like the
  other controllers.
- The validation slice has 684 cluster-steps with load < 1 Mbps, where the SEC
  term explodes. Validation returns, and hence checkpoint selection, are
  dominated by those steps for every controller. The test slice has none.
- n = 8 seeds: the smallest attainable exact two-sided permutation p is 2/256 = 0.0078.
