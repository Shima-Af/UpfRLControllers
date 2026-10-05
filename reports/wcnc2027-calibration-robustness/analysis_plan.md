# Pre-registered analysis plan — calibration-robustness evaluation

Written 2026-09-14, finished before 07:59 CEST (launch of the alternative-world runs, logs/eval_main_worlds.log), **after** the feasibility audit, bundle
reconstruction and the pinned reproduction check, and **before** any controller
was evaluated in an alternative calibration world. Nothing below may be changed
after the alternative-world results exist; any later analysis is labelled
post hoc.

## Worlds

| Tier | Worlds | Bundle |
|---|---|---|
| reference | `pinned` | deployed surrogate bundle `data/external/profiling_twin/models` (DVC md5 a92ad2ca) under twin v0.4.0 |
| **primary** | `LOSO-dep-f0..4`, `LOLO-dep-f0..4` (10) | deployed model families + hyper-parameters, refit on the fold's training runs |
| secondary | `LOSO-nes-f0..4`, `LOLO-nes-f0..4` (10) | nested-selection fold models |

Every world uses twin v0.4.0, the pinned `switching_costs.yaml` (activation durations
and spike energies fixed), the same test slice, reset(seed=42), deterministic actions,
observation order, action encoding and reward weights. No controller has
normalisation statistics (none exist on disk), so none are recomputed.

## Controller sets

- **S1 — primary ("manuscript checkpoints").**
  - MAPPO: cohort T, the 8 checkpoints that reproduce Table I per seed and per cluster under twin v0.3.0.
  - IPPO: cohort T (`ippo_nr_seed*`), 8 seeds.
  - Fixed hysteresis: t_up 81, t_down 31 Mbps, cooldown 1, exactly as evaluated for Table I.
  - Always-DPDK.
  - Centralized PPO: the manuscript's checkpoints are **unavailable** (not traceable). The substitute is the 4-seed set of `multiseed_summary_v04twin.json` (`ppo_multi_site_seed{7,13,42,99}_20260517T004954`), labelled as a substitute. Seed counts are 8 / 8 / 4.
- **S2 — secondary ("same-code revision set", cohort R).**
  - MAPPO, IPPO and centralized PPO from `experiments/wcnc2027_ablation`, 8 seeds each, plus fixed hysteresis and always-DPDK.
  - Shared-PPO (8 seeds) is reported but not ranked in the verdict.
- **S3 — tertiary (cohort P, v0.4 re-scored set).** MAPPO 8, IPPO 8 and centralized PPO 4 seeds, plus the two baselines.
- **Other checkpoints.** Every other surviving checkpoint (cohort L) is evaluated and saved in the CSVs, but not ranked.
- **Always-USR** was not part of the manuscript evaluation. It is reported as a supplementary reference and excluded from every ranking.

## Statistics (no pooling of time steps or profiling runs; folds are not independent campaigns)

- **Controller score in a world.** The mean over its seeds of each metric. The two deterministic baselines have a single value.
- **Rankings.** Computed separately for scalarized reward (higher is better), energy Wh (lower is better) and QoS-violation rate (lower is better). Median-over-seeds rankings are a sensitivity check.
- **Pareto.** On (mean Wh, mean QoS-violation rate), minimising both; A dominates B if A ≤ B on both and A < B on at least one.
  - (P1) MAPPO is non-dominated.
  - (P2) MAPPO dominates IPPO, hysteresis and centralized PPO. These are the manuscript's statements; whether each holds in the pinned world is recorded first.
- **Key differences per world.**
  - Δ_IPPO: the mean over seed-matched pairs of MAPPO − IPPO reward, with the number of seeds favouring MAPPO.
  - Δ_hyst: the mean over MAPPO seeds of MAPPO − fixed hysteresis, with the number of seeds favouring MAPPO.
  - Δ_DPDK and Δ_central: difference of means; unpaired for S1 and S3 because the centralized seeds differ.
  - For each: the sign relative to the pinned world, and the ratio ρ = Δ(world) / Δ(pinned).
- **Rank agreement.** Kendall τ-b and Spearman ρ between each world's reward ranking of the five ranked controllers and the pinned ranking; how often each controller ranks first.
- **Two sources of uncertainty, reported separately.**
  - Seed uncertainty: within each world, the seed SD and min–max of each controller and of the paired differences.
  - Calibration uncertainty: across the 10 primary worlds, the median, minimum and maximum of the seed-mean statistic, with all 10 fold values shown.
  - No confidence interval across folds is computed.

## Experiments

- **Experiment A (closed-policy zero-shot)** is the basis of the verdict.
- **Experiment B (fixed-action replay of the pinned action trajectory)** is reported separately. A − B isolates the effect of policy re-adaptation to the world's observations; B − pinned isolates re-scoring of an unchanged trace.

## Verdict rule (applied to S1, Experiment A, 10 primary worlds)

Let:

- n₁ = the number of worlds where MAPPO ranks first by mean reward;
- r_I = the number of worlds with Δ_IPPO ≤ 0;
- r_H = the number of worlds with Δ_hyst ≤ 0;
- p₁ = the number of worlds where (P1) fails;
- m = the number of worlds where all of the following hold:
  - 0.5 ≤ ρ ≤ 2 for both Δ_IPPO and Δ_hyst;
  - the pinned-world sign of MAPPO − IPPO and of MAPPO − hysteresis is unchanged for **both** energy Wh and QoS-violation rate;
  - every (P2) relation that holds in the pinned world still holds.

Verdicts:

- **A — Stable ranking:** n₁ ≥ 9, r_I = 0, r_H = 0, p₁ = 0 and m ≥ 9.
- **B — Partially stable:** not A, but n₁ ≥ 8, r_I ≤ 1, r_H ≤ 1 and p₁ ≤ 2.
- **C — Unstable ranking:** otherwise.

The same rule is also computed, and reported as secondary, for S2 and S3 and for the 10 secondary (nested) worlds. If S2's verdict is less stable than S1's, the report must say so next to the S1 verdict.

## Decision-boundary analysis (50–600 Mbps)

- **Region.** A decision falls in the region when the actual offered load of that step lies in [50, 600] Mbps. Counts by forecast load are reported alongside.
- **Per controller set, controller, world and experiment**, report:
  - the number and percentage of region decisions;
  - the share of region decisions whose requested action differs from the same checkpoint's pinned action (Experiment A);
  - the USR share, and among USR steps in the region, the world-predicted safe and unsafe shares.
- **Measured labels.** Profiling measurements exist only at the design levels 60, 80, 100, 200, 400 and 600 Mbps. At each of those levels, each world's USR `is_safe` prediction on the measured samples is compared with the measured label, giving false-safe and false-unsafe counts, flagged by whether the level or sweep was held out of that world's fit. No label is assigned to traffic loads between measured levels.
- **Usage intervals.** USR usage near 100, 200, 400 and 600 Mbps uses intervals bounded by the geometric midpoints between neighbouring design levels:
  - 100 Mbps: [89.4, 141.4)
  - 200 Mbps: [141.4, 282.8)
  - 400 Mbps: [282.8, 489.9)
  - 600 Mbps: [489.9, 692.8)
- **Threshold sensitivity.** Each world's derived break-even, QoS limit and decision threshold (`bundle_thresholds.csv`), plus the USR share at loads above the world's QoS limit. 149 Mbps is a surrogate-derived quantity, not a measured physical limit.

## Recalibrated hysteresis (secondary, never mixed with fixed hysteresis)

- In each world, t_up is the world's derived decision threshold (min(break-even, QoS limit) − 10 Mbps).
- The band b is swept over {5, 10, 20, 50, 100} Mbps with cooldown 1. The best test-slice reward is reported, which is the same selection protocol as the manuscript's band sweep (selection on the test slice). The b = 50 point is also reported.
- If t_up ≤ 0, the controller is always-DPDK.

## Decision-trace figure (selected by traffic only)

- **Cluster.** The cluster whose median actual test-slice load is closest to the pinned decision threshold (81 Mbps).
- **Window.** The 48 h window (192 steps) that maximises the number of steps with actual load in [50, 600] Mbps; ties go to the earliest start.
- **Controllers.** S1 MAPPO and IPPO at the smallest seed (seed 1), plus fixed hysteresis.
- **Worlds.** Pinned, plus the primary worlds with the lowest and highest derived QoS limit (ties broken by decision threshold, then world order).
