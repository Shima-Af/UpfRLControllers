# Calibration robustness of the controller comparison

WCNC 2027 revision of *"A Measurement-Grounded Digital Twin for Evaluating Cooperative
MARL in Energy-Aware 5G UPF Orchestration"*. Run on 2026-09-14 between 04:35 and 09:30 CEST (CPU only; the alternative-world runs took 46 min).

**What this is.** A zero-shot re-evaluation of the saved controllers in 20 alternative
surrogate calibrations. Each calibration is refit from the existing profiling data with
one chronological profiling sweep (LOSO) or one group of load levels (LOLO) withheld.

**What this is not.** It is not live validation, independent physical validation, or
evidence of deployed-network performance.

**What stayed fixed.** No controller was retrained, retuned or modified. No measurement
was created, and the physical testbed was not used. The deployed surrogate directory, the
manuscript and all earlier results are unchanged.

Companion documents:

- `feasibility_report.md`: paths, inventory, runtime, risks.
- `reproduction_check.md`: the gate that ran before any alternative world.
- `analysis_plan.md`: pre-registered sets, statistics and verdict rule, fixed before the alternative worlds were run.

## Summary

1. **Reproduction gate.**
   - **Pinned twin v0.4.0:** the evaluator reproduces the recorded per-step and per-seed results exactly (843/843 checks).
   - **Twin v0.3.0, which produced Table I:** the MAPPO, IPPO, hysteresis and always-DPDK rows reproduce within printing precision. All 8 MAPPO and all 8 IPPO seeds trace to surviving checkpoints, per seed and per cluster.
   - **Centralized-PPO row:** it cannot be reproduced, because its checkpoints no longer exist. That row is therefore not tested; centralized PPO appears only through substitute checkpoint sets (†).
2. **Ranking.** With the manuscript's checkpoints (Experiment A, closed-policy), the reward ranking MAPPO > IPPO > hysteresis > always-DPDK > centralized PPO† is **identical to the pinned twin in 9 of 10 primary calibration worlds** (Kendall τ-b = 1). The exception is **LOLO-dep-f2**, where always-DPDK ranks first and MAPPO second (τ-b = 0.4).
3. **Key paired differences never change sign in any primary world.**
   - MAPPO − IPPO: +156 pinned; +31 to +477 across folds (median +198). 5–8 of 8 seeds favour MAPPO.
   - MAPPO − fixed hysteresis: +605 pinned; +297 to +1,333. 8 of 8 seeds in every world.
4. **Magnitudes depend materially on calibration.**
   - The MAPPO − IPPO margin shrinks to 0.20× its pinned value in LOSO-dep-f2 (+31, only 5/8 seeds) and grows to 3.1× in LOLO-dep-f2.
   - Relative to always-DPDK, MAPPO's reward advantage ranges from +27 % to **−12 %** (pinned +21 %).
   - Absolute rewards move far more than any seed effect: MAPPO −8,096 to −4,848 across folds, against a seed SD of 104.
5. **Energy and QoS conclusions hold in all 10 primary worlds.**
   - MAPPO has the lowest energy of all controllers.
   - Always-DPDK has the lowest QoS-violation rate, and MAPPO the second lowest.
   - The Pareto non-dominated set is {MAPPO, always-DPDK}.
   - MAPPO dominates IPPO, hysteresis and centralized PPO† on both axes.
   - Absolute QoS-violation rates rise by up to 2.7× in LOLO worlds (MAPPO 0.54 % → 1.47 %).
6. **Experiment B (fixed-action replay) gives the same picture.** The saved policies barely re-adapt to the altered observations (≤ 0.07 % of MAPPO's 50–600 Mbps decisions change). The median |A − B| is 0.1 % of the shift from pinned, so the calibration effect is almost entirely a re-scoring of unchanged decisions.
7. **Secondary (nested-selection) worlds break the ranking in 2 of 10: LOLO-nes-f2 and f3.**
   - In these two worlds the refit USR model predicts USR unsafe over the whole 1–500 Mbps grid (derived QoS limit 0 Mbps).
   - There always-DPDK wins by about 16,000 reward units, and IPPO overtakes MAPPO (−1,725 and −1,754).
   - The Pareto set changes too.
8. **Pre-registered verdict.**
   - **B — partially stable** for the manuscript checkpoints.
   - The same-code retrained set passes the stricter A rule. It is not less stable.

## 1. Calibration worlds

| World | Tier | Bundle (Layer-1 thr/cpu/loss/delay + Layer-2 power, DPDK and USR) | Energy break-even | USR QoS limit | Decision threshold |
|---|---|---|---|---|---|
| pinned | reference | deployed bundle (`data/external/profiling_twin/models`) | 91 | 149 | 81 |
| LOSO-dep-f0 … f4 | **primary** | deployed families + hyper-parameters, refit without sweep k | 91–93 | 149–153 | 81–83 |
| LOLO-dep-f0 … f4 | **primary** | deployed configuration, refit without load-level group k | 80–93 | **81**–170 | 70–83 |
| LOSO-nes-f0 … f4 | secondary | nested-selection fold models | 90–93 | 149–153 | 80–83 |
| LOLO-nes-f0 … f4 | secondary | nested-selection fold models | 61–81 | **0**–149 | 0–71 |

All thresholds are in Mbps, derived with the manuscript's rule on each bundle (`bundle_thresholds.csv`). 149 Mbps is a surrogate-derived quantity, not a measured physical limit.

**How the bundles were rebuilt.**
- The fold models were not saved by the validation study. They were rebuilt with its own code and recorded selections.
- They reproduce its saved held-out predictions to ≤ 1e-9.
- All 146 sanity checks pass (`bundle_checks.csv`): feature order and kbit/s units (UL := DL), DPDK→`dpdk` / USR→`usr_full`, env-code-path equality, grid predictions, finite outputs, power clip ≥ 0, and derived thresholds equal to `threshold_stability.csv`.
- Each bundle replaces the complete steady-state set. Activation durations and switching energies stay at the pinned values.

**Two LOLO primary worlds deserve attention.**
- **LOLO-dep-f2** withholds the 1, 10, 100 and 1,000 Mbps levels. Its USR model predicts USR unsafe from 81 Mbps: 91 % of region loads, against 69 % pinned. At the measured 100 Mbps level it predicts all 200 samples unsafe, while all 200 measured samples are safe (200 false-unsafe).
- **LOLO-dep-f3** withholds 2, 20, 200 and 2,000 Mbps and raises the QoS limit to 170 Mbps.

## 2. Experiment A — closed-policy zero-shot (primary controller set S1)

**Set S1.**
- MAPPO and IPPO: the 8 Table-I checkpoints each.
- Hysteresis: fixed at t_up 81 / t_down 31 Mbps, cooldown 1, as published.
- Always-DPDK.
- Centralized PPO†: the 4-seed substitute set from `multiseed_summary_v04twin.json`.

Values are seed means of total scalarized reward on the 1,009-step test slice.

| World | QoS limit / decision (Mbps) | MAPPO | IPPO | Cen. PPO† | Hyst. (fixed) | Always-DPDK | Reward ranking | Δ MAPPO−IPPO (seeds MAPPO>IPPO) | Δ MAPPO−hyst. (seeds) | τ-b |
|---|---|---|---|---|---|---|---|---|---|---|
| pinned | 149 / 81 | -5,658 | -5,814 | -14,459 | -6,262 | -7,201 | MAPPO > IPPO > Hyst. > DPDK > Cen. PPO† | +156 (7/8) | +605 (8/8) | 1.0 |
| LOSO-dep-f0 | 149 / 83 | -5,820 | -5,985 | -15,725 | -6,487 | -7,293 | MAPPO > IPPO > Hyst. > DPDK > Cen. PPO† | +165 (7/8) | +667 (8/8) | 1.0 |
| LOSO-dep-f1 | 149 / 83 | -5,116 | -5,249 | -14,401 | -5,711 | -6,603 | MAPPO > IPPO > Hyst. > DPDK > Cen. PPO† | +133 (7/8) | +595 (8/8) | 1.0 |
| LOSO-dep-f2 | 153 / 83 | -5,845 | -5,876 | -12,575 | -6,142 | -7,440 | MAPPO > IPPO > Hyst. > DPDK > Cen. PPO† | **+31 (5/8)** | +297 (8/8) | 1.0 |
| LOSO-dep-f3 | 150 / 81 | -5,476 | -5,720 | -11,637 | -6,113 | -7,238 | MAPPO > IPPO > Hyst. > DPDK > Cen. PPO† | +244 (7/8) | +637 (8/8) | 1.0 |
| LOSO-dep-f4 | 149 / 81 | -4,848 | -5,085 | -11,498 | -5,494 | -6,613 | MAPPO > IPPO > Hyst. > DPDK > Cen. PPO† | +238 (7/8) | +646 (8/8) | 1.0 |
| LOLO-dep-f0 | 149 / 80 | -6,838 | -7,065 | -16,630 | -7,598 | -8,334 | MAPPO > IPPO > Hyst. > DPDK > Cen. PPO† | +227 (8/8) | +760 (8/8) | 1.0 |
| LOLO-dep-f1 | 149 / 70 | -5,848 | -6,062 | -14,446 | -6,522 | -7,232 | MAPPO > IPPO > Hyst. > DPDK > Cen. PPO† | +213 (7/8) | +674 (8/8) | 1.0 |
| **LOLO-dep-f2** | **81 / 71** | -8,096 | -8,573 | -20,780 | -9,429 | **-7,224** | **DPDK > MAPPO > IPPO > Hyst. > Cen. PPO†** | +477 (6/8) | +1,333 (8/8) | **0.4** |
| LOLO-dep-f3 | 170 / 83 | -5,197 | -5,306 | -10,151 | -5,592 | -7,116 | MAPPO > IPPO > Hyst. > DPDK > Cen. PPO† | +109 (7/8) | +395 (8/8) | 1.0 |
| LOLO-dep-f4 | 149 / 83 | -6,657 | -6,839 | -16,932 | -7,375 | -8,152 | MAPPO > IPPO > Hyst. > DPDK > Cen. PPO† | +182 (8/8) | +718 (8/8) | 1.0 |

**Energy and QoS rankings, and Pareto (S1, Experiment A).** Cells show seed-mean Wh / QoS-violation %.

| World | MAPPO | IPPO | Cen. PPO† | Hyst. (fixed) | Always-DPDK | Energy ranking | QoS ranking | Non-dominated |
|---|---|---|---|---|---|---|---|---|
| pinned | 1,966 / 0.54 | 1,974 / 0.57 | 2,130 / 3.73 | 1,995 / 0.68 | 2,071 / 0.38 | MAPPO > IPPO > Hyst. > DPDK > Cen. PPO† | DPDK > MAPPO > IPPO > Hyst. > Cen. PPO† | MAPPO, DPDK |
| LOSO-dep-f0 | 1,967 / 0.64 | 1,974 / 0.67 | 2,136 / 3.92 | 1,996 / 0.79 | 2,071 / 0.47 | same | same | MAPPO, DPDK |
| LOSO-dep-f1 | 1,968 / 0.54 | 1,975 / 0.57 | 2,139 / 3.72 | 1,996 / 0.68 | 2,072 / 0.38 | same | same | MAPPO, DPDK |
| LOSO-dep-f2 | 1,969 / 0.64 | 1,976 / 0.67 | 2,132 / 3.81 | 1,997 / 0.78 | 2,073 / 0.47 | same | same | MAPPO, DPDK |
| LOSO-dep-f3 | 1,967 / 0.54 | 1,975 / 0.60 | 2,136 / 2.66 | 1,996 / 0.67 | 2,071 / 0.47 | same | same | MAPPO, DPDK |
| LOSO-dep-f4 | 1,965 / 0.54 | 1,973 / 0.57 | 2,134 / 3.42 | 1,996 / 0.65 | 2,071 / 0.38 | same | same | MAPPO, DPDK |
| LOLO-dep-f0 | 1,968 / 0.96 | 1,978 / 1.00 | 2,141 / 4.22 | 2,003 / 1.11 | 2,068 / 0.80 | same | same | MAPPO, DPDK |
| LOLO-dep-f1 | 1,979 / 0.63 | 1,993 / 0.66 | 2,169 / 3.84 | 2,019 / 0.77 | 2,070 / 0.47 | same | same | MAPPO, DPDK |
| LOLO-dep-f2 | 1,982 / 1.47 | 1,990 / 1.60 | 2,136 / 6.61 | 2,008 / 1.89 | 2,068 / 0.47 | same | same | MAPPO, DPDK |
| LOLO-dep-f3 | 1,963 / 0.63 | 1,970 / 0.64 | 2,073 / 3.30 | 1,987 / 0.72 | 2,069 / 0.47 | same | same | MAPPO, DPDK |
| LOLO-dep-f4 | 1,967 / 0.97 | 1,974 / 1.01 | 2,106 / 4.31 | 1,993 / 1.13 | 2,072 / 0.80 | same | same | MAPPO, DPDK |

**Manuscript statements tested (S1, primary worlds, Experiment A).**

| Manuscript statement (Table I / §Results) | Pinned (v0.4.0) | 10 primary folds | Holds? |
|---|---|---|---|
| MAPPO has the best reward of all controllers | yes | 9/10 (LOLO-dep-f2: always-DPDK best) | mostly |
| +9.7 % reward vs hysteresis (b = 50) | +9.7 % | median +10.3 % [+4.8, +14.1] | sign yes; magnitude varies |
| +21.5 % vs always-DPDK | +21.4 % | median +20.8 % [**−12.1**, +27.0] | **no in 1/10** |
| +33.8 % vs centralized PPO | row untraceable; substitute +60.9 % | median +59.2 % [+48.8, +64.5] | not testable for the published row |
| MAPPO beats IPPO | +156, 7/8 seeds | +31 … +477; 5–8/8 seeds; no reversal | sign yes; magnitude 0.2–3.1× |
| MAPPO lowest energy among trained policies | 1,966 Wh | lowest of all controllers in 10/10 | yes |
| MAPPO dominates IPPO, hysteresis, centralized PPO on energy–QoS; always-DPDK the only other non-dominated point | yes | 10/10 | yes |

Absolute QoS-violation rates and Wh are calibration dependent: MAPPO QoSv 0.54–1.47 %, Wh 1,963–1,982.

## 3. Experiment B — fixed-action counterfactual replay (S1)

Each checkpoint's pinned action trajectory is replayed unchanged in every world.

| World | MAPPO | IPPO | Cen. PPO† | Hyst. (fixed) | Always-DPDK | Reward ranking | Δ MAPPO−IPPO (seeds) | Δ MAPPO−hyst. (seeds) |
|---|---|---|---|---|---|---|---|---|
| pinned | -5,658 | -5,814 | -14,459 | -6,262 | -7,201 | MAPPO > IPPO > Hyst. > DPDK > Cen. PPO† | +156 (7/8) | +605 (8/8) |
| LOSO-dep-f0 | -5,816 | -5,985 | -15,664 | -6,487 | -7,293 | same | +169 (7/8) | +671 (8/8) |
| LOSO-dep-f1 | -5,129 | -5,249 | -14,424 | -5,711 | -6,603 | same | +119 (7/8) | +582 (8/8) |
| LOSO-dep-f2 | -5,841 | -5,876 | -12,541 | -6,142 | -7,440 | same | +34 (5/8) | +301 (8/8) |
| LOSO-dep-f3 | -5,474 | -5,720 | -11,586 | -6,113 | -7,238 | same | +246 (7/8) | +638 (8/8) |
| LOSO-dep-f4 | -4,863 | -5,085 | -11,509 | -5,494 | -6,613 | same | +223 (7/8) | +631 (8/8) |
| LOLO-dep-f0 | -6,841 | -7,065 | -16,450 | -7,598 | -8,334 | same | +223 (8/8) | +757 (8/8) |
| LOLO-dep-f1 | -5,846 | -6,062 | -14,406 | -6,522 | -7,232 | same | +216 (7/8) | +677 (8/8) |
| LOLO-dep-f2 | -8,055 | -8,678 | -20,968 | -9,429 | -7,224 | DPDK > MAPPO > IPPO > Hyst. > Cen. PPO† | +623 (6/8) | +1,375 (8/8) |
| LOLO-dep-f3 | -5,207 | -5,306 | -10,118 | -5,592 | -7,116 | same | +99 (6/8) | +385 (8/8) |
| LOLO-dep-f4 | -6,651 | -6,842 | -16,735 | -7,375 | -8,152 | same | +191 (7/8) | +725 (8/8) |

**Decomposition.**
- **Fixed hysteresis and the static policies.** They act only on the traffic forecast, so their Experiment A and B results are identical in every world (verified to 4e-12).
- **Learned policies.** Their closed-loop re-adaptation (A − B) is small:
  - MAPPO: −112 to +107 per checkpoint;
  - IPPO: −1 to +159;
  - centralized PPO†: −529 to +885.
- **Scale of adaptation.** The median |A − B| is 0.1 % of |B − pinned|. The saved policies change at most 0.07 % (MAPPO), 0.24 % (IPPO) and 1.1 % (centralized PPO†) of their 50–600 Mbps decisions relative to the pinned twin.
- **Conclusion.** The robustness of the ranking is therefore robustness of how the existing decisions are **scored**, not evidence that the policies adapt.
- **Verdict.** Experiment B's pre-registered verdict is also B, with identical counts.

## 4. Rank stability and uncertainty

**Rank stability (S1).** Counts are out of 10 worlds per tier.

| Criterion | Primary, Exp. A | Primary, Exp. B | Secondary (nested), Exp. A |
|---|---|---|---|
| MAPPO ranked first by reward | 9 | 9 | 8 |
| Reward ranking identical to pinned (τ-b = 1) | 9 (other: τ-b 0.4) | 9 | 8 (others: τ-b 0.2, 0.2) |
| MAPPO − IPPO ≤ 0 | 0 | 0 | 2 (LOLO-nes-f2 −1,725; f3 −1,754) |
| MAPPO − fixed hysteresis ≤ 0 | 0 | 0 | 0 |
| Energy ranking ≠ pinned | 0 | 0 | 2 (hysteresis lowest energy in LOLO-nes-f2/f3) |
| QoS ranking ≠ pinned | 0 | 0 | 2 |
| MAPPO Pareto-dominated | 0 | 0 | 1 (LOLO-nes-f2) |
| Non-dominated set ≠ {MAPPO, always-DPDK} | 0 | 0 | 2 |
| First-ranked controller by reward | MAPPO 9, always-DPDK 1 | MAPPO 9, always-DPDK 1 | MAPPO 8, always-DPDK 2 |
| Lowest energy | MAPPO 10 | MAPPO 10 | MAPPO 8, hysteresis 2 |
| Lowest QoS-violation rate | always-DPDK 10 | always-DPDK 10 | always-DPDK 10 |

**Complete five-fold results and best / median / worst fold per protocol (S1, primary).** Full detail for every statistic is in `fold_summary.csv`.

| Statistic | Exp. | Protocol | Pinned | Worst (fold) | Median | Best (fold) | f0 | f1 | f2 | f3 | f4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Δ MAPPO−IPPO | A | LOSO | +156 | +31 (f2) | +165 | +244 (f3) | +165 | +133 | +31 | +244 | +238 |
| Δ MAPPO−IPPO | A | LOLO | +156 | +109 (f3) | +213 | +477 (f2) | +227 | +213 | +477 | +109 | +182 |
| Δ MAPPO−IPPO | B | LOSO | +156 | +34 (f2) | +169 | +246 (f3) | +169 | +119 | +34 | +246 | +223 |
| Δ MAPPO−IPPO | B | LOLO | +156 | +99 (f3) | +216 | +623 (f2) | +223 | +216 | +623 | +99 | +191 |
| Δ MAPPO−hyst. | A | LOSO | +605 | +297 (f2) | +637 | +667 (f0) | +667 | +595 | +297 | +637 | +646 |
| Δ MAPPO−hyst. | A | LOLO | +605 | +395 (f3) | +718 | +1,333 (f2) | +760 | +674 | +1,333 | +395 | +718 |
| MAPPO reward | A | LOSO | -5,658 | -5,845 (f2) | -5,476 | -4,848 (f4) | -5,820 | -5,116 | -5,845 | -5,476 | -4,848 |
| MAPPO reward | A | LOLO | -5,658 | -8,096 (f2) | -6,657 | -5,197 (f3) | -6,838 | -5,848 | -8,096 | -5,197 | -6,657 |
| MAPPO QoSv % | A | LOSO | 0.54 | 0.64 (f2) | 0.54 | 0.54 (f1) | 0.64 | 0.54 | 0.64 | 0.54 | 0.54 |
| MAPPO QoSv % | A | LOLO | 0.54 | 1.47 (f2) | 0.96 | 0.63 (f3) | 0.96 | 0.63 | 1.47 | 0.63 | 0.97 |

**Seed uncertainty and calibration uncertainty, reported separately.** From `uncertainty_decomposition.csv`; S1, Experiment A, primary worlds. No interval is formed across folds: the five folds of a protocol share ~80 % of their profiling runs and are not independent calibration campaigns.

| Quantity | Seed uncertainty (pinned world) | Calibration uncertainty (range of seed means over 10 primary worlds) |
|---|---|---|
| MAPPO reward | SD 104, range 335 (8 seeds) | −8,096 … −4,848 (range 3,248) |
| IPPO reward | SD 117, range 367 | −8,573 … −5,085 (range 3,488) |
| Δ MAPPO − IPPO (paired) | SD 179, per-seed −159 … +452 | +31 … +477 |
| Δ MAPPO − fixed hysteresis | per-seed +385 … +720 | +297 … +1,333 |
| MAPPO energy | SD 3.7 Wh | 1,963 … 1,982 Wh |
| MAPPO QoS-violation rate | SD 0.05 pp | 0.54 … 1.47 % |

- **Absolute metrics** are dominated by calibration uncertainty, by roughly 30× for reward.
- **Paired MAPPO advantages** vary across calibrations by an amount comparable to their seed-to-seed spread, but never change sign.
- **Where the calibration effect sits.** It is concentrated in the QoS-penalty term rather than the energy term. On high-load cluster 9, which is served by DPDK, MAPPO's reward moves −1,113 … +589 per world with the energy term unchanged. In LOLO-dep-f2 clusters 0, 4 and 6 add −553, −293 and −502, mostly as QoS penalty.

## 5. Secondary analyses

**Nested-selection worlds (secondary tier).**
- Eight of ten behave like the primary worlds.
- **LOSO-nes-f1** lowers every controller's reward by about 3,800 (a QoS-violation rate about 6 pp higher for all controllers), but leaves the ranking intact.
- **LOLO-nes-f2 and LOLO-nes-f3** are degenerate: derived QoS limit 0 Mbps, with USR predicted unsafe on 100 % and 90 % of region loads. There:
  - always-DPDK wins by about 16,000 reward units;
  - IPPO beats MAPPO (3/8 seeds favour MAPPO);
  - fixed hysteresis becomes the lowest-energy controller;
  - the Pareto set changes.
- Their verdict under the same rule is C for S1.

**Other controller sets.** Verdict codes follow the pre-registered rule: A = stable, B = partially stable, C = unstable.

| Set | Exp. A, primary | Exp. B, primary | Exp. A, secondary |
|---|---|---|---|
| **S1 manuscript checkpoints** (MAPPO 8, IPPO 8, Cen. PPO† 4) | **B** (n₁ 9, r_I 0, r_H 0, p₁ 0, m 8) | B | C |
| S2 same-code retrained set (8/8/8) | A (n₁ 9, r_I 0, r_H 0, p₁ 0, m 9) | A | B |
| S3 v0.4 re-scored set (8/8/4) | B (m 8) | B | C |

- **S2 details.** S2's pinned ranking differs from Table I (MAPPO > hysteresis > always-DPDK > IPPO > centralized PPO): its IPPO seed 99 collapsed during training, so IPPO's mean is dominated by that seed. Across the 10 primary worlds MAPPO − IPPO stays +2,366 … +2,928, and MAPPO − hysteresis +326 … +1,545.
- **Shared-PPO (supplementary, S2).** It stays below MAPPO in 9/10 primary worlds (MAPPO − Shared-PPO +32 … +570; LOLO-dep-f3 −27).
- **Always-USR (supplementary; not in the manuscript evaluation).** −135,669 … −204,490 against −185,290 pinned.

**Recalibrated hysteresis (secondary; never mixed with the fixed controller).** `recalibrated_hysteresis.csv`.

- **Protocol.** Re-deriving t_up in each world and re-selecting the band on the test slice (the manuscript's protocol) recovers the pinned controller exactly in the pinned world (t_up 81, b = 50, −6,262).
- **Primary worlds.** MAPPO beats the recalibrated controller in 9/10, by a median of +641 (S1).
- **LOLO-dep-f2.** The recalibrated controller (t_up 71, b = 100, effectively always-DPDK) beats MAPPO by 872.
- **Degenerate nested worlds.** Recalibration collapses to always-DPDK (t_up 0), which beats MAPPO by about 18,000.

## 6. Decision-boundary region (50–600 Mbps)

70.8 % of cluster-steps have actual offered load in 50–600 Mbps (72.8 % by forecast load). `decision_boundary_summary.csv` gives per-world values.

**Pinned world, S1, Experiment A (USR shares are seed means).**

| Controller | USR share in region | USR steps predicted unsafe | USR share near 100 Mbps [89–141) | near 200 [141–283) | near 400 [283–490) | near 600 [490–693) | USR share above 149 Mbps |
|---|---|---|---|---|---|---|---|
| MAPPO | 1.8 % | 0.0 % | 0.04 % | 0 | 0 | 0 | 0.03 % |
| IPPO | 3.1 % | 0.3 % | 1.4 % | 0.07 % | 0 | 0.05 % | 0.03 % |
| Hysteresis (fixed) | 3.7 % | 6.9 % | 5.0 % | 1.0 % | 0.05 % | 0 | 0.28 % |
| Centralized PPO† | 9.6 % | 42 % | 16 % | 9.7 % | 2.9 % | 2.2 % | 4.3 % |

**Across the 10 primary worlds.**
- **Action changes.** Region decisions that differ from pinned are at most 0.07 % for MAPPO, 0.24 % for IPPO, 1.1 % for centralized PPO† and 0 % for hysteresis.
- **World-predicted unsafe share of USR region steps.** This is where calibration acts:
  - MAPPO: at most 0.9 %;
  - IPPO: at most 16 %;
  - hysteresis: at most 40 %;
  - centralized PPO†: at most 80 %.
  - The maxima occur in LOLO-dep-f2.
- **Region-load USR predictions per world.** 48 %–91 % of region loads are predicted USR-unsafe (pinned 69 %).
- **Interpretation.** The learned policies take most of their USR decisions below 50 Mbps: 84 % of MAPPO's USR steps, 73 % of IPPO's, against 65 % for fixed hysteresis and 25 % for centralized PPO†. Miscalibration near the USR QoS boundary therefore affects MAPPO and IPPO less than it affects the threshold controller.

**Measured labels are available only at the profiling design levels** (`boundary_measured_labels.csv`). Unsafe means loss > 5 packets per interval or delay > 200 µs, the validation study's definition. USR, pinned model, all 5 runs per level:

| Level (Mbps) | Samples | Measured unsafe | Pinned-predicted unsafe | False-safe | False-unsafe |
|---|---|---|---|---|---|
| 60 | 200 | 0.5 % | 0 % | 1 | 0 |
| 80 | 197 | 2.0 % | 0 % | 4 | 0 |
| 100 | 200 | 0 % | 0 % | 0 | 0 |
| 200 | 199 | 16.6 % | 11.6 % | 25 | 15 |
| 400 | 200 | 33.5 % | 100 % | 0 | 133 |
| 600 | 200 | 100 % | 100 % | 0 | 0 |

- **Size of the errors.** Across the primary worlds, the false-safe count at these levels is 5–30 samples and the false-unsafe count 148–349.
- **Held-out share.** Of these, 33 false-safe and 667 false-unsafe samples come from runs held out of the respective world's fit.
- **What the data show.** Measured USR behaviour between 100 and 400 Mbps is mixed: 17 % and 34 % unsafe, not a cliff. The surrogate-derived QoS limit (149 Mbps, 81–170 Mbps across primary worlds) sits in a region with no measurement between 100 and 200 Mbps. No label is assigned to traffic loads between measured levels.
- **Consequence for the controllers.** MAPPO's and IPPO's USR decisions near 200–600 Mbps are too rare (≤ 0.07 %) for measurement-level false-safe exposure to matter. The fixed hysteresis controller uses USR on 5 % of steps near 100 Mbps and 1 % near 200 Mbps.

## 7. Figures (`figures/`, PDF and PNG)

1. `fig1_calibration_robustness`: reward, energy and QoS-violation rate for each controller in every world; seed-mean markers and seed min–max bars. Secondary worlds are shaded; off-axis values are printed. `_S2`: same-code set.
2. `fig2_energy_qos_scatter`: energy–QoS positions of every controller in every world, with a zoom on the pinned and primary worlds. `_S2` variant.
3. `fig3_ranking_heatmap`: reward, energy and QoS ranks, controllers × 21 worlds. `_S2` variant.
4. `fig4_difference_from_pinned`: MAPPO, IPPO and fixed hysteresis, world minus pinned, Experiments A and B side by side.
5. `fig5_decision_trace`: 48 h on cluster 0, chosen by the traffic-only rule (median load 99.8 Mbps, closest to 81 Mbps; test steps 553–744, the window with the most 50–600 Mbps steps). Worlds: pinned plus the primary worlds with the lowest (LOLO-dep-f2, 81 Mbps) and highest (LOLO-dep-f3, 170 Mbps) QoS limit. Selection record: `figures/fig5_selection.json`.

**TeX table:** `calibration_summary.tex`.

## 8. Limitations

- **Surrogates, not measurements.** Every world is a refit of the same 220 profiling runs. Agreement across folds is robustness to *which* runs or load levels calibrate the surrogate, not validation against an independent system. The folds are correlated.
- **Scope of the refits.** Only the steady-state surrogates vary. Traffic, forecasts, activation durations and energies, the reward weights and the QoS budgets are fixed, and their uncertainty is not explored.
- **Centralized PPO.** The manuscript's centralized-PPO checkpoints are unavailable, so its row (−8,542 ± 1,744) is untested. The substitute (4 seeds, −14,459 pinned) ranks last in every world.
- **Twin version.** Table I was produced with twin v0.3.0, and the robustness worlds use the repository's v0.4.0. For the Table I checkpoints this changes reward by ≤ 4.2 units and no action.
- **Recalibrated hysteresis.** Its band is selected on the test slice, mirroring the manuscript.
- **No seed-level significance.** Seed-level significance tests are not repeated per world. Per-world seed counts favouring MAPPO are reported instead.
- **Manuscript text mismatch.** The manuscript's text describes a four-step hysteresis cooldown; the evaluated (and here reproduced) controller uses one step.

## 9. Final scientific verdict

**B — Partially stable.** MAPPO generally remains superior across alternative calibrations derived from the existing profiling data. The size of its advantage, some absolute energy and QoS values, and its superiority over the static always-DPDK policy depend materially on the calibration.

**Evidence supporting "MAPPO generally remains superior"** (manuscript checkpoints, closed-policy, 10 primary worlds):

- **Reward.** MAPPO ranks first in 9/10 worlds. The full reward ranking is unchanged in 9/10 (Kendall τ-b = 1).
- **Paired differences.** MAPPO − IPPO is positive in 10/10 (+31 … +477; 5–8 of 8 seeds). MAPPO − fixed hysteresis is positive in 10/10 (+297 … +1,333; 8 of 8 seeds every time).
- **Energy.** MAPPO has the lowest energy in 10/10.
- **Energy–QoS dominance.** MAPPO is never Pareto-dominated, it dominates IPPO, hysteresis and centralized PPO† in 10/10, and the non-dominated set is always {MAPPO, always-DPDK}.
- **Replay.** The fixed-action replay (Experiment B) gives identical counts.

**Evidence that the conclusion is only partially stable:**

- **Always-DPDK.** In LOLO-dep-f2 always-DPDK outperforms MAPPO by 872 reward units (+21 % pinned → −12 %). A calibration refit without the 1–1,000 Mbps decade levels therefore overturns the "MAPPO beats the static policy" claim.
- **Size of the IPPO margin.** It varies from 0.20× to 3.1× its pinned value (+156). At its smallest (LOSO-dep-f2, +31) only 5 of 8 seeds favour MAPPO.
- **Absolute metrics.** They move far more across calibrations than across seeds: MAPPO reward −8,096 … −4,848 against a seed SD of 104, and QoS-violation rate 0.54 … 1.47 %. The percentage improvements printed in the manuscript (9.7 %, 21.5 %) are calibration specific: +4.8 … +14.1 % and −12.1 … +27.0 % across primary worlds.
- **Nested-selection refits.** Two of them (LOLO-nes-f2/f3) make USR unusable. There IPPO overtakes MAPPO, always-DPDK wins and the Pareto set changes (secondary verdict C).
- **Why the ranking holds.** Its stability comes from re-scoring near-identical decisions (median |A − B| is 0.1 % of the calibration shift). It is not evidence that the saved policies adapt to a changed plant.

The same-code retrained set (S2) meets the stricter stable-ranking rule (A) in the primary worlds, so it is not less stable than the manuscript checkpoints.

**Suggested wording for the revision.**

> Re-evaluating the saved controllers, without retraining, in ten alternative surrogate calibrations derived from the existing profiling data (five leave-one-sweep-out and five leave-load-level-out refits), MAPPO kept the best reward in nine and a positive paired advantage over IPPO and the hysteresis baseline in all ten, and remained on the energy–QoS Pareto front with always-DPDK in every calibration. The size of these advantages and the absolute energy and QoS values depended on the calibration, and in one refit always-DPDK obtained a higher reward than every learned controller.

Do not call this validation of real-system or deployed-network performance.
