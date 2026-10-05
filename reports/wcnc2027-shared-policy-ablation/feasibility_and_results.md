# Centralized-PPO seed completion and shared-actor/local-critic ablation

WCNC 2027 revision of *"A Measurement-Grounded Digital Twin for Evaluating
Cooperative MARL in Energy-Aware 5G UPF Orchestration"*. The work was run on
2026-09-14 between 03:45 and 04:30 CEST. **All computation finished more than 55 h
before the 16 Sep 12:00 cutoff.** Decision rules were fixed in `analysis_plan.md`
(03:59) before any full-length ablation result existed.

## Summary

- **Seeds verified.** MAPPO and IPPO used **[1, 7, 13, 23, 42, 64, 77, 99]**. The
  manuscript's four-seed centralized-PPO row (−8,542 ± 1,744) cannot be traced to any
  surviving set of checkpoints. The existing MAPPO, IPPO and centralized checkpoints
  come from different reward and twin versions (§1.2), and seeds 7, 13 and 77 of the
  manuscript's MAPPO values have no surviving checkpoint.
- **What was run.** Because a fair ablation requires one code version, *all four*
  controllers were retrained with identical code on the eight verified seeds, in
  16.5 min of wall time (32 CPU cores, no GPU). Existing checkpoints were left
  untouched and are reported separately.
- **Centralized PPO, 8 seeds.** −15,025 ± 6,931 (median −13,365). Every seed is
  worse than always-DPDK. MAPPO beats it on 8/8 seeds by +9,401 [+5,072, +14,158],
  p = 0.0078.
- **Ablation.** MAPPO −5,624 ± 22; **Shared-PPO** (shared actor + shared local critic)
  −5,707 ± 77; IPPO −8,236 ± 6,912 (median −5,794; seed 99 collapsed on one cluster).
- **Pre-registered conclusion: Statement 1.** The centralized critic materially
  improves over a shared local critic: +82.9 [+31, +151], exact two-sided p = 0.023,
  7/8 seeds. The effect is modest (≈1.5 % of cost, about half of MAPPO's typical-seed
  advantage over IPPO), acts through fewer QoS penalties, and is marginal on
  materiality (Holm p = 0.047; one leave-one-out subset falls below the threshold).
  Parameter sharing does **not** explain "most" of MAPPO's advantage in typical
  seeds; it and implementation differences account for roughly the other half.

## Phase 1 — Feasibility audit (read-only, 2026-09-14 03:45–04:05 CEST)

### 1.1 Seeds

| Question | Finding | Evidence |
|---|---|---|
| MAPPO / IPPO seeds | **[1, 7, 13, 23, 42, 64, 77, 99]**, verified | `reports/phase-7/multiseed_summary_n8.json` (the paper's Table I values: MAPPO −5,655 ± 98, IPPO −5,812 ± 110 reproduce from it exactly), `paired_bootstrap_mappo_vs_ippo.json` (`common_seeds`), `multiseed_summary_v04twin.json`, `hyst_sweep/band_*.json` |
| Centralized-PPO seeds in the paper (−8,542 ± 1,744, n = 4) | **Not traceable.** No set of four distinct-seed checkpoints on disk reproduces it. The closest reward matches reuse two different seed-7 runs and still miss the switching rate (0.80–0.93 vs 0.44 Sw/d). | brute force over all 20 surviving centralized checkpoints (`audit_existing_checkpoints.csv`) |
| Other four-seed centralized sets on disk | May-16 runs, seeds {7, 13, 42, 99}: −12,269 (= Phase-7 README / MODULE_SHEET); May-17 runs, {7, 13, 42, 99}: −14,459 (= `multiseed_summary_v04twin.json`); June `central_nr`, {1, 7, 13, 42}: −13,613 | test re-evaluation with current code |
| Remaining centralized seeds already run? | **Partly.** `central_nr_seed{23,64,77}` were trained 2026-08-27 with the current reward and twin v0.4; `ppo_multi_site_seed{1,23,64,77}_bridge` are byte-identical copies of `central_nr_seed{1,23,64,77}`. The untracked `multiseed_summary_n8central.json` combines these with the May-17 {7, 13, 42, 99} runs, **mixing two reward versions**. **Seed 99 was never trained under the current reward.** | md5 of checkpoints, directory dates, JSON contents |

### 1.2 Provenance of the existing checkpoints

- **Reward change.** Commit `781c90c` (2026-06-06 09:47) moved the switching cost
  from an energy term folded into SEC to an explicit L_SW = λ_sw · E_sw. Every
  `mappo_seed*`, `ppo_multi_site_seed*_2026051*` and
  `ppo_single_site_ensemble_seed*` run predates it. The `*_nr` runs (June 6) and
  the August centralized runs postdate it.
- **Twin change.** Twin v0.4.0 (2026-07-23) changed the switching-energy spike.
  The paper's Table I was computed before it; the `v04twin` summary after it.
- **What Table I is built from.**
  - IPPO = `ippo_nr_seed*` (all 8 match within 1.5 units, the twin-fix shift).
  - MAPPO: seeds 23, 64 and 99 = `mappo_nr_*`; seeds 1 and 42 match May runs within 5 units; **no surviving checkpoint matches seeds 7, 13 and 77**.
  - Centralized PPO: not traceable (above).
- **Consequence.** The existing MAPPO, IPPO and centralized sets were not trained
  under one common code version. A shared-actor ablation trained today would be
  confounded with reward and twin version if compared against them. **All
  controllers are therefore retrained with identical code.** The legacy
  checkpoints are evaluated separately and left untouched.

### 1.3 Protocol checks

| Check | Result |
|---|---|
| Same chronological train/val/test slices | **Yes.** Every env reads `targets_/predictions_{train,val,test}.npy` (5,073 / 1,009 / 1,009 steps, 15 min), horizon index 0 |
| Model selection on validation only | **Yes.** `EvalCallback(eval_split="val")` for SB3; MAPPO's trainer refuses `eval_split="test"`; best-of-val checkpoint every 4,096 steps, 1 deterministic episode. **Caveat:** the val slice has 684 cluster-steps with load < 1 Mbps, where α·SEC explodes (always-DPDK val return −5.0·10⁷), so selection is dominated by those steps. The test slice has none |
| Deterministic test evaluation reproducible from checkpoints | **Yes.** All 52 existing checkpoints re-evaluated; every per-seed value in `multiseed_summary_v04twin.json` reproduced to ≤ 0.1 reward units; always-DPDK −7,201.08 |
| Training reproducible bit-for-bit | **Not across execution settings.** Retraining centralized seed 23 with current code gives test −18,575 vs −27,425 for the 2026-08-27 run of the same seed (env identical: the baseline rows of `summary.txt` match exactly). Determinism under fixed settings is checked below |
| Per-step reward components and physical metrics saved | **No** (only per-cluster aggregates). The env exposes them in `info`; the new evaluator saves them per step (`rollouts/*.npz`) |
| Existing parameter-sharing PPO or local-critic ablation | **None** in `src/`, `scripts/`, `research/` or the reports |
| Test reset | `MultiAgentUPFEnv(split="test").reset(seed=42)`. The reset is deterministic (no random start), so the seed has no effect |

### 1.4 Cost per seed (32-core host; measured single-threaded with `/usr/bin/time -v`)

| Controller | Wall time | Peak RSS | Storage | Hardware |
|---|---|---|---|---|
| MAPPO | 4 min 03 s | 1.24 GB | 0.34 MB | 1 CPU thread; GPU not used |
| Shared-PPO | ≈ MAPPO (8 updates in 8.7 s smoke test) | ≈ 1.2 GB | 0.2 MB | 1 CPU thread |
| Centralized PPO | 6 min 24 s | 1.69 GB | 1.3 MB | 1 CPU thread |
| IPPO | 10 × single-site PPO (~110 s each, Phase-7 recurrent-ablation log) | ~0.6 GB per worker | 4.6 MB | 3 CPU threads |

**Feasibility estimate given before launch:** all four controllers × 8 seeds fit in
under one hour of wall time on this host, against ≈ 56 h until the 16 Sep 12:00
compute cutoff. No prioritisation was needed.

## Phase 2 — Centralized PPO on the verified eight seeds

### 2.1 What was run

All four learned controllers were retrained on seeds {1, 7, 13, 23, 42, 64, 77, 99}
with identical code: commit `71a1b2f` plus two new untracked files, twin v0.4.0 @
`95de456`, the current reward, CPU, one thread per process. Every script ran
unmodified at its defaults with 200,000 environment steps and best-of-validation
checkpoints (`run_all.sh`). The legacy checkpoints were not touched.
Centralized PPO used the same environment, splits, observation, load-weighted
training reward, forecast inputs, hyper-parameters (`repro/hyperparameters.json`),
validation selection, test reset and deterministic evaluation as the existing
`central_nr_*` runs. Nothing was tuned; no run was discarded.

**Reproducibility check before launch.** Existing checkpoints re-evaluate to their
reported values (§1.3). Training determinism under fixed settings was also
confirmed: two independent MAPPO seed-23 trainings gave bitwise-identical best and
final checkpoints, and two centralized seed-23 trainings gave the identical test
reward (−18,575.4). Across *different* execution settings training is not
reproducible: the 2026-08-27 centralized seed-23 run scores −27,425.2.

### 2.2 Result (held-out test slice, 8 seeds each)

| Controller | Reward mean ± SD | Median | 95 % t-CI | Wh | QoS-viol. % | USR % | Sw/cluster/day |
|---|---|---|---|---|---|---|---|
| MAPPO | −5,624 ± 22 | −5,626 | [−5,642, −5,605] | 1,967 | 0.52 | 7.8 | 2.88 |
| Shared-PPO | −5,707 ± 77 | −5,704 | [−5,771, −5,642] | 1,963 | 0.57 | 8.5 | 3.00 |
| IPPO | −8,236 ± 6,912 | −5,794 | [−14,014, −2,457] | 2,026 | 1.47 | 8.9 | 2.03 |
| **Centralized PPO** | **−15,025 ± 6,931** | **−13,365** | **[−20,819, −9,231]** | 2,132 | 4.03 | 11.1 | 1.66 |
| Hysteresis (b = 50 Mbps) | −6,262 | – | – | 1,995 | 0.68 | 7.5 | 1.06 |
| Always-DPDK | −7,201 | – | – | 2,071 | 0.38 | 0.0 | 0.00 |

Per-seed centralized-PPO test rewards: 1: −20,317 · 7: −11,568 · 13: −11,179 ·
23: −18,575 · 42: −7,370 · 64: −8,208 · 77: −27,823 · 99: −15,161. **All eight are
worse than always-DPDK and than every MAPPO and Shared-PPO run.**
MAPPO − centralized PPO = +9,401 [+5,072, +14,158] (paired bootstrap), exact
two-sided p = 0.0078 (the minimum attainable with n = 8), Holm 0.031, 8/8 seeds.

**Discrepancy with the manuscript.** The manuscript reports −8,542 ± 1,744 over 4
seeds. The eight-seed same-code result is −15,025 ± 6,931. The existing centralized
sets also disagree with each other: −14,459 (v0.4 set), −18,457 (`central_nr`, 7
seeds) and −12,269 (May-16 set). The failure mode is unchanged — unsafe USR use on
some clusters, with QoS penalties 1.1–17× MAPPO's per seed (7.1× on means) — but its size and seed variance are
much larger than the manuscript states. The "failure is consistent" justification
for using four seeds does not hold for the magnitude.

## Phase 3 — Shared-actor / local-critic ablation (**Shared-PPO**)

**Name.** *Shared-PPO* = one actor and one critic, both parameter-shared across
the ten sites; the critic sees only the corresponding agent's own observation.

### 3.1 Implementation and verification

- `src/trainers/shared_ppo.py` subclasses MAPPO. Only the critic construction,
  the value forward in rollout, update and bootstrap, and one `critic_type` field in
  the checkpoint differ; the method-level diff against `MAPPO` was inspected.
  `scripts/train_shared_ppo.py` is `train_mappo.py` with the trainer class swapped
  (same arguments and defaults). No existing file was modified.
- **Shapes.** Actor input (10, 14) → (10, 2) logits. MAPPO critic 140 → 10 values;
  Shared-PPO critic (B, 14) → (B,). The global state equals the concatenated local
  observations exactly.
- **Architecture.** Identical actor, 14-64-64-2 with ReLU, **5,250 parameters**.
  MAPPO critic 140-128-128-10 (35,850); Shared-PPO critic 14-128-128-1 (18,561).
  For the same seed the actor's initial weights are **bitwise identical** to MAPPO's.
- **Unchanged.** Same Adam over actor + critic (lr 3e-4, eps 1e-5), gradient-norm
  clip 0.5, vf 0.5, entropy 0.05, γ 0.995, λ 0.9, clip 0.2, rollout 1,024 env steps
  (10,240 pooled agent-steps), minibatch 256, 10 epochs, reward × 0.01, advantage
  normalisation over (T, K), evaluation every 4,096 steps, best-of-val checkpoint,
  200,000 env steps. Env and reward are unchanged: always-DPDK is still −7,201.08 and
  the config/env hashes are in `repro/code_and_environment.txt`.
- **Smoke test.** 8,192 steps ran and saved a checkpoint with the expected critic
  shapes.
- **Timing.** One full MAPPO seed of the same code path took 4:03. The eight MAPPO
  and eight Shared-PPO training runs then took 3:51–4:21 each, with 24 processes in parallel.
  **Deviation:** the eight Shared-PPO seeds were launched together after the smoke
  test and the full MAPPO timing, rather than after a first full Shared-PPO seed.
  With a ~56 h margin this carried no schedule risk.

### 3.2 Seed-matched results (reward; 8 pairs; two-sided tests)

| Contrast | Per-seed differences (seeds 1, 7, 13, 23, 42, 64, 77, 99) | Mean | Median | Paired bootstrap 95 % CI | Exact perm. p | Holm p | Wilcoxon p | d_z | A better |
|---|---|---|---|---|---|---|---|---|---|
| **Δ_crit = MAPPO − Shared-PPO** | +292, +4, −23, +75, +46, +93, +77, +97 | **+82.9** | +76.4 | **[+31.0, +150.9]** | **0.023** | 0.047 | 0.023 | 0.87 | 7/8 |
| Δ_share = Shared-PPO − IPPO † | −95, +221, +113, +170, +42, +43, +104, +19,633 | +2,529 | +108.9 | [+34.6, +7,433] | 0.039 | 0.047 | 0.039 | 0.37 | 7/8 |
| G = MAPPO − IPPO | +198, +225, +91, +245, +88, +136, +182, +19,730 | +2,612 | +189.8 | [+136.9, +7,514] | 0.0078 | 0.031 | 0.0078 | 0.38 | 8/8 |

† Δ_share mixes parameter sharing with implementation and hyper-parameter
differences: custom trainer vs SB3; lr 3e-4 vs 1e-4; entropy 0.05 vs 0.15;
minibatch 256 vs 64; reward scaling; critic width. It is **not** a pure
parameter-sharing effect.

**IPPO seed 99 is a catastrophic but genuine run.** Cluster 3 (mean test load
0.92 Gbps) selected a best-of-validation checkpoint that uses USR on 73 % of test
steps: QoS-violation 72 %, QoS penalty 19,326, total −25,341. Its validation return
was the best among that run's evaluations (−27,575 at step 163,840), consistent with
the near-zero-load validation caveat (§1.3). The legacy June run of the same seed kept
cluster 3 on DPDK. It is included in every primary statistic, as pre-registered.

**Where Δ_crit comes from** (MAPPO − Shared-PPO, mean over seeds):

| Component | Difference | 95 % CI |
|---|---|---|
| QoS penalty | −107 | [−203, −9] |
| α·SEC cost | +25 | [−33, +75] |
| Energy | +4.2 Wh | [−2.7, +10.3] |
| QoS-violation rate | −0.048 pp | [−0.092, −0.005] |
| USR share | −0.68 pp | [−1.48, +0.14] |
| Switches per cluster per day | −0.12 | [−0.40, +0.17] |

With the centralized critic, the shared actor uses USR slightly more conservatively
and incurs fewer QoS penalties at essentially equal energy. It is also far more
stable across seeds: reward SD 22 for MAPPO vs 77 for Shared-PPO.

**Per cluster** (median over seeds, `per_cluster_contrasts.csv`, `fig_per_cluster`):

- The critic's contribution is concentrated on the mid-load clusters c5 (+37),
  c7 (+29) and c8 (+17).
- The sharing (+ implementation) contribution is largest on the lowest-load cluster
  c0 (+46) and on c5 (+30).
- IPPO is better on c2 (−19).
- The high-load clusters c3 and c9 are DPDK-bound for every controller, apart from
  IPPO seed 99's cluster-3 failure.

## Phase 4 — Evaluation protocol and outputs

Every controller in every set was evaluated once:
`MultiAgentUPFEnv(split="test")`, `reset(seed=42)`, the full 1,009-step episode,
argmax actions from the best-of-validation checkpoint, using the Phase-7 policy
loaders. Per-step, per-cluster arrays are saved in `rollouts/*.npz`. Two consistency
assertions pass for all 72 rollouts: reward = −(sum of the four components), and
requested switches = realised switches.

### 4.1 Continuity: the same evaluator applied to the existing checkpoints

| Set | n | Reward mean ± SD | Wh | QoS-viol. % | USR % | Sw/d | Note |
|---|---|---|---|---|---|---|---|
| Manuscript Table I, MAPPO | 8 | −5,655 ± 98 | 1,966 | 0.54 | 8.1 | 2.92 | pre-v0.4 twin; seeds 7/13/77 untraceable |
| legacy MAPPO (v0.4 re-scored) | 8 | −5,647 ± 92 | 1,967 | 0.53 | 7.8 | 2.93 | May checkpoints, earlier reward |
| **MAPPO (same-code, this study)** | 8 | **−5,624 ± 22** | 1,967 | 0.52 | 7.8 | 2.88 | |
| Manuscript Table I, IPPO | 8 | −5,812 ± 110 | 1,973 | 0.57 | 8.2 | 1.99 | = `ippo_nr` |
| legacy IPPO (`ippo_nr`) | 8 | −5,814 ± 117 | 1,974 | 0.57 | 8.2 | 1.99 | |
| **IPPO (same-code, this study)** | 8 | **−8,236 ± 6,912** (median −5,794) | 2,026 | 1.47 | 8.9 | 2.03 | seed 99 catastrophic |
| Manuscript Table I, centralized PPO | 4 | −8,542 ± 1,744 | 2,066 | 1.18 | 4.3 | 0.44 | untraceable |
| **Centralized PPO (same-code, this study)** | 8 | **−15,025 ± 6,931** | 2,132 | 4.03 | 11.1 | 1.66 | |

## Conclusion — which statement the evidence supports

**Pre-registered outcome: Statement 1 — the centralized critic materially improves
over a shared local critic** (`decision.json`). Δ_crit = +82.9 reward units, 95 % CI
[+31, +151], exact two-sided p = 0.023, 7 of 8 seeds, d_z = 0.87, above the fixed
materiality threshold M = 56.2 (1 % of MAPPO's cost).

The effect is real but **modest**, and it should be reported with these qualifications:

1. **Size.** It is about 1.5 % of MAPPO's total cost, comparable to the
   sharing-plus-implementation contribution. Excluding the one catastrophic IPPO seed
   (post hoc): G = +166 [+122, +208], of which Δ_crit = +81 (49 %) and Δ_share = +86
   (51 %). Medians: Δ_crit +76, Δ_share +109, G +190.
2. **Materiality is marginal.** Leave-one-seed-out, Δ_crit ranges +53 to +98 and stays
   significant in 8/8 subsets, but falls below M when seed 1 is dropped. Holm-adjusted
   across the four primary contrasts, p = 0.047.
3. **Mechanism.** It comes from fewer QoS penalties (−107), not from energy (+4 Wh,
   not significant). The centralized critic also reduced seed-to-seed spread 3.5×.
4. **The S2 conditions also formally hold**, but only because IPPO seed 99 inflates
   both Δ_share and G. By the pre-registered precedence S1 is reported. The evidence
   does **not** support S2 ("parameter sharing explains *most* of the advantage");
   in typical seeds sharing explains about half, and that half is confounded with
   implementation differences.
5. Statement 4 is contradicted: MAPPO outperforms Shared-PPO on 7/8 seeds.

## How the manuscript's claims must change

| Current manuscript statement | Status after this study | Suggested revision |
|---|---|---|
| Table I centralized PPO −8,542 ± 1,744, n = 4 | Not traceable; the same-code 8-seed value is −15,025 ± 6,931 | Replace with the 8-seed row (`centralized_ppo_8seed_summary.tex`); drop "four seeds because its failure mode is consistent" |
| Table I MAPPO −5,655 ± 98 / IPPO −5,812 ± 110 | Built from checkpoints of mixed reward/twin versions; three MAPPO seeds untraceable | Report the same-code set (MAPPO −5,624 ± 22; IPPO −8,236 ± 6,912, median −5,794) with the IPPO seed-99 failure disclosed, or state clearly that Table I mixes versions |
| "Paired bootstrap … one-sided p = 0.004" | One-sided test, with no evidence the direction was fixed in advance | Two-sided exact permutation: G = +2,612 [+137, +7,514], p = 0.0078, 8/8 seeds; excluding the failed IPPO seed (post hoc) +166 [+122, +208], p = 0.016 |
| Gains explained by cross-site transfer through the shared actor, plus the centralized critic | Both contribute, about equally in typical seeds | "Roughly half of MAPPO's advantage over IPPO is attributable to the centralized critic (+83 reward units, p = 0.023; mainly fewer QoS penalties and lower seed variance) and the remainder to policy sharing, which is confounded here with implementation differences" |
| Per-cluster story: c5, c0, c7, c4 gains via transfer | c0 gains come from sharing; c5, c7 and c8 gains partly from the critic | Qualify accordingly (`fig_per_cluster`) |
| "MAPPO has the lowest energy among the trained policies" | Shared-PPO is 4 Wh lower (n.s.) | "MAPPO and Shared-PPO reach the same energy (1,967 vs 1,963 Wh)" |
| "Coordination structure is a key factor" | Supported against centralized PPO (large) and against IPPO; the critic effect is modest | Keep, but avoid implying a large centralized-critic effect |

## Deviations, limitations and threats to validity

1. **Scope.** The request was to add *missing* centralized seeds and compare against
   the existing MAPPO/IPPO. Because the existing checkpoints come from different
   reward and twin versions (§1.2), all four controllers were retrained with identical
   code instead. Legacy results are kept separately (`all_seed_metrics.csv`,
   `group = legacy`). The legacy `central_nr` set was not topped up with a seed-99 run:
   training is not reproducible across execution settings (§2.1), so a top-up would
   not have been comparable to the August runs either.
2. **One full Shared-PPO seed was not run in isolation** before launching all eight
   (§3.1). The runtime estimate came from the smoke test plus a full MAPPO seed of the
   same code path.
3. **The IPPO comparison is confounded.** IPPO uses SB3 with different
   hyper-parameters. Δ_share, and so any "parameter-sharing" attribution, mixes
   sharing with implementation. A matched independent-actor variant in the MAPPO
   trainer was not built.
4. **Centralized PPO** trains on the load-weighted joint reward and is evaluated on
   the unweighted sum, as in the manuscript.
5. **Validation selection** for every controller is dominated by 684 near-zero-load
   cluster-steps in the validation slice. This plausibly produced the IPPO seed-99
   failure.
6. **n = 8.** The smallest attainable exact two-sided p is 0.0078. Seed pairing is
   genuine only for MAPPO vs Shared-PPO (identical actor initialisation).
7. **Energy** is the twin's modelled UPF-attributed power (Scaphandre per-process
   scope), not wall power.
8. **Post-hoc analyses** (leave-one-out, exclusion of runs worse than always-DPDK,
   medians) are labelled as such and did not enter the decision.

## Reproduction

```bash
# training (writes experiments/wcnc2027_ablation/; refuses to overwrite)
bash reports/wcnc2027-shared-policy-ablation/run_all.sh train_a   # MAPPO, Shared-PPO, centralized PPO
bash reports/wcnc2027-shared-policy-ablation/run_all.sh train_b   # IPPO
# evaluation, statistics, tables, figures
bash reports/wcnc2027-shared-policy-ablation/run_all.sh evaluate
bash reports/wcnc2027-shared-policy-ablation/run_all.sh analyze
```

A single run, e.g. Shared-PPO seed 13:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/train_shared_ppo.py \
    --seed 13 --no-progress --out-dir experiments/wcnc2027_ablation/shared_ppo_seed13
```

Configuration and environment: `repro/scenario_rl.yaml`,
`repro/digital_twin_paths.yaml`, `repro/hyperparameters.json` (resolved defaults of
every trainer), `repro/pip_freeze.txt`, `repro/code_and_environment.txt` (git HEAD,
twin commit, sha256 of all code used). Training logs with `/usr/bin/time -v` are in
`experiments/wcnc2027_ablation/logs/`.

## Files

| File | Content |
|---|---|
| `feasibility_and_results.md` | this report |
| `analysis_plan.md` | pre-registered metrics, tests and decision rule |
| `all_seed_metrics.csv` | every controller × seed × cluster (`all` = fleet): reward, energy, QoS-violation, q_score < τ, USR share, switches, switches per cluster per day, reward components, checkpoint path |
| `aggregate_metrics.csv` | mean, SD, median, min/max, t- and bootstrap 95 % CIs |
| `per_cluster_summary.csv`, `per_cluster_contrasts.csv` | per-cluster aggregates and ablation contrasts |
| `paired_comparisons.csv` | seed-matched contrasts for every metric: per-seed differences, bootstrap CI, exact permutation, Wilcoxon, d_z, g_z, Holm |
| `decision.json`, `sensitivity_posthoc.json` | pre-registered decision; post-hoc robustness |
| `audit_existing_checkpoints.csv` | all 52 legacy checkpoints re-evaluated, with provenance |
| `centralized_ppo_8seed_summary.tex`, `shared_policy_ablation_summary.tex` | IEEEtran `table*` (compiled without overfull boxes) |
| `figures/fig_seed_rewards`, `fig_paired_differences`, `fig_energy_qos`, `fig_per_cluster`, `fig_reward_components` (`.pdf` + `.png`) | publication figures |
| `evaluate_all.py`, `analyze.py`, `make_figures.py`, `run_all.sh`, `repro/` | code, commands and configuration |
| `rollouts/*.npz`, `rollouts/manifest.json` | per-step, per-cluster test trajectories for all 72 evaluated runs |
| `src/trainers/shared_ppo.py`, `scripts/train_shared_ppo.py` | the new Shared-PPO trainer (new files only) |
