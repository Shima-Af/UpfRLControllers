# Why do MAPPO and Shared-PPO reach different energy–QoS trade-offs?

Investigation for the WCNC 2027 revision, 2026-09-17. The cohort is the primary eight-seed set (seeds 1, 7, 13, 23, 42, 64, 77, 99; set S2 in the calibration study). No legacy checkpoint is used.

**Evidence classes used below.**

- **(a) Existing saved evidence.** Everything in this report is (a).
- **(b) Software evaluation of the archived models.** Described in §8 and **not run**.
- **(c) New training.** Described in §8 and **not run**.

No implementation, checkpoint, experiment or manuscript file was changed. No policy was run and no environment was stepped. The new files are analysis scripts and outputs in `reports/wcnc2027-mappo-sharedppo-mechanism/`.

## 1. Answer in brief

### Supported by the saved records (a)

1. **The gap is a QoS-penalty difference at jump-forced exits from user space.**
   - Mean over 8 seeds (Shared-PPO − MAPPO cost): QoS penalty **+106.9**, SEC term **−25.4**, switching +0.2, cooldown +1.2; reward −82.9.
   - Decomposed, the QoS difference is +145.8 on exits only Shared-PPO was exposed to, −35.6 on exits only MAPPO was exposed to, and −3.4 on one MAPPO step. `decomposition_by_seed.csv`, Fig. 1c.
   - The QoS penalty is not incurred where the two policies choose different UPFs.
2. **The two policies agree on 98.5 % of cluster-steps.**
   - They disagree on 150 of 10,090 per seed: Shared-PPO selects USR where MAPPO selects DPDK on 109.5; the reverse on 40.9.
   - **Every disagreement step is QoS-safe for USR.** Both controllers essentially never select USR in a QoS-risky state: once in 80,720 cluster-steps for MAPPO (seed 42), never for Shared-PPO.
   - The disagreements only move the SEC term: −60.7 on Shared-PPO's extra USR steps, +35.1 on MAPPO's. `usr_usage_by_risk.csv`, `disagreement_cells_summary.csv`.
3. **The disagreements sit just below the decision threshold.**
   - Shared-PPO selects USR on **27.3 %** of cluster-steps with 50–81 Mbps offered load; MAPPO on **18.4 %**. Shared-PPO is higher in 6/8 seeds.
   - Of Shared-PPO's extra USR steps, 76.5 per seed fall at 50–81 Mbps, 23.6 at 25–50 and 9.3 at 81–149. At 25–50 Mbps the two controllers are equal (85 % vs 84 %). `load_bin_summary.csv`, Fig. 1a.
4. **The QoS penalty comes from exposure to one-step load jumps.**
   - Both controllers pay about 23 penalty units whenever they are in USR on the step before load jumps into the USR-unsafe range. The median jump is 41–47 → 241 Mbps within one 15-min interval.
   - The observation contains the current load, so both leave USR on that same step. The twin nevertheless charges the switching step the larger of the two UPFs' predicted loss (`upf_digital_twin/twin/digital_twin.py::compute_step`, v0.4.0 @95de456).
   - In every run each such exposure produced exactly one penalized exit. The only extra penalized exit is MAPPO seed 42's, which follows its single USR entry into a risky state, so exits equal exposures in 15/16 runs. Neither controller reacts to the jump differently.
   - Exposures per seed are 14.0 (MAPPO) vs 19.1 (Shared-PPO), with Shared-PPO higher in 6/8 seeds. The penalized exits track USR use at 50–81 Mbps (Spearman ρ = 0.97, 16 runs). `switch_steps.csv`, `penalized_exits.csv`, `exposure_by_run.csv`.
5. **In this traffic, holding USR at 50–81 Mbps loses on average.**
   - Holding USR there saves 0.37 SEC-units per step, but the next step turns USR-risky with probability 0.089 (test). Expected exit penalty is 2.06, so the one-step net is **−1.7**. At 25–50 Mbps the net is +0.9. `hold_vs_onset_risk_by_band.csv`, Fig. 1b.
   - The onset probability from 50–81 Mbps is similar in the training (0.081) and validation (0.080) traffic. This uses the proxy "load ≥ 149 Mbps", which agrees with the twin's USR-risk flag on 98.9 % of test cluster-steps and never misses a risky one; `onset_risk_by_split_proxy.csv`.
   - So the more conservative mid-load behaviour of MAPPO is the better trade under this reward, and the information to learn it was present in training.
6. **Jumps are not predictable from the local observation.**
   - The forecast in the observation predicts the *current* step: correlation 0.960 with that step's load, 0.983 with the previous step's, 0.950 with the next step's.
   - At penalized exits the forecast was 44–52 Mbps against 241 Mbps actual.
   - A rising load over the previous step does not raise the onset probability: 0.078 when rising vs 0.098 otherwise, at 50–81 Mbps.
7. **Physical energy is not a reliable difference.** Shared-PPO uses 4.2 Wh less (paired 95 % CI [−2.7, +10.3], exact two-sided p = 0.29, only 2/8 seeds favour MAPPO). Keep Wh separate from the SEC term: SEC weights savings at low load more heavily.
8. **Checkpoint selection differs systematically.**
   - Shared-PPO's best-of-validation checkpoint comes from an earlier training step in **8/8 seeds** (mean 51k vs 71k of 200k steps) and has a worse validation return in **8/8** seeds.
   - Validation return is dominated by the 684 near-zero-load cluster-steps of the validation slice (6.8 %; train has 4 and test has 0).
   - Most runs later fall to about −4.9·10⁷, the always-DPDK level on that slice.
   - Across the 16 runs the selected step is **not** associated with USR use at 50–81 Mbps (ρ = −0.19). `checkpoint_selection.csv`, Fig. 2.
9. **The mechanism also shows across calibrations** (closed-policy calibration study, set S2).
   - The MAPPO − Shared-PPO reward gap follows the twin's USR QoS limit:
     - LOLO-dep-f2 (limit 81 Mbps): +570, with a QoS difference of +588;
     - LOLO-dep-f3 (limit 170 Mbps): −27 and −3.
   - The other 8 primary worlds give +32 to +109.

### Plausible, not established

- **Centralized critic.** It may have helped MAPPO learn the less exposed mid-load behaviour.
  - Other sites' observations carry some temporal context: in 46–52 % of onsets, another cluster crossed 149 Mbps one step earlier.
  - Per-agent value heads let MAPPO's critic learn cluster-specific baselines.
  - But no critic prediction, return target, advantage or value loss was saved, so there is no evidence that value estimates around these states differed.
- **Optimization variability and checkpoint timing.** Either could equally produce the difference.
  - Seed 13 favours Shared-PPO (−22.7) and seed 7 is a tie (+4.1).
  - The per-seed reward gap is only loosely tied to the per-seed exposure gap (ρ = 0.55, p = 0.15).

### Unknown from the saved records

- Why Shared-PPO's actor places its effective threshold higher at 50–81 Mbps. Action probabilities and decision margins were not saved.
- Whether critic information, critic capacity, agent-specific heads, joint gradient clipping, the random-stream divergence or checkpoint timing causes it. These are confounded (§3).
- How sensitive the gap is to the twin's max-loss switching-step rule. Recomputing it would be a (b) evaluation.

## 2. Provenance of the results

`provenance_summary.json`, `provenance_code.csv`, `provenance_checkpoints.csv`, `provenance_rollouts.csv`.

| Item | Identity | Status |
|---|---|---|
| Repository | `UpfRLControllers` HEAD `71a1b2fc`, branch `chore/cleanup-research-vs-library` | unchanged since training |
| MAPPO trainer | `src/trainers/mappo.py` sha256 `fd8d81ab…` (git-tracked) | identical to the hash recorded at launch (`reports/wcnc2027-shared-policy-ablation/repro/code_and_environment.txt`) |
| Shared-PPO trainer | `src/trainers/shared_ppo.py` sha256 `762d9725…`; `scripts/train_shared_ppo.py` `e00f3f02…` | identical to launch hashes, but **untracked by git**; provenance rests on the recorded hash |
| Training scripts and environment | `scripts/train_mappo.py` `b9282eb6…`; `src/envs/multi_agent_upf_env.py` `ab812bb8…`; `single_site_upf_env.py` `f1feb81a…`; `configs/scenario_rl.yaml` `f4637a75…`; `configs/digital_twin_paths.yaml` `900e1dd8…` | identical to launch hashes |
| Fitted environment (twin) | `upf_digital_twin` v0.4.0 @`95de456`; surrogate bundle `data/external/profiling_twin/models` (DVC dir md5 `a92ad2ca…`); `switching_costs.yaml` thesis-v1.2 sha256 `dc559bbe…` | same install used for training and evaluation |
| Traffic, forecasts, split | `data/external/traffic_forecaster/{targets,predictions}_{train,val,test}.npy` (test `b49aafe5…` / `bd679942…`); 5,073 / 1,009 / 1,009 steps × 10 clusters, horizon index 0, α = 1.0 | chronological split, same for both |
| Training runs | `run_all.sh train_a` (`reports/wcnc2027-shared-policy-ablation/run_all.sh`), 2026-09-14 04:03:32–04:10:22; defaults: 200k steps, n_steps 1024, 10 epochs, minibatch 256, lr 3e-4, γ 0.995, λ 0.9, clip 0.2, ent 0.05, vf 0.5, grad clip 0.5, reward scale 0.01, evaluation every 4,096 steps on validation | checkpoint config dicts are identical apart from the seed |
| Checkpoints | `experiments/wcnc2027_ablation/{mappo,shared_ppo}_seed<N>/mappo_best.pt` (used), `mappo_final.pt`, `eval_log.json`; logs `experiments/wcnc2027_ablation/logs/*.log`, `*.time` | all 16 best checkpoints match the calibration study's hash inventory |
| Test rollouts (float64, reward costs) | `reports/wcnc2027-shared-policy-ablation/rollouts/{MAPPO,Shared-PPO}__seed<N>.npz`, from `evaluate_all.py` (sha256 `2ac6e77a…`, which imports the Phase-7 loaders `research/phase7/evaluate_multiseed.py` `03a6c57f…`) | reproduce the paper's fleet means exactly |
| Test rollouts (delay, loss) | `reports/wcnc2027-calibration-robustness/rollouts/main/pinned/A/*` (evaluator `evaluate_worlds.py`) | **action arrays identical** to the ones above for all 16 runs |
| Counterfactual per-UPF outcomes | same folder, `Always-DPDK__-.npz`, `Always-USR__-.npz` | same twin and traffic |

**Missing or ambiguous links.**

- The Shared-PPO trainer and script are not committed. Commit them, or quote the recorded sha256 values.
- MAPPO checkpoints carry no `critic_type` field. The critic type was inferred from the saved critic shapes (140 → … → 10).
- `evaluate_all.py` was not hashed at launch. It is cross-validated by the independent evaluator producing identical actions and rewards.
- Training determinism under fixed settings was shown for MAPPO seed 23 (`experiments/wcnc2027_ablation/_repro/mappo_seed23`). No full-length Shared-PPO repeat exists; `_repro/smoke_shared_ppo_seed23` has only 2 evaluations.

## 3. What differs between the implementations

| Aspect | MAPPO | Shared-PPO | Difference beyond critic information? |
|---|---|---|---|
| Actor | `Actor` 14-64-64-2, ReLU, orthogonal init (head gain 0.01) | same class | No. Same torch seed and the actor is built first, so the initial actor weights match *in code*. Not artifact-verified: initial weights were not saved. |
| Critic input | 140-d concatenation of all 10 local observations (`MultiAgentUPFEnv.state()`; `mappo.py` L100-118, 309, 369) | the agent's own 14-d observation (`shared_ppo.py` L41-59, 112, 196) | intended difference |
| Critic output | 10 heads, one per cluster, so **agent identity is implicit** | 1 shared head; the observation has **no cluster identifier** | **Yes.** Agent identity is confounded with cross-site information. |
| Critic size | 35,850 parameters | 18,561 parameters | **Yes**, capacity differs (1.9×) |
| Value loss | MSE on the sampled (t, k) entries, mean over 256 | same | No (same scaling and coefficient 0.5) |
| Optimizer | one Adam over actor + critic, lr 3e-4, eps 1e-5; **one gradient-norm clip of 0.5 over actor + critic** (`mappo.py` L424-427) | same | **Yes, indirectly.** The actor's clipped step depends on the critic's gradient norm, which differs between critics. |
| Reward | per-agent unweighted reward × 0.01; pool and budget penalties disabled (`scenario_rl.yaml`: `power_cap_w: null`, `budget_wh: null`) | same | No |
| Rollout, GAE, advantage normalization, minibatching, epochs | 1,024 steps × 10 agents; per-agent GAE (γ 0.995, λ 0.9) bootstrapped from the critic; advantages normalized over (T, K); 256 agent-steps × 10 epochs; numpy shuffle | same code; bootstrap from the local critic | Only through V |
| Random streams | torch stream consumed by a 140-input critic init | consumed by a 14-input critic init | **Yes.** Action sampling diverges from the first training step, so "matched seeds" do not give matched trajectories. |
| Checkpoint selection | best deterministic validation return, every 4,096 steps, strict improvement | inherited, same rule | Same rule, but it selects different steps (8/8 earlier for Shared-PPO) |
| Evaluation | argmax of the saved actor on the test slice, `reset(seed=42)` | same | No |

**Physical coupling.** None.

- Each site's reward, delay, loss, power and cooldown depend only on its own actions and load (`MultiAgentUPFEnv.step`, separate `SingleSiteUPFEnv` instances). The only cross-site reward terms (fleet power pool, energy budget) are disabled.
- Other sites therefore cannot affect a site's outcomes. Their observations can matter only as **predictive context inside MAPPO's critic during training**; the actors of both controllers see only local observations.

## 4. Inventory of mechanism evidence

| Record | Saved? | Where / how |
|---|---|---|
| Offered load and forecast per step | yes | `rollouts/main/traffic_test.npz`; raw arrays in `data/external/traffic_forecaster/` |
| Observations | not saved; **reconstructed exactly** from saved records (SEC reconstruction error 0) | `m01_stepwise_analysis.py::derived` |
| Actions (argmax) and realised UPF | yes | both rollout sets |
| Action probabilities or logits | **no** | (b) |
| Previous UPF, switching events, cooldown state | derived exactly from the realised UPF sequence | `switch_steps.csv` |
| Delay, loss, Q score, violation flag | yes (delay and loss float32) | calibration rollouts |
| Separate reward costs per step | yes (float64) | ablation rollouts |
| Counterfactual outcome of each UPF per step | yes | always-DPDK and always-USR rollouts |
| Critic predictions | **no** | (b), and only for the best and final checkpoints |
| Training return targets, advantages, value loss | **no**. `update()` computes value loss, but `learn()` logs only entropy and KL | (c) |
| Entropy, KL | about 11 points per run | `logs/*.log` |
| Validation return | 48 evaluations per run | `eval_log.json` |
| Intermediate checkpoints | **no** (best and final only) | (c) |
| Per-step validation rollouts | **no** | (b) |

## 5. Analysis of the saved records

### A. Where Shared-PPO chooses USR while MAPPO chooses DPDK

- **Scale.** 876 cells over 8 seeds (109.5 per seed); the reverse 327 (40.9).
- **Load.** The median offered load of Shared-PPO-only USR steps is 60 Mbps (MAPPO-only: 51 Mbps); 76.5 of the 109.5 per seed fall in 50–81 Mbps.
- **Same inputs.** **58 %** of Shared-PPO-only USR steps occur with *identical 14-d observations* for both policies, including the same previous UPF, Q, SEC and cooldown. The remaining 42 % follow diverged histories (previous UPF differs in 31 %; `disagreement_cells.parquet`).
- **Forecast error and load transitions.** Disagreement rates rise where forecast and actual lie on opposite sides of 81 Mbps (2.7 % vs 1.1 % overall) and where load crossed 81 Mbps since the previous step (3.8 %). At 149 Mbps they are near zero. `strata_summary.csv`.
- **Clusters.** Disagreements concentrate on clusters 0, 6, 7 and 1 (median loads 100, 142, 176 and 242 Mbps). The QoS cost lands on clusters 5, 7, 6, 0, 4 and 8. Cluster 5 is worse for Shared-PPO in 8/8 seeds; clusters 4, 7 and 8 in 6/8. `cluster_summary.csv`.

### B. Do the disagreements explain the extra QoS penalty?

Mean per seed, Shared-PPO − MAPPO:

| Cells | n / seed | SEC term | QoS penalty | Switching | Cooldown | Wh (total) |
|---|---|---|---|---|---|---|
| Shared-PPO USR, MAPPO DPDK (all USR-safe) | 109.5 | **−60.6** | 0.0 | −1.3 | −3.2 | −8.3 |
| MAPPO USR, Shared-PPO DPDK (40.6 safe, one risky step in one seed) | 40.9 | **+35.1** | −3.4 | +0.4 | −0.5 | +3.8 |
| Same UPF: Shared-PPO leaves USR, MAPPO already in DPDK | 45.4 (of which 6.5 jump-forced) | 0.0 | **+145.8** | +1.7 | +12.9 | +0.5 |
| Same UPF: MAPPO leaves USR, Shared-PPO already in DPDK | 17.1 (of which 1.5 jump-forced) | 0.0 | **−35.6** | −0.7 | −5.6 | −0.2 |
| Same UPF, other state differences (entries, both switching, cooldown counter) | 3,197 | −0.1 | 0.0 | 0.0 | −2.5 | 0.0 |
| Same UPF and state | 6,680 | 0 | 0 | 0 | 0 | 0 |
| **Total** | 10,090 | **−25.4** | **+106.9** | **+0.2** | **+1.2** | **−4.2** |

- **What the table shows.** The disagreement steps are *safe energy-saving* assignments. The QoS cost is paid one step later, at jump-forced exits, and only by the controller that was still in USR.
- **Per penalized exit.** Q = 0.17 (predicted loss 9.2 packets against the 5-packet budget), QoS penalty about 23, switching cost 0.009 Wh.
- **Counts.** Penalized exits per seed are 14.1 (MAPPO) vs 19.1 (Shared-PPO).
- **Energy.** Physical Wh follows the SEC term only loosely: −8.3 Wh on Shared-PPO's extra USR steps against −60.6 SEC-units, because SEC weights low-load savings more.

### C. Learning diagnostics

- **What exists.** No critic, return or advantage records. The only saved learning signal is validation return, which measures one deterministic validation episode's total reward.
- **What it shows.**
  - It is dominated by the near-zero-load steps of the validation slice (always-DPDK scores −5.0·10⁷ there).
  - It oscillates by two orders of magnitude between evaluations (Fig. 2a).
  - Best-of-validation selection therefore picks snapshots from 14–55 % of training, earlier for Shared-PPO in every seed.
- **What it cannot show.** It says nothing about action-specific preferences at 50–81 Mbps test states. It also cannot say whether the critics' state values were less accurate around exposure states: state values would not identify an action preference in any case, and none were saved.

### D. Consistency

- **Across seeds.** The direction (more mid-load USR, more jump exposure, more QoS penalty, less SEC) holds in 6/8 seeds. The exceptions are seeds 7 and 13, where Shared-PPO uses *less* USR at 50–81 Mbps (−8.8 and −8.5 pp) and is not worse (+4.1, −22.7).
- **Across calibrations.** The pattern holds across the calibration worlds (§1 item 9).

## 6. What the evidence does and does not support about the critic

The saved evidence explains **how** the reward gap arises: more USR exposure at 50–81 Mbps, which the twin's switching-step accounting penalizes when traffic jumps. It does **not** isolate **why** Shared-PPO's actor ended up with that behaviour. At least five differences are confounded with the critic's information:

1. agent-specific value heads;
2. critic capacity;
3. joint actor–critic gradient clipping;
4. random-stream divergence after critic initialization;
5. a best-of-validation criterion that selected earlier checkpoints for Shared-PPO in 8/8 seeds, dominated by validation states that never occur on the test slice.

A generic "the centralized critic gives better credit assignment" statement is not supported by these records.

## 7. Recommendation

- **Keep the result:** MAPPO > Shared-PPO by +83 reward units, exact two-sided p = 0.023.
- **Replace the causal wording with the mechanism evidence below.**
- **Do not claim** that MAPPO is more energy-efficient than Shared-PPO, or that the centralized critic causes the difference.
- **Name the confounds** and the dependence on the twin's switching-step QoS accounting.
- **If the causal claim matters for the paper**, run the (b) diagnostics first (fast, no training), then the targeted (c) controls.

**Suggested paragraph (claims limited to the saved evidence).**

> Over the eight matched-seed test episodes, MAPPO's reward advantage over the shared-actor, local-critic variant (+83 reward units) is almost entirely a difference in QoS penalty (−107), partly offset by a small SEC-term saving of Shared-PPO (+25); the physical energy difference (4.2 Wh) is not significant. The two policies select the same UPF on 98.5 % of cluster-steps and neither uses the user-space UPF where it is predicted to violate QoS. Shared-PPO, however, keeps the user-space UPF more often at 50–81 Mbps (27 % vs 18 % of such steps). When traffic then rises within one 15-min interval into the range where the user-space UPF is predicted to lose packets, the controller switches back to DPDK, but the switching interval is charged the user-space loss. Such jump-forced exits occur 19.1 vs 14.1 times per episode and account for the QoS difference. The saved records do not establish whether the centralized critic causes MAPPO's more conservative mid-load behaviour: the two variants also differ in value-head structure, critic size, and the training step of the selected checkpoint.

## 8. Recovering the missing evidence

### (b) Software evaluation of the archived models

These need only the frozen checkpoints and the saved or reconstructed test states, with twin v0.4.0 for items 3–5. Expected cost is minutes on CPU. **Not run.**

1. **Actor action probabilities** at every test state for both controllers, from the reconstructed observations. First check that argmax reproduces the saved actions, then compare the USR-probability margin at 50–81 Mbps and at identical-observation disagreements.
2. **Critic state values** on the test trajectories: MAPPO V(s_t)[k] from the concatenated observations, Shared-PPO V(o_{t,k}). Compute TD errors and GAE advantages with the saved rewards × 0.01 around exposure and onset steps.
   - Caveats: these are state values on the deterministic policy's trajectory, not action values or training-time estimates.
   - They show whether a critic anticipated jump risk, not which action it preferred.
3. **Evaluate `mappo_final.pt`** (both controllers) on test and validation. This tests whether the selected training step, rather than the architecture, drives the mid-load behaviour.
4. **Per-step validation rollouts** of the selected checkpoints. These quantify how much of each selection decision came from the near-zero-load validation steps.
5. **Re-score the saved action trajectories** under an alternative switching-step QoS rule, e.g. time-weighted rather than maximum loss during the 24 s activation. This measures how much of the gap is an accounting assumption.

### (c) New training experiments

1. **Exact re-runs of the 16 trainings with logging hooks** for value loss, value predictions, returns, advantages and periodic checkpoints. MAPPO training was shown bitwise-deterministic under fixed settings; each re-run must reproduce the saved `mappo_best.pt` sha256 before its diagnostics are used.
2. **Controls that separate critic information from its confounds:**
   - local critic plus a one-hot cluster ID;
   - capacity-matched critics;
   - separate gradient clipping for actor and critic;
   - more seeds.
3. **Selection-protocol control:** a validation criterion that excludes near-zero-load steps, or a fixed training step, applied identically to both controllers.
4. **Training under the alternative switching-step accounting.** Needed if item 5 of (b) shows large sensitivity.

## 9. Files and commands

```bash
cd reports/wcnc2027-mappo-sharedppo-mechanism
../../.venv/bin/python m00_provenance.py          # hashes, checkpoint metadata, selection history
../../.venv/bin/python m01_stepwise_analysis.py   # per-step decomposition from saved rollouts
../../.venv/bin/python m03_supporting_tables.py   # exposure, onset risk by band and split, penalized exits
../../.venv/bin/python m02_figures.py             # figures/fig_mechanism, figures/fig_checkpoint_selection (PDF+PNG)
```

| File | Content |
|---|---|
| `provenance_*.csv`, `provenance_summary.json`, `checkpoint_selection.csv` | §2, §5C |
| `decomposition_by_seed.csv` | cost deltas by cell category, USR-risk class and seed (§5B) |
| `disagreement_cells.parquet`, `disagreement_cells_summary.csv` | every disagreement cell with its context (§5A) |
| `load_bin_summary.csv`, `strata_summary.csv`, `cluster_summary.csv`, `usr_usage_by_risk.csv` | where and under which conditions disagreements occur |
| `switch_steps.csv`, `penalized_exits.csv`, `exposure_by_run.csv` | every switching step; jump-forced exits; exposure per run |
| `hold_vs_onset_risk_by_band.csv`, `onset_risk_by_split_proxy.csv` | one-step value of holding USR; onset risk in train, validation and test |
| `figures/fig_mechanism.{pdf,png}` | (a) USR use by load; (b) value of holding USR; (c) per-seed cost decomposition |
| `figures/fig_checkpoint_selection.{pdf,png}` | (a) validation return and selected checkpoints; (b) selection step vs mid-load USR use |
