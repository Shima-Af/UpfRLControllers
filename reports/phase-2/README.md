# Phase 2 — Single-Site PPO (paper-aligned, out-of-sample)

**Status:** PPO trained on the forecaster's training slice (~53 days,
5073 steps) and selected on a separate validation slice (~10.5 days,
1009 steps) generalises to the held-out **test slice** (1009 steps).
On cluster 0 it beats always-DPDK by **32.5%** in total reward and
**15.7%** in raw energy, with a **0.50%** QoS-violation rate. Reward
formulation and observation are aligned with COMCOM-S-26-00430
(Section 4.5 and 4.3), adapted to a binary action space (DPDK vs USR;
horizontal scaling is a future extension) and to a soft cooldown
penalty.

> **Methodology note.** Earlier iterations of this phase quoted
> in-sample numbers — PPO was trained on the only forecaster slice
> available locally (the test slice), then evaluated on the same
> slice. Once the upstream forecaster started emitting
> `predictions_{train,val,test}.npy`, we wired the env's new `split=`
> argument and retrained from scratch. Headline numbers below are
> now produced by [`scripts/evaluate_test_split.py`](../../scripts/evaluate_test_split.py),
> which is the *only* script in the repo allowed to touch the test
> slice for PPO. The trainer and `EvalCallback` are pinned to
> train/val.

## Summary

The agent picks DPDK or USR every 15 minutes for one cluster's full
1009-step traffic series. The reward combines:

- **Specific energy consumption** `SEC_t = P_t / max(load_Mbps, ε)` —
  watts per Mbps — scaled by α = 100.
- **Graded QoS penalty** `λ_QoS · max(0, τ − Q_t)` with λ_QoS = 30 and
  τ = 0.9. Q is constructed from the twin's continuous `delay_us` and
  `predicted_loss` surrogate outputs, using the same headroom-decay
  shape the paper uses for its 5QI=9 scoring.
- **Asymmetric switching cost** `c_DPDK = 0.03` charged when the
  realised action switches *into* DPDK; `c_USR = 0.012` when it
  switches into USR. These are the paper's measured-startup values
  (DPDK takes ~24 s, USR ~3.3 s).
- **Soft cooldown surcharge** (our extension over the paper's hard
  guard) — switching within 4 steps of the previous switch pays up
  to `0.5` extra reward, scaled by how deep into the window we are.

The observation is the paper's history-augmented schema, adapted:
current load + W = 8 past actual loads + 1-step forecast + previous
realised action + previous Q score + previous SEC + cooldown progress
(14-dim Box). Best-model checkpointing via SB3's `EvalCallback`
guarantees the deployed policy is the highest-eval-return seen during
training, not the final-step weights — the paper documents a late
collapse pattern that bites this codebase too.

Training: 200 000 timesteps on **`split="train"`** (the 5073-step
chronological training slice), with `EvalCallback` running on
**`split="val"`** to pick the best deterministic-rollout checkpoint.
PPO with paper hyperparameters (`n_steps = 1024`,
`learning_rate = 1e-4`, `ent_coef = 0.15`, `γ = 0.995`,
`gae_λ = 0.9`). ~20 min wall on this box (slower than before because
each train-time episode is ~5× longer).

## Result — total reward per policy (test split, cluster 0)

![Total reward bar chart](figures/fig2_total_reward.png)

| Policy | Total reward | Energy Wh | Mean SEC | Unsafe % | USR % | Flips |
|---|---|---|---|---|---|---|
| **PPO (best-of-val checkpoint)** | **−892.05** | **174.28** | **0.00771** | **0.50 %** | 30 % | 45 |
| Always DPDK | −1321.36 | 206.76 | 0.01310 | 0.00 % | 0 % | 0 |
| Threshold (USR < 0.05 Gbps) | −1688.38 | 194.69 | 0.01020 | 2.87 % | 21 % | 109 |
| Random | −4928.62 | 219.09 | 0.01091 | 16.35 % | 51 % | 511 |
| Always USR | −5640.57 | 230.77 | 0.00852 | 21.70 % | 100 % | 0 |

PPO is the only policy that uses USR meaningfully (30 % of steps)
while keeping the unsafe rate within an order of magnitude of the
always-DPDK floor. Compared to the in-sample iteration this report
used previously, the held-out test number is ~11 % worse on total
reward (−892 vs −803) and adds a small unsafe rate (0.5 % vs 0 %);
PPO also flips a third less often (45 vs 81) — a sign the val-selected
checkpoint commits to longer dwells. The threshold rule's
contribution is the same in both regimes because it is deterministic
and the test slice is unchanged. The paper's published threshold
operates with a dead-band + cooldown guard inside a 3-class
configuration space that isn't a fair point for a binary action
without those guards; for this phase the meaningful comparison is
PPO vs always-DPDK.

## Cluster 0 traffic profile

![Cluster 0 load](figures/fig1_load_profile.png)

Bursty around mean 0.108 Gbps, peaks to 0.59 Gbps over the test slice.
The training slice has mean 0.097 Gbps and the validation slice 0.077
Gbps with a higher peak (1.11 Gbps). Val is intentionally the harder
slice for QoS — a useful check on PPO's USR-bias under peaks, which
is why we use it for best-checkpoint selection rather than test.

## How PPO behaves

![Action timelines](figures/fig3_action_timelines.png)

Top panel is the load; the other rows are per-policy action choices.
PPO selects USR for sustained low-load stretches and DPDK during
load peaks, matching the qualitative description in the paper
("replaces disruptive mode changes with lower-cost decisions").
The threshold rule's pattern is reactive — it flips on every load
dip past its cutoff. PPO uses the history window in its observation
to commit to longer dwells where the data supports it.

## Cumulative reward over the episode

![Cumulative reward](figures/fig4_cumulative_reward.png)

PPO's curve stays cleanly above always-DPDK from ~step 50 onward and
diverges progressively. Random and always-USR fall off catastrophically
once the load enters the QoS-fragile regime for USR (above ~0.45 Gbps),
which is what their λ_QoS-weighted penalty pays for.

## How to reproduce

Train from scratch (~20 min wall — defaults to `train_split="train"`,
`eval_split="val"`, never touches `test`):

```bash
python scripts/train_ppo_single_site.py \
  --total-timesteps 200000 --n-steps 1024 \
  --ent-coef 0.15 --learning-rate 1e-4
```

One-shot held-out test evaluation, the source of the headline numbers:

```bash
python scripts/evaluate_test_split.py
# writes reports/phase-2/test_split_summary.json
```

Static figures (use `--split test` for the report, `--split val`
to debug the checkpoint-selection criterion):

```bash
python scripts/generate_phase2_report_figures.py --split test
```

Interactive exploration — the FastAPI + React dashboard:

```bash
# terminal 1
uvicorn dashboard.backend.app.main:app --reload --port 8000

# terminal 2
cd dashboard/frontend && npm run dev
```

Then open <http://localhost:5173>. The dashboard now charts
`q_score`, `sec_w_per_mbps`, and `steps_since_switch` in addition to
the original load/power/delay/loss/cumulative-reward panes.

## Caveats and next steps

- **Q score is constructed from `delay_us` and `predicted_loss`
  only** — the new twin does not predict jitter. The paper uses a
  weighted (0.4, 0.4, 0.2) sum of delay/jitter/loss scores; we use
  `min(delay_score, loss_score)` with the same headroom-decay shape.
  Equivalent in regime but slightly more conservative.
- **Surrogate extrapolation at load ≈ 0** — the USR layer-1 regressor
  predicts a delay of ~469 μs at load = 0 (artefact of sparse
  training data at the edge). PPO learns to avoid the corner;
  re-training USR upstream with more low-load samples would clean it
  up.
- **Validation slice contains very-low-load steps that explode SEC
  for always-DPDK.** Mean SEC on val for DPDK is ~48 W/Mbps vs ~0.013
  on test — driven by a handful of near-zero-load points where the
  denominator `max(load_Mbps, 1e-3)` saturates. PPO sidesteps this by
  switching to USR. The pattern doesn't appear in test, so it doesn't
  affect headline numbers, but the val-set absolute rewards are not
  comparable to test.
- **Single cluster.** Cluster 0 is favourable; cluster 1 has 2.6× the
  mean load and 2.2× the peak. The natural next step is K-cluster
  training (run the same script across `--cluster-idx 0..9`), then
  Phase 3 — multi-site centralised PPO over an aggregated env.
- **Late-stage policy collapse exists.** `EvalCallback` against val
  selects the best deployment automatically, so a final-step collapse
  during training does not corrupt the deployed policy — but the
  underlying training stability deserves a longer investigation
  later.

The pipeline is now in the shape the paper describes, with an
honest chronological train/val/test split sourced from the upstream
forecaster. Phase 3 will extend the env to multiple clusters in
parallel; the reward formulation, observation construction,
checkpoint-selection criterion, and dashboard layering are all
designed to scale to that step without refactor.
