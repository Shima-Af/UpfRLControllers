# Phase 2 — Single-Site PPO Sanity Check

**Status:** complete on cluster 0. The trained PPO controller converges
to **always-DPDK** under realistic switching costs, matching the
optimal static baseline exactly (−206.76 reward) and beating the
threshold rule (−208.24) by a small margin. The result is correct but
flat for this single cluster — Phase 2 is closed; the natural next
step is to train one PPO per cluster (K=10) where load profiles with
sustained low-load windows are likely to favour an actively switching
policy.

## Summary

We trained a Stable-Baselines3 PPO controller on the single-site
Gymnasium environment built in Phase 1, which wraps the
[UpfDigitalTwin](https://github.com/Shima-Af/UpfDigitalTwin) surrogate
for one traffic cluster. At every 15-minute decision step the agent
picks DPDK or USR. The reward combines:

- **steady-state energy** (`power_watts × 0.25 h`, weighted at 1.0),
- **graded QoS penalty** from the twin's continuous delay/loss
  predictions (`5 × max(0, 0.90 − performance)`),
- **switching energy** from the twin's `compute_step` activation
  spike (weighted at 1.0 — the spike is real physics, not a knob),
- a **soft cooldown** that charges up to `0.5` extra reward when the
  agent switches inside a 4-step (1 h) cooldown window, scaled by how
  deep into the cooldown we still are.

The cooldown is the only purely-operational ingredient — everything
else is grounded in either the twin's profiling-derived physics or
the prior controller's QoS semantics. After 200 000 training steps
(~12 min wall, 98 PPO iterations) the trained policy is:

- 100 % DPDK, 0 switches, 0 unsafe steps.
- Identical total reward to the always-DPDK baseline (−206.76).
- Marginally better than the threshold rule (−208.24).

This is not a failure of training — it is the correct optimum for this
cluster under the calibrated cost regime. The threshold rule's 22 %
USR usage saves ~26 Wh of energy but pays ~14 Wh in QoS penalties
(from surrogate noise) and ~13 Wh in soft-cooldown costs, leaving a
net loss of ~1.5 reward against always-DPDK.

## Cluster 0 traffic profile

![Cluster 0 load](figures/fig1_load_profile.png)

Mean load 0.108 Gbps, p95 0.225 Gbps, peaks to 0.59 Gbps. About 21 %
of the steps sit below the 0.05 Gbps line where USR's surrogate
predicts safe operation. The load is bursty — there are no long
sustained low-load windows that would let a single USR activation pay
off across many steps.

## Result — total reward per policy

![Total reward bar chart](figures/fig2_total_reward.png)

| Policy | Total reward | Energy Wh | Unsafe % | USR share | Flips |
|---|---|---|---|---|---|
| **PPO (trained)** | **−206.76** | 206.76 | 0.0 % | 0 % | 0 |
| Always DPDK | −206.76 | 206.76 | 0.0 % | 0 % | 0 |
| Threshold (USR<0.05) | −208.24 | 180.22 | 0.4 % | 22 % | 61 |
| Random | −1020.72 | 219.09 | 16.4 % | 51 % | 511 |
| Always USR | −1027.55 | 230.77 | 21.7 % | 100 % | 0 |

PPO ties always-DPDK to the cent. The threshold rule's energy
advantage doesn't survive the cooldown surcharge. Random and always-USR
collapse — at this cluster's load profile the surrogate predicts
catastrophic packet loss for USR above ~0.45 Gbps, and the QoS penalty
in those regions dwarfs any energy savings.

## How PPO sees the policy space

![Action timelines](figures/fig3_action_timelines.png)

Top panel is the load. The next three rows are the actions chosen by
PPO, the threshold rule, and always-DPDK. Every row in the bottom
panel is DPDK — PPO learned that, given the cooldown surcharge and
surrogate noise, every USR window the rule exploits costs more than
the energy it saves. The threshold rule still selects USR in the
low-load dips but pays for it.

## Cumulative reward over the episode

![Cumulative reward](figures/fig4_cumulative_reward.png)

PPO's curve sits exactly on top of always-DPDK's. The threshold rule's
curve diverges briefly below the others — visible energy savings
during USR windows — but is pulled back up by cooldown penalties
between switches. Random and always-USR are the obvious losers.

## Engineering interpretation

Three concrete observations for the supervisor:

1. **The reward formulation is now physics-grounded.** Switching cost
   comes from the twin's measured activation spike (~5 mWh per DPDK
   switch, ~1 mWh per USR switch) rather than a hand-picked flat
   penalty. The only purely-operational dial is `cooldown_cost = 0.5`,
   and the report transparently shows what changing it would do.
2. **At cluster 0, always-DPDK is genuinely optimal.** With realistic
   switching costs, the brief low-load windows the threshold rule
   exploits don't pay back over a 1009-step episode. PPO is not
   underfit — it has converged correctly.
3. **The interesting clusters are elsewhere.** Some of the other 9
   clusters have higher peak loads (cluster 1 means 0.28 Gbps and
   peaks at 1.33 Gbps), but they likely also have nightly troughs
   below cluster 0's. A K-cluster training pass is where we expect
   PPO to discover sustained-USR-overnight policies that genuinely
   beat always-DPDK.

## How to reproduce

Static figures and the table above:

```bash
python scripts/generate_phase2_report_figures.py
```

Interactive exploration (recommended for a live supervisor demo) —
spin up the FastAPI + React dashboard:

```bash
# terminal 1
uvicorn dashboard.backend.app.main:app --reload --port 8000

# terminal 2
cd dashboard/frontend && npm run dev
```

Then open <http://localhost:5173>. The dashboard now also charts
`steps_since_switch` so you can see the cooldown counter for each
policy, and the time-series include the surrogate-predicted delay,
predicted loss, and the continuous performance score with their
respective budget lines.

## Caveats and next steps

- **Cooldown is the only knob without a physics grounding.** Setting
  `cooldown_cost` to zero would let PPO recover the policy from the
  pre-cooldown report (USR ~25 %, −173.89 reward); operationally that
  policy switches too often. The 0.5 value is a placeholder reflecting
  "switching is undesirable but not banned."
- **Surrogate extrapolation at load ≈ 0.** The USR layer-1 regressor
  predicts a delay of ~469 μs at load = 0 (artefact of sparse training
  data at the edge). PPO learns to avoid the corner; fixing it
  upstream by retraining USR with more low-load samples is on the
  to-do list.
- **Single cluster is a single data point.** Phase 2 establishes the
  pipeline end-to-end. The next experiment is K=10 per-cluster
  training, where load profiles with overnight troughs are the likely
  win regions for PPO over always-DPDK.

Phase 3 — multi-site centralized PPO — builds directly on this stack:
the env's batched-surrogate trick extends to N clusters in parallel,
and the dashboard's layering is set up for an additional multi-site
view without refactor.
