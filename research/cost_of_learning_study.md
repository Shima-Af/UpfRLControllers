# Cost of Learning vs. Hysteresis — Is RL Worth It for UPF Control?

*Future work / personal project (after graduation). Seeded 2026-06-06.*

## The core question

The MASCOTS paper shows MAPPO beats an optimized hysteresis baseline by ~10% in
reward and ~5% in fleet energy. But a learning controller is **not free**: it
costs compute, memory, engineering effort, and operational overhead to train,
deploy, and maintain. A hysteresis controller is ~free to run and trivial to
deploy.

> **Once you account for the full lifecycle cost of the learning-based
> controller, is the energy/QoS gain actually worth shifting away from a simple
> hysteresis rule in real 5G-core deployments?**

This flips the paper's lens from *"RL wins on reward"* to *"does RL win on **net**
benefit, including its own footprint?"* — i.e. **does the intelligence pay back
its own cost?**

## What to measure

### A. The controllers' own footprint (MAPPO, IPPO, centralized PPO)
- **Training cost**: wall-clock, CPU/GPU-hours, and **energy (Wh)** to train to
  convergence — including seeds and any hyperparameter search.
- **Inference cost**: per-decision latency (ms), CPU cycles, RAM, and energy per
  inference, at the 15-min cadence and scaled across `K` sites.
- **Memory footprint**: params, on-disk MB, runtime RAM.
- **Retraining / maintenance**: how often must it be retrained as traffic
  drifts? cost of drift detection + periodic retraining.
- **Implementation complexity**: LoC, dependencies, ML stack, MLOps pipeline,
  expertise required, need for a safety fallback (revert to hysteresis).
- **Deployment cost**: where it runs (MANO layer), integration effort, failure
  modes.

### B. The baseline (hysteresis / threshold / always-DPDK)
- ~Zero training, negligible inference, ~10 LoC, no ML stack, no retraining.
  Quantify anyway for a fair table.

### C. The benefit side (already have from the paper)
- Energy saved vs DPDK (~5%), QoS, switching → translate to Wh/day, €/year, CO₂.

## The decisive comparison

```
net_benefit = (energy saved by better control)
            − (energy + € + ops cost of the controller itself),
              amortized over deployment lifetime and fleet size
```

- **Break-even analysis**: at what fleet size `K` / traffic scale does RL's gain
  outweigh its overhead (one-off training + per-site inference + retraining)?
- **Sensitivity**: if the gain is only ~5% energy, the controller's own
  energy + engineering/ops cost may dominate for small/medium fleets.
- **Carbon payback**: `kWh_to_train ÷ kWh_saved_per_day` = days to repay the
  training carbon. A clean sustainability headline.

## Data points we already have (seed the study)

- **Inference is NOT the differentiator.** MAPPO actor is `14→64→64→2` = **5,250
  params** (critic `140→128→128→10` = 35,850, training-only). A ~5k-param MLP is
  sub-millisecond on CPU; both MAPPO and hysteresis are effectively free at
  inference at a 15-min cadence.
- **Training cost IS the differentiator — and is wildly hardware-sensitive:**
  - MAPPO (custom PyTorch, batched, GPU): ~7 min/seed.
  - IPPO (SB3, 10 tiny PPOs/seed): on a **shared GPU** ~3 h/cluster (32 tiny jobs
    queued on one device → catastrophic contention); on **CPU** (32 cores,
    `OMP_NUM_THREADS=1`, GPU hidden) ~minutes. **Lesson: for many tiny models,
    CPU parallelism ≫ one shared GPU.** (See the 2026-06-06 GPU-contention
    incident — a concrete cautionary tale for the "deployment cost" section.)
  - Hysteresis: zero training, ~zero inference, ~10 LoC.

## Experiments to run

1. **Instrument training**: log wall-clock + device-hours and measure energy
   with **RAPL/Scaphandre — the same tooling already used for the UPF twin** — for
   each controller's full training (incl. retraining).
2. **Instrument inference**: micro-benchmark per-decision latency + energy,
   MAPPO vs hysteresis, across `K`.
3. **Lifecycle / TCO model**: notebook combining one-off training + retrain
   cadence + per-site inference vs energy saved → find break-even fleet size.
4. **Drift study**: how fast does a trained policy degrade as the traffic
   distribution shifts → retraining frequency → ongoing cost.
5. **Carbon-payback metric**: training kWh ÷ daily kWh saved.

## Why it's interesting / publishable

- The energy-efficiency field rarely accounts for the **energy/compute cost of
  the AI doing the optimizing**. A rigorous "is the intelligence worth its own
  footprint?" study for 5G-core control is novel and timely (green networking /
  sustainable AI).
- Natural sequel to the MASCOTS paper: **same twin, same controllers, flipped
  lens** — total cost instead of just reward.

## Notes

- Reuse the existing **Scaphandre/RAPL** measurement setup to measure the
  *controllers'* energy too — symmetric and clean.
- Keep hysteresis as the "free" reference. The question is always: *what does the
  extra cleverness cost, and does it pay back?*
- Related: the MASCOTS paper (`reports/paper-mascots/`).
