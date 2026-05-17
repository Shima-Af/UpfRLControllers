# UpfRLControllers

RL controller implementations for energy-aware UPF (User Plane Function)
orchestration. This repository depends on the separate
[UpfDigitalTwin](https://github.com/Shima-Af/UpfDigitalTwin) repository for
measurement-grounded UPF evaluation.

## Repository separation

The two repositories have intentionally distinct responsibilities:

- **UpfDigitalTwin** — the digital twin / evaluator. Owns traffic
  forecasting, profiling models, switching-cost estimation, and all
  measurement-grounded simulation of the UPF. It is controller-agnostic and
  must stay that way.
- **UpfRLControllers** (this repo) — RL environments, reward functions,
  trainers, evaluation scripts, and baseline controllers. It consumes the
  digital twin as an external dependency; it never modifies it.

Artifacts produced by the digital twin (forecasts, cluster assignments,
profiling models, switching-cost tables) are pulled from S3 into
[data/external/](data/external/). They are not committed to git and are not
vendored from the digital-twin repo.

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Shima-Af/UpfRLControllers.git
   cd UpfRLControllers
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install requirements**

   ```bash
   pip install -r requirements.txt
   ```

   This installs `UpfDigitalTwin` directly from GitHub. There are no git
   submodules and no vendored copies of the digital-twin source.

4. **Populate [data/external/](data/external/) with digital-twin artifacts**

   This is a two-step pull because not every required file lives in S3.

   **(a) DVC-tracked files** — pulled from two S3 remotes configured in
   [.dvc/config](.dvc/config) (`forecaster` and `profiling`). AWS
   credentials with read access to both buckets must be in your
   environment.

   ```bash
   dvc pull
   ```

   This resolves the `.dvc` pointer files under [data/external/](data/external/) and
   downloads:
   - `traffic_forecaster/bs_locations.parquet`
   - `traffic_forecaster/cluster_series.npy`
   - `traffic_forecaster/cluster_assignments.parquet`
   - `traffic_forecaster/cluster_bs_map.json`
   - `profiling_twin/models/` (entire surrogate-model tree + `manifest.json`)

   **(b) Non-DVC files** — the train/val/test prediction and target
   arrays (`predictions_{train,val,test}.npy`, `targets_{train,val,test}.npy`,
   `forecast_eval_summary.json`) plus hand-authored files
   (`switching_costs.yaml`, `params.yaml`). These are copied from the
   upstream forecaster repo output or a peer directory. The default source
   is `/home/ubuntu/UPF_NDT/data/external`; override with `--source`.

   ```bash
   python scripts/bootstrap_external_data.py
   ```

   See [scripts/bootstrap_external_data.py](scripts/bootstrap_external_data.py)
   for the exact file list and the rationale. The script is idempotent —
   existing files are left alone unless `--force` is passed.

5. **Run the setup check**

   ```bash
   python scripts/check_setup.py
   ```

   This loads both config files, verifies that the artifact directories
   exist, and tries to import the digital-twin dependency. Missing
   artifacts produce warnings, not failures, so it is safe to run on a
   fresh checkout.

## Repository layout

```
UpfRLControllers/
  configs/                  YAML configs (digital-twin paths, RL scenario)
  data/external/            Digital-twin artifacts pulled from S3 (gitignored)
  src/
    envs/                   Gymnasium / PettingZoo environments
    rewards/                Reward function modules
    trainers/               PPO / MAPPO trainers
    evaluation/             Evaluation utilities and metrics
    baselines/              Static, threshold, hysteresis controllers
    utils/                  Helpers (config loading, etc.)
  scripts/                  Entry-point scripts (setup checks, training)
  experiments/              Per-experiment outputs (gitignored)
  results/                  Aggregated results and figures (gitignored)
  notebooks/                Exploratory notebooks
```

## Phase 1 — Single-site Gymnasium environment ✓

[src/envs/single_site_upf_env.py](src/envs/single_site_upf_env.py) wraps
one traffic cluster at a fixed forecast horizon.

- **Action space**: `Discrete(2)` — `0 = DPDK`, `1 = USR`.
- **Observation**: `Box((14,))` — current load + 8 history loads (oldest
  first) + 1-step forecast + previous action + previous Q score + previous
  SEC + cooldown progress. Aligned with paper Section 4.3.
- **Reward**: `-(α·SEC + λ_QoS·max(0,τ−Q) + switch_cost + cooldown_cost)`.
  See [configs/scenario_rl.yaml](configs/scenario_rl.yaml) for all weights.

```python
from src.envs.single_site_upf_env import SingleSiteUPFEnv

env = SingleSiteUPFEnv(cluster_idx=0, horizon_idx=0, split="train")
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(0)  # 0=DPDK, 1=USR
```

`split=` accepts `"train"` / `"val"` / `"test"` — selects which forecaster
slice the episode draws from. Per-step surrogate calls are pre-cached at
`__init__` (≈1.3 s) so `step()` runs in ≈10 μs.

## Phase 2 — Single-site PPO (paper-aligned, out-of-sample) ✓

[src/trainers/ppo_single_site.py](src/trainers/ppo_single_site.py) trains
a feed-forward PPO on `SingleSiteUPFEnv`. Training uses `split="train"`
(5073 steps, ~53 days); `EvalCallback` selects the best checkpoint on
`split="val"`; the headline number is evaluated once on `split="test"` by
[scripts/evaluate_test_split.py](scripts/evaluate_test_split.py).

**Test-split result on cluster 0** (200k steps, ~20 min wall):

| Policy | Total reward | Energy Wh | Unsafe % | USR % | Flips |
|---|---|---|---|---|---|
| **Phase 2 PPO** | **−892.05** | **174.28** | 0.50 % | 30 % | 45 |
| Always DPDK | −1321.36 | 206.76 | 0.00 % | 0 % | 0 |
| Threshold (USR < 0.05 Gbps) | −1688.38 | 194.69 | 2.87 % | 21 % | 109 |

PPO beats always-DPDK by 32.5% on total reward and 15.7% on energy.
Full writeup with figures: [reports/phase-2/README.md](reports/phase-2/README.md).

Train from scratch:

```bash
python scripts/train_ppo_single_site.py \
  --total-timesteps 200000 --n-steps 1024 --ent-coef 0.15
```

### Phase 2 interactive dashboard

FastAPI + React dashboard — replay episodes, compare policies, explore
per-step KPIs in the browser:

```bash
uvicorn dashboard.backend.app.main:app --reload --port 8000   # terminal 1
cd dashboard/frontend && npm run dev                          # terminal 2
```

Then open <http://localhost:5173>.

## Phase 3 — Multi-site PPO (centralised vs per-cluster ensemble) ✓

[src/envs/multi_site_upf_env.py](src/envs/multi_site_upf_env.py) runs all
K=10 clusters in parallel. Action: `MultiDiscrete([2]*10)`. Observation:
140-dim concatenation of 10 × Phase-1 obs vectors. Reward: per-step
load-weighted sum `Σ_k w_k(t)·r_k(t)`.

Two approaches were evaluated:

**Phase 3a — Centralised PPO** — one network, joint action.
Training: 200k steps on `split="train"`, best-of-val checkpoint.

**Phase 3b — Per-cluster ensemble** — 10 separate Phase-2 PPOs stacked
at evaluation time. Training: 10 × 200k steps (4-way parallel, ~50 min
wall via `scripts/train_ppo_ensemble.py`).

**Test-split result (K=10 clusters, load-weighted reward):**

| Policy | Weighted reward | Energy Wh | Unsafe % | USR % |
|---|---|---|---|---|
| **Phase 3b ensemble** | **−667.50** | **1973.19** | 0.59 % | 8.1 % |
| Always DPDK | −643.74 | 2070.72 | 0.38 % | 0.0 % |
| Threshold | −680.10 | 1984.53 | 0.62 % | 7.7 % |
| **Phase 3a centralised** | **−1072.45** | 2139.87 | **4.51 %** | 14.8 % |

The centralised PPO collapsed to always-USR on cluster 0 (21.7% unsafe
there) — the joint 140-dim/10-D-action policy is too large for 200k steps
and Phase-2 hyperparameters. The ensemble is the deployable deliverable.
Full writeup with per-cluster breakdown and action heatmap:
[reports/phase-3/README.md](reports/phase-3/README.md).

```bash
# Train Phase 3a (centralised)
python scripts/train_ppo_multi_site.py --total-timesteps 200000

# Train Phase 3b (ensemble, 4-way parallel)
python scripts/train_ppo_ensemble.py --total-timesteps 200000 --n-parallel 4

# Evaluate both on test split
python scripts/evaluate_multi_site_test.py \
  --ensemble-dir experiments/ppo_single_site_ensemble_<ts>
```

## Development roadmap

| Phase | Description | Status |
|---|---|---|
| Phase 0 | Repository setup and artifact loading | ✓ done |
| Phase 1 | Single-site Gymnasium environment | ✓ done |
| Phase 2 | Single-site PPO (paper-aligned, train/val/test split) | ✓ done |
| Phase 3a | Centralised multi-site PPO | ✓ done (negative result) |
| Phase 3b | Per-cluster ensemble (deployable deliverable) | ✓ done |
| Phase 4 | Switching-cost physics + cooldown sensitivity sweep | pending |
| Phase 5 | PettingZoo-style multi-agent environment | pending |
| Phase 6 | MAPPO / CTDE | pending |
| Phase 7 | Final comparison across all controllers | pending |
