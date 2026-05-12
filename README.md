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

4. **Pull artifacts from S3 into [data/external/](data/external/)**

   ```bash
   aws s3 sync s3://YOUR_BUCKET/YOUR_PREFIX/ data/external/
   ```

   Replace `YOUR_BUCKET/YOUR_PREFIX` with the bucket and prefix that hold
   the digital-twin outputs. The expected layout under `data/external/` is
   described in [configs/digital_twin_paths.yaml](configs/digital_twin_paths.yaml).

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

## Phase 1: single-site environment

The first concrete piece of RL infrastructure is [src/envs/single_site_upf_env.py](src/envs/single_site_upf_env.py),
which exposes a Gymnasium `Env` for one traffic cluster at a fixed forecast
horizon.

- **Action space**: `Discrete(2)` — `0 = DPDK`, `1 = USR`.
- **Observation**: `Box((6,))` =
  `[actual_load_gbps, predicted_load_gbps, prev_action, prev_power_watts,
  prev_safety_flag, timestep_progress]`. The predicted load is the
  forecaster's estimate for the current decision step; no future values are
  exposed.
- **Reward**: `-(energy_weight * energy_wh + qos_weight * unsafe_penalty +
  switching_weight * switching_energy_wh)`, with weights from
  [configs/scenario_rl.yaml](configs/scenario_rl.yaml). Switching energy
  comes from `DigitalTwin.compute_step`.

Minimal usage:

```python
from src.envs.single_site_upf_env import SingleSiteUPFEnv

env = SingleSiteUPFEnv(cluster_idx=0, horizon_idx=0)
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(0)  # 0=DPDK, 1=USR
```

To smoke-test the env without writing any training code:

```bash
python scripts/smoke_test_single_site_env.py
```

This resets the env, takes 10 random actions, and prints observation shape,
per-step rewards, and the `info` dict keys. It exits cleanly with a warning
if the digital-twin artifacts are missing.

PPO / MAPPO are intentionally not part of Phase 1 — they live in Phase 2
and beyond.

## Development roadmap

- **Phase 0** — Repository setup and artifact loading.
- **Phase 1** — Single-site Gymnasium environment wrapping the digital
  twin (this section). **This is the first technical target after Phase 0**
  — do not jump straight to MAPPO.
- **Phase 2** — Single-site PPO sanity check on the Phase 1 environment.
- **Phase 3** — Multi-site centralized PPO.
- **Phase 4** — Independent per-site PPO baseline (IPPO).
- **Phase 5** — PettingZoo-style multi-agent environment.
- **Phase 6** — MAPPO / CTDE on the Phase 5 environment.
- **Phase 7** — Final comparison across static, threshold, hysteresis,
  independent PPO, centralized PPO, and MAPPO controllers.

The first implementation must not start directly with MAPPO. The Phase 1
single-site Gymnasium environment is the prerequisite for everything that
follows.
