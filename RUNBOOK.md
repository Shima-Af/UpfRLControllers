# Runbook

How to set up, smoke-test, demo, train, and reproduce the paper numbers.
All commands run from the repo root with `.venv` activated.

## 1. First-time setup (only once)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # installs library + UpfDigitalTwin from git
dvc pull                                  # forecaster + profiling artifacts from S3 (needs AWS creds)
python scripts/bootstrap_external_data.py # non-DVC artifacts (predictions/targets, switching_costs.yaml)
python scripts/check_setup.py             # sanity-check artifacts + imports
```

If `dvc pull` fails, AWS credentials for the S3 remotes are missing — talk
to whoever owns the forecaster bucket.

## 2. Smoke test — is the env alive?

```bash
pytest -q                                       # 9 tests, ~4 sec
python scripts/smoke_test_single_site_env.py    # one-shot env construction + 10 random steps
```

Both should print clean output. If tests skip or the smoke script warns,
the artifacts under `data/external/` are missing — go back to step 1.

## 3. Run the dashboard (easiest demo path)

Two terminals, both from the repo root:

```bash
# terminal 1 — FastAPI backend
uvicorn dashboard.backend.app.main:app --reload --port 8000

# terminal 2 — React frontend
cd dashboard/frontend && npm install && npm run dev
```

Open <http://localhost:5173>. Replay PPO / MAPPO / threshold / always-DPDK /
always-USR side by side. The backend loads checkpoints from `experiments/`
automatically.

## 4. Train from scratch (optional — checkpoints already exist under `experiments/`)

```bash
# Phase 2 — single-site PPO  (~20 min)
python scripts/train_ppo_single_site.py --total-timesteps 200000 --n-steps 1024 --ent-coef 0.15

# Phase 3a — centralised multi-site PPO  (~20 min)
python scripts/train_ppo_multi_site.py --total-timesteps 200000

# Phase 3b — per-cluster ensemble  (4-way parallel, ~50 min)
python scripts/train_ppo_ensemble.py --total-timesteps 200000 --n-parallel 4

# Phase 6 — MAPPO (CTDE)
python scripts/train_mappo.py --seed 42
```

Each writes into `experiments/<run-name>_seed<N>_<timestamp>/`.

## 5. Reproduce paper numbers + figures

```bash
# Phase 2 — single-site PPO headline table (cluster 0, test slice)
python research/phase2/evaluate_test_split.py
python research/phase2/generate_figures.py --split test

# Phase 3 — multi-site
python research/phase3/evaluate_test_split.py --ensemble-dir experiments/ppo_single_site_ensemble_<ts>
python research/phase3/generate_figures.py --split test

# Phase 6 — MAPPO
python research/phase6/evaluate_test_split.py
python research/phase6/generate_figures.py --split test

# Phase 7 — final multi-seed comparison (paper's headline numbers)
python research/phase7/evaluate_multiseed.py
python research/phase7/paired_bootstrap.py
python research/phase7/generate_figures.py
python research/phase7/compare_reward_revisions.py
```

Outputs land in `reports/phase-{2,3,6,7}/`.

## TL;DR — "just show me it works"

```bash
source .venv/bin/activate
pytest -q                                                       # library is healthy
uvicorn dashboard.backend.app.main:app --reload --port 8000 &   # backend
cd dashboard/frontend && npm run dev                            # frontend → :5173
```

That is the full visual demo path. To confirm the headline numbers in the
paper, run step 5's Phase-7 block — it reads the existing checkpoints in
`experiments/` and rewrites `reports/phase-7/`.

## Where things live

| Need | Location |
|---|---|
| Library code (envs, trainers, baselines) | [`src/`](src/) |
| Canonical CLIs (setup + trainers) | [`scripts/`](scripts/) |
| Per-phase one-off scripts (evaluators, figures, sweeps) | [`research/`](research/) — see [`research/README.md`](research/README.md) |
| Tests | [`tests/`](tests/) |
| Supervisor reports + paper | [`reports/`](reports/) |
| Trained checkpoints | [`experiments/`](experiments/) (gitignored) |
| Digital-twin artifacts | [`data/external/`](data/external/) (gitignored) |
| Dashboard | [`dashboard/`](dashboard/) |
