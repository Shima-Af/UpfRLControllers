# UPF RL Controllers Dashboard

Interactive episode replay + policy comparison for the single-site UPF
digital twin. v1 covers Phase 2: compare any subset of {PPO, threshold,
always-DPDK, always-USR, random} on any of the 10 clusters, see KPIs
and time-series side by side.

## Architecture

```
dashboard/
  backend/                 FastAPI app
    app/
      main.py              endpoint definitions
      policies.py          policy registry + rollout helper
      schemas.py           Pydantic models (shared API contract)
    requirements.txt
  frontend/                Vite + React + TypeScript
    src/
      api/                 axios client + TS mirrors of the Pydantic models
      components/          Controls, SummaryTable, TimeSeriesChart
      App.tsx              page composition
```

The backend imports `src.envs.single_site_upf_env` and
`src.trainers.ppo_single_site` directly — there is no separate service
layer, so dashboard rollouts use the same code paths as
`scripts/train_ppo_single_site.py`.

## v1 endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness probe |
| GET | `/policies` | list policies + whether PPO has a checkpoint loaded |
| GET | `/clusters` | cluster_idx 0..9 + load statistics |
| POST | `/rollout` | one policy → trajectory + summary |
| POST | `/compare` | N policies → N aligned trajectories + summaries |

Request / response shapes live in
[`backend/app/schemas.py`](backend/app/schemas.py) — Pydantic v2 on the
server side, mirrored 1:1 in [`frontend/src/api/types.ts`](frontend/src/api/types.ts).

## Run locally

Two terminals, from the repo root.

**Terminal 1 — backend:**

```bash
# First time only:
pip install -r dashboard/backend/requirements.txt

# Run:
uvicorn dashboard.backend.app.main:app --reload --port 8000
```

The backend will use the most-recently-modified PPO checkpoint under
`experiments/ppo_single_site_*/ppo_single_site.zip`. If no checkpoint
exists, the "PPO" policy is greyed out in the UI; train one first with
`scripts/train_ppo_single_site.py`.

**Terminal 2 — frontend:**

```bash
cd dashboard/frontend
# First time only:
npm install

# Run:
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api/*` to the backend on
port 8000 (see [`frontend/vite.config.ts`](frontend/vite.config.ts)), so
no CORS headache during development.

## What the UI shows

Top bar: cluster + forecast horizon + max-step cap + threshold-rule
threshold + multi-select pill row of policies. The PPO pill is
disabled when no checkpoint is loaded.

Body, after pressing **Run comparison**:

- **Summary table** — one row per policy (total reward, energy Wh,
  switch Wh, unsafe %, DPDK/USR share, flip count, steps), best row
  highlighted.
- **Seven time-series charts**, one line per policy:
  - offered load (Gbps)
  - power consumed (W)
  - predicted delay (μs) with 200 μs budget line
  - predicted packet loss (pkts/interval) with 5 pkts budget line
  - performance score with 0.9 threshold line
  - cumulative reward
  - action timeline (0 = DPDK, 1 = USR)

## Extending for future phases

The schema → service → endpoint → frontend-hook → component layering
isolates each addition. To add a new endpoint:

1. Add request/response models to `backend/app/schemas.py`.
2. Add a handler to `backend/app/main.py` (or a new router module).
3. Mirror the types in `frontend/src/api/types.ts` and the call in
   `frontend/src/api/client.ts`.
4. Build a component that calls it with `useQuery` / `useMutation`.

Planned next steps (out of v1 scope):

- **v2** — `POST /train` background task + `GET /train/{id}/events`
  Server-Sent Events stream for live training curves.
- **v3** — Model registry view backed by `experiments/`,
  hyperparameter editor, downloadable rollouts (CSV/Parquet).
