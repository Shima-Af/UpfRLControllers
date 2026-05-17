"""FastAPI app for the UPF RL dashboard.

Endpoints:
  GET  /policies                — list available policies
  GET  /clusters                — list cluster_idx values with load stats
  POST /rollout                 — run one policy on one cluster, return trajectory
  POST /compare                 — run multiple policies, return aligned trajectories

Run locally:
  cd dashboard/backend
  uvicorn app.main:app --reload --port 8000

CORS is wide open to ``http://localhost:5173`` (Vite dev server). Tighten
this if you ever deploy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Make src.* importable regardless of where uvicorn is launched from.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .multi_policies import MultiPolicyRegistry, run_multi_rollout  # noqa: E402
from .policies import PolicyRegistry, run_rollout  # noqa: E402
from .schemas import (  # noqa: E402
    ClusterInfo,
    CompareRequest,
    CompareResponse,
    MultiClusterRolloutResponse,
    MultiCompareRequest,
    MultiCompareResponse,
    MultiPolicyInfo,
    MultiRolloutRequest,
    PolicyInfo,
    RolloutRequest,
    RolloutResponse,
)

app = FastAPI(
    title="UPF RL Controllers Dashboard",
    description="Interactive episode replay + policy comparison for the single-site UPF digital twin.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = PolicyRegistry(repo_root=REPO_ROOT)
multi_registry = MultiPolicyRegistry(repo_root=REPO_ROOT)


# Cached load-statistics computation. Built lazily once and reused.
_CLUSTER_STATS_CACHE: list[ClusterInfo] | None = None


def _cluster_stats() -> list[ClusterInfo]:
    global _CLUSTER_STATS_CACHE
    if _CLUSTER_STATS_CACHE is not None:
        return _CLUSTER_STATS_CACHE

    targets_path = (
        REPO_ROOT / "data" / "external" / "traffic_forecaster" / "targets_test.npy"
    )
    if not targets_path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Traffic targets not found at {targets_path}. "
                "Run `dvc pull` and `python scripts/bootstrap_external_data.py` first."
            ),
        )
    arr = np.load(targets_path)  # (N, H, K)
    n, horizon_count, k = arr.shape
    out: list[ClusterInfo] = []
    for cidx in range(k):
        series = arr[:, 0, cidx]  # horizon 0 is the canonical default
        out.append(
            ClusterInfo(
                cluster_idx=cidx,
                episode_length=int(n),
                horizon_count=int(horizon_count),
                load_mean_gbps=float(series.mean()),
                load_p95_gbps=float(np.percentile(series, 95)),
                load_max_gbps=float(series.max()),
            )
        )
    _CLUSTER_STATS_CACHE = out
    return out


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/policies", response_model=list[PolicyInfo])
def list_policies() -> list[PolicyInfo]:
    return registry.list_policies()


@app.get("/clusters", response_model=list[ClusterInfo])
def list_clusters() -> list[ClusterInfo]:
    return _cluster_stats()


@app.post("/rollout", response_model=RolloutResponse)
def post_rollout(req: RolloutRequest) -> RolloutResponse:
    try:
        return run_rollout(
            registry,
            policy_id=req.policy,
            cluster_idx=req.cluster_idx,
            horizon_idx=req.horizon_idx,
            seed=req.seed,
            threshold_gbps=req.threshold_gbps,
            hysteresis_band_mbps=req.hysteresis_band_mbps,
            hysteresis_cooldown_steps=req.hysteresis_cooldown_steps,
            max_steps=req.max_steps,
            split=req.split,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/compare", response_model=CompareResponse)
def post_compare(req: CompareRequest) -> CompareResponse:
    if not req.policies:
        raise HTTPException(status_code=400, detail="Must request at least one policy.")
    rollouts: list[RolloutResponse] = []
    for pid in req.policies:
        try:
            rollouts.append(
                run_rollout(
                    registry,
                    policy_id=pid,
                    cluster_idx=req.cluster_idx,
                    horizon_idx=req.horizon_idx,
                    seed=req.seed,
                    threshold_gbps=req.threshold_gbps,
                    hysteresis_band_mbps=req.hysteresis_band_mbps,
                    hysteresis_cooldown_steps=req.hysteresis_cooldown_steps,
                    max_steps=req.max_steps,
                    split=req.split,
                )
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    return CompareResponse(
        cluster_idx=req.cluster_idx,
        horizon_idx=req.horizon_idx,
        rollouts=rollouts,
    )


# ---------------------------------------------------------------------------
# Multi-cluster (Phase 7) endpoints
# ---------------------------------------------------------------------------


@app.get("/multi/policies", response_model=list[MultiPolicyInfo])
def list_multi_policies() -> list[MultiPolicyInfo]:
    return multi_registry.list_policies()


@app.get("/multi/derived")
def get_derived_thresholds() -> dict:
    """Expose the twin-derived threshold operating points (Mbps/Gbps)."""
    try:
        return multi_registry.derived_thresholds()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/multi/rollout", response_model=MultiClusterRolloutResponse)
def post_multi_rollout(req: MultiRolloutRequest) -> MultiClusterRolloutResponse:
    try:
        return run_multi_rollout(
            multi_registry,
            policy_id=req.policy,
            horizon_idx=req.horizon_idx,
            split=req.split,
            seed=req.seed,
            max_steps=req.max_steps,
            threshold_gbps=req.threshold_gbps,
            hysteresis_band_mbps=req.hysteresis_band_mbps,
            hysteresis_cooldown_steps=req.hysteresis_cooldown_steps,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/multi/compare", response_model=MultiCompareResponse)
def post_multi_compare(req: MultiCompareRequest) -> MultiCompareResponse:
    if not req.policies:
        raise HTTPException(
            status_code=400, detail="Must request at least one policy."
        )
    rollouts: list[MultiClusterRolloutResponse] = []
    K: int | None = None
    for pid in req.policies:
        try:
            r = run_multi_rollout(
                multi_registry,
                policy_id=pid,
                horizon_idx=req.horizon_idx,
                split=req.split,
                seed=req.seed,
                max_steps=req.max_steps,
                threshold_gbps=req.threshold_gbps,
                hysteresis_band_mbps=req.hysteresis_band_mbps,
                hysteresis_cooldown_steps=req.hysteresis_cooldown_steps,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        rollouts.append(r)
        K = r.summary.K
    return MultiCompareResponse(
        horizon_idx=req.horizon_idx,
        split=req.split,
        K=K or 10,
        rollouts=rollouts,
    )
