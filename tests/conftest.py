"""Pytest fixtures.

The env tests need the digital-twin artifacts under ``data/external/``.
On a fresh checkout those may not be present yet — in that case the
tests should *skip*, not fail. ``has_twin_artifacts`` lets test modules
gate themselves uniformly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def has_twin_artifacts(repo_root: Path) -> bool:
    """True if the artifacts needed to construct SingleSiteUPFEnv are present."""
    ext = repo_root / "data" / "external"
    required = [
        ext / "traffic_forecaster" / "cluster_series.npy",
        ext / "profiling_twin" / "models" / "manifest.json",
    ]
    return all(p.exists() for p in required)
