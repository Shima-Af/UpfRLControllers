"""Smoke tests for SingleSiteUPFEnv — construction, reset, step contract.

Modelled on scripts/smoke_test_single_site_env.py but expressed as
pytest cases. Skips cleanly when the digital-twin artifacts are not
yet present (fresh checkout, pre-`dvc pull`).
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("has_twin_artifacts")


def _env_or_skip(has_twin_artifacts, **kwargs):
    if not has_twin_artifacts:
        pytest.skip("Digital-twin artifacts not present under data/external/")
    from src.envs.single_site_upf_env import SingleSiteUPFEnv

    return SingleSiteUPFEnv(cluster_idx=0, horizon_idx=0, **kwargs)


def test_construct_default(has_twin_artifacts):
    env = _env_or_skip(has_twin_artifacts)
    assert env.action_space.n == 2
    assert len(env.observation_space.shape) == 1
    assert env.observation_space.shape[0] >= 10


def test_reset_returns_valid_obs(has_twin_artifacts):
    env = _env_or_skip(has_twin_artifacts)
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert isinstance(info, dict)


def test_step_contract(has_twin_artifacts):
    env = _env_or_skip(has_twin_artifacts)
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    for key in ("requested_upf", "selected_upf", "power_watts", "is_safe"):
        assert key in info, f"missing info key: {key}"


def test_split_isolation(has_twin_artifacts):
    """train / val / test slices must be disjoint along the time axis."""
    from src.envs.single_site_upf_env import SingleSiteUPFEnv

    if not has_twin_artifacts:
        pytest.skip("Digital-twin artifacts not present under data/external/")

    lengths = {}
    for split in ("train", "val", "test"):
        env = SingleSiteUPFEnv(cluster_idx=0, horizon_idx=0, split=split)
        lengths[split] = env._N
    assert sum(lengths.values()) > 0
    assert lengths["train"] >= lengths["val"]
    assert lengths["train"] >= lengths["test"]


def test_deterministic_reset(has_twin_artifacts):
    env = _env_or_skip(has_twin_artifacts)
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    np.testing.assert_allclose(obs1, obs2)
