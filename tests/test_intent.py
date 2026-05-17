"""Unit tests for the intent schema and deterministic compilers (Slice 0).

No checkpoint loading, no env construction here — those live behind the
twin-artifacts gate in `conftest.py` and would slow the suite down.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intent.compile.plan_to_predicate import compile_predicate
from intent.compile.plan_to_weights import BASE, compile_weights
from intent.examples.canonical_plans import CANONICAL_PLANS
from intent.schema.metrics import METRIC_VOCAB, normalize_metrics
from intent.schema.plan import HardConstraint, Plan
from intent.schema.predicate import Comparison, Predicate


# ----------------------------------------------------------------------
# Schema validation
# ----------------------------------------------------------------------

class TestPlanValidation:
    def test_canonical_plans_all_validate(self):
        for name, plan in CANONICAL_PLANS.items():
            assert isinstance(plan, Plan), name

    def test_invalid_metric_rejected(self):
        with pytest.raises(ValidationError):
            HardConstraint(metric="not_a_real_metric", op="<", threshold=1.0)

    def test_invalid_op_rejected(self):
        with pytest.raises(ValidationError):
            HardConstraint(metric="unsafe_pct", op="~~", threshold=1.0)

    def test_invalid_goal_rejected(self):
        with pytest.raises(ValidationError):
            Plan(goal="minimize_galaxies")

    def test_extra_field_rejected(self):
        # extra="forbid" on the model_config — guard against silent typos.
        with pytest.raises(ValidationError):
            Plan(goal="balance", made_up_field=42)


# ----------------------------------------------------------------------
# Weight compiler
# ----------------------------------------------------------------------

class TestWeightCompiler:
    def test_balanced_returns_base(self):
        w = compile_weights(CANONICAL_PLANS["balanced"])
        for k, v in BASE.items():
            assert w[k] == v, k

    def test_minimize_energy_raises_alpha(self):
        w = compile_weights(CANONICAL_PLANS["energy_greedy"])
        assert w["alpha"] > BASE["alpha"]
        assert w["tau"] <= BASE["tau"]

    def test_minimize_unsafe_raises_lambda_qos(self):
        w = compile_weights(CANONICAL_PLANS["safety_first"])
        assert w["lambda_qos"] > BASE["lambda_qos"]
        assert w["tau"] >= BASE["tau"]

    def test_soft_pref_overrides_goal_delta(self):
        # Build a Plan where a soft pref should beat the goal delta.
        # minimize_unsafe sets lambda_qos to 60; maintain_qos_headroom
        # also sets it to 60, so this is just a no-op sanity check that
        # later-write-wins doesn't regress to BASE.
        plan = Plan(
            goal="minimize_unsafe",
            soft=["maintain_qos_headroom"],
        )
        w = compile_weights(plan)
        assert w["lambda_qos"] == 60.0


# ----------------------------------------------------------------------
# Predicate compiler + evaluator
# ----------------------------------------------------------------------

class TestPredicate:
    def test_empty_plan_compiles_to_trivially_true_predicate(self):
        plan = Plan(goal="balance")
        pred = compile_predicate(plan)
        assert pred.evaluate({}).satisfied

    def test_satisfied_when_all_leaves_pass(self):
        pred = Predicate(
            leaves=[
                Comparison(metric="unsafe_pct", op="<", threshold=1.0),
                Comparison(metric="energy_wh", op="<", threshold=2000.0),
            ]
        )
        r = pred.evaluate({"unsafe_pct": 0.5, "energy_wh": 1900.0})
        assert r.satisfied
        assert all(leaf.satisfied for leaf in r.witness.values())

    def test_unsatisfied_when_any_leaf_fails(self):
        pred = Predicate(
            leaves=[
                Comparison(metric="unsafe_pct", op="<", threshold=1.0),
                Comparison(metric="energy_wh", op="<", threshold=2000.0),
            ]
        )
        r = pred.evaluate({"unsafe_pct": 0.5, "energy_wh": 2100.0})
        assert not r.satisfied
        # Witness should pinpoint the failing leaf.
        bad = [k for k, leaf in r.witness.items() if not leaf.satisfied]
        assert len(bad) == 1 and "energy_wh" in bad[0]

    def test_missing_metric_raises(self):
        pred = Predicate(leaves=[Comparison(metric="mean_q", op=">", threshold=0.9)])
        with pytest.raises(KeyError):
            pred.evaluate({"unsafe_pct": 0.5})

    @pytest.mark.parametrize("op,v,thr,expect", [
        ("<",  1.0, 2.0, True),
        ("<",  2.0, 2.0, False),
        ("<=", 2.0, 2.0, True),
        (">",  3.0, 2.0, True),
        (">=", 2.0, 2.0, True),
        ("==", 2.0, 2.0, True),
        ("!=", 2.0, 2.0, False),
    ])
    def test_comparison_ops(self, op, v, thr, expect):
        pred = Predicate(leaves=[Comparison(metric="mean_q", op=op, threshold=thr)])
        assert pred.evaluate({"mean_q": v}).satisfied == expect


# ----------------------------------------------------------------------
# Metric normalisation
# ----------------------------------------------------------------------

class TestNormalizeMetrics:
    def test_aliases_map_correctly(self):
        raw = {
            "total_reward": -5000.0,
            "mean_reward": -5.0,
            "total_energy_wh": 1990.0,
            "total_switch_wh": 5.0,
            "mean_sec": 0.05,
            "mean_q": 0.99,
            "qos_violation_rate": 0.01,
            "unsafe_rate": 0.005,
            "dpdk_rate": 0.5,
            "usr_rate": 0.5,
            "n_switches": 42,
            "steps": 1009,
        }
        out = normalize_metrics(raw)
        assert out["energy_wh"] == 1990.0
        assert out["flips"] == 42.0
        assert out["unsafe_pct"] == pytest.approx(0.5)
        assert out["qos_violation_pct"] == pytest.approx(1.0)
        # Every canonical metric name should be present.
        for name in METRIC_VOCAB:
            assert name in out, name
