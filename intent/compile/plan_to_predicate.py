"""Deterministic Plan -> Predicate compiler.

Slice 0: the Predicate is the conjunction of the Plan's hard
constraints. Empty `hard` gives a trivially-true Predicate (every
metrics dict satisfies it) — that's a valid choice the operator can
make, not an error.
"""

from __future__ import annotations

from intent.schema.plan import Plan
from intent.schema.predicate import Comparison, Predicate


def compile_predicate(plan: Plan) -> Predicate:
    leaves = [
        Comparison(metric=hc.metric, op=hc.op, threshold=hc.threshold)
        for hc in plan.hard
    ]
    return Predicate(leaves=leaves)
