"""Thin wrapper around `Predicate.evaluate` for a consistent import path."""

from __future__ import annotations

from intent.schema.predicate import EvalResult, Predicate


def evaluate(predicate: Predicate, metrics: dict[str, float]) -> EvalResult:
    return predicate.evaluate(metrics)
