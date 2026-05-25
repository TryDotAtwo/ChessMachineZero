"""Deterministic temperature schedules."""

from __future__ import annotations


def constant_temperature(value: float) -> float:
    if value < 0.0:
        raise ValueError("temperature must be nonnegative")
    return value
