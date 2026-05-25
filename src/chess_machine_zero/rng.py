"""Deterministic seed helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass


DEFAULT_SEED = 20260524


@dataclass(frozen=True, slots=True)
class DeterministicSeed:
    """Small seed wrapper used by host-side Milestone 1 code."""

    value: int = DEFAULT_SEED

    def random(self) -> random.Random:
        return random.Random(self.value)
