"""Nested convex hulls for exact top-k 2D retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from chess_machine_zero.hullkv.convex_hull_2d import ConvexHull2D, Point2D


@dataclass(frozen=True, slots=True)
class _RemainingPoint:
    original_index: int
    key: Point2D


class NestedHullTopK2D:
    """Exact top-k by repeatedly querying hulls over remaining keys."""

    def __init__(self, keys: Iterable[Point2D]) -> None:
        self.keys: tuple[Point2D, ...] = tuple((float(x), float(y)) for x, y in keys)
        if not self.keys:
            raise ValueError("NestedHullTopK2D requires at least one key")
        self.used_dense_scan = False

    def topk(self, query: Point2D, k: int) -> tuple[int, ...]:
        if k <= 0:
            raise ValueError("k must be positive")
        remaining = [_RemainingPoint(index, key) for index, key in enumerate(self.keys)]
        selected: list[int] = []
        while remaining and len(selected) < k:
            hull = ConvexHull2D(point.key for point in remaining)
            support = hull.support(query)
            selected_point = remaining[support.index]
            selected.append(selected_point.original_index)
            remaining.pop(support.index)
        return tuple(selected)
