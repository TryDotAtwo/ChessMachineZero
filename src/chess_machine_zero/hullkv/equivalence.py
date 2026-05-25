"""Dense reference helpers used only by tests and equivalence checks."""

from __future__ import annotations

from typing import Iterable

from chess_machine_zero.hullkv.convex_hull_2d import Point2D


def dense_hardmax_index(query: Point2D, keys: Iterable[Point2D]) -> int:
    key_tuple = tuple(keys)
    if not key_tuple:
        raise ValueError("keys must be nonempty")
    scores = [_dot(query, key) for key in key_tuple]
    max_score = max(scores)
    return next(index for index, score in enumerate(scores) if score == max_score)


def dense_topk_indices(query: Point2D, keys: Iterable[Point2D], k: int) -> tuple[int, ...]:
    if k <= 0:
        raise ValueError("k must be positive")
    scored = [(index, _dot(query, key)) for index, key in enumerate(keys)]
    if not scored:
        raise ValueError("keys must be nonempty")
    return tuple(index for index, _score in sorted(scored, key=lambda item: (-item[1], item[0]))[:k])


def _dot(query: Point2D, key: Point2D) -> float:
    return float(query[0]) * float(key[0]) + float(query[1]) * float(key[1])
