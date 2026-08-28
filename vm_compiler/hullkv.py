"""Offline 2D hull metadata builder and dense development oracle."""

from __future__ import annotations

import numpy


def _cross(origin, first, second) -> float:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (
        first[1] - origin[1]
    ) * (second[0] - origin[0])


def _validate_keys(keys: numpy.ndarray) -> numpy.ndarray:
    points = numpy.asarray(keys, dtype=numpy.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("HullKV keys must have shape [N,2]")
    if not numpy.isfinite(points).all():
        raise ValueError("HullKV keys must be finite")

    return points


def _outer_coordinates(coordinates):
    ordered = sorted((x, y) for x, y in coordinates)

    def half(sequence):
        result = []
        for point in sequence:
            while len(result) >= 2 and _cross(result[-2], result[-1], point) <= 0.0:
                result.pop()
            result.append(point)
        return result

    if len(ordered) <= 2:
        return set(ordered)
    lower = half(ordered)
    upper = half(reversed(ordered))
    return set(lower[:-1] + upper[:-1])


def build_nested_hull_2d(keys: numpy.ndarray, layers: int) -> numpy.ndarray:
    """Return every original index on the first ``layers`` convex onion layers."""
    points = _validate_keys(keys)
    if layers <= 0:
        raise ValueError("HullKV layer count must be positive")

    coordinate_indices = {}
    for index, point in enumerate(points):
        coordinate_indices.setdefault((float(point[0]), float(point[1])), []).append(index)

    remaining = set(coordinate_indices)
    selected = set()
    for _ in range(layers):
        if not remaining:
            break
        outer = _outer_coordinates(remaining)
        for coordinate in outer:
            selected.update(coordinate_indices[coordinate])
        remaining.difference_update(outer)

    # Every query score is tied at zero for q=(0,0); preserve its stable top-k prefix.
    selected.update(range(min(layers, len(points))))
    return numpy.array(sorted(selected), dtype=numpy.int64)


def build_hull_2d(keys: numpy.ndarray) -> numpy.ndarray:
    return build_nested_hull_2d(keys, layers=1)


def support_topk_2d(
    queries: numpy.ndarray,
    keys: numpy.ndarray,
    hull_indices: numpy.ndarray,
    k: int,
) -> numpy.ndarray:
    query_matrix = numpy.asarray(queries, dtype=numpy.float32)
    key_matrix = numpy.asarray(keys, dtype=numpy.float32)
    hull = numpy.asarray(hull_indices, dtype=numpy.int64)
    if query_matrix.ndim != 2 or query_matrix.shape[1] != 2:
        raise ValueError("HullKV queries must have shape [M,2]")
    if k <= 0 or k > len(hull):
        raise ValueError("HullKV k is outside the hull capacity")

    result = []
    for query in query_matrix:
        ranked = sorted(
            ((-float(query @ key_matrix[index]), int(index)) for index in hull),
            key=lambda item: (item[0], item[1]),
        )
        result.append([index for _, index in ranked[:k]])
    return numpy.asarray(result, dtype=numpy.int64)
