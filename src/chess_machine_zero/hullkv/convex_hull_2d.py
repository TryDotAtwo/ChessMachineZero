"""Exact 2D convex-hull support queries."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Iterable


Point2D = tuple[float, float]
EPS = 1e-12


@dataclass(frozen=True, slots=True)
class _CoordRecord:
    x: float
    y: float
    min_index: int

    @property
    def point(self) -> Point2D:
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class HullSupportResult:
    index: int
    key: Point2D
    score: float
    used_dense_scan: bool = False


class ConvexHull2D:
    """Convex hull over fixed 2D keys with exact support-face tie handling."""

    def __init__(self, keys: Iterable[Point2D]) -> None:
        self.keys: tuple[Point2D, ...] = tuple((float(x), float(y)) for x, y in keys)
        if not self.keys:
            raise ValueError("ConvexHull2D requires at least one key")
        self._global_min_index = 0
        self._coord_records = _unique_coord_records(self.keys)
        self._hull: tuple[_CoordRecord, ...] = _monotonic_chain(self._coord_records)
        self._edge_min_indices = _edge_min_indices(self._hull, self.keys)

    @property
    def hull_vertex_count(self) -> int:
        return len(self._hull)

    def support(self, query: Point2D) -> HullSupportResult:
        qx, qy = float(query[0]), float(query[1])
        if _zero(qx) and _zero(qy):
            index = self._global_min_index
            return HullSupportResult(index=index, key=self.keys[index], score=0.0, used_dense_scan=False)
        if len(self._hull) == 1:
            index = self._hull[0].min_index
            return HullSupportResult(index=index, key=self.keys[index], score=_dot(query, self.keys[index]), used_dense_scan=False)

        scores = [_dot(query, record.point) for record in self._hull]
        max_score = max(scores)
        tied_positions = [idx for idx, score in enumerate(scores) if isclose(score, max_score, rel_tol=0.0, abs_tol=EPS)]
        candidate_indices = [self._hull[position].min_index for position in tied_positions]
        if len(tied_positions) >= 2:
            tied_set = set(tied_positions)
            hull_len = len(self._hull)
            for position in tied_positions:
                next_position = (position + 1) % hull_len
                if next_position in tied_set:
                    candidate_indices.append(self._edge_min_indices[(position, next_position)])
                prev_position = (position - 1) % hull_len
                if prev_position in tied_set:
                    candidate_indices.append(self._edge_min_indices[(prev_position, position)])
        index = min(candidate_indices)
        return HullSupportResult(index=index, key=self.keys[index], score=_dot(query, self.keys[index]), used_dense_scan=False)


def _unique_coord_records(keys: tuple[Point2D, ...]) -> tuple[_CoordRecord, ...]:
    coord_to_min: dict[Point2D, int] = {}
    for index, key in enumerate(keys):
        coord_to_min[key] = min(coord_to_min.get(key, index), index)
    return tuple(_CoordRecord(x, y, min_index) for (x, y), min_index in sorted(coord_to_min.items()))


def _monotonic_chain(records: tuple[_CoordRecord, ...]) -> tuple[_CoordRecord, ...]:
    if len(records) <= 2:
        return records
    lower: list[_CoordRecord] = []
    for record in records:
        while len(lower) >= 2 and _cross(lower[-2].point, lower[-1].point, record.point) <= EPS:
            lower.pop()
        lower.append(record)
    upper: list[_CoordRecord] = []
    for record in reversed(records):
        while len(upper) >= 2 and _cross(upper[-2].point, upper[-1].point, record.point) <= EPS:
            upper.pop()
        upper.append(record)
    hull = tuple(lower[:-1] + upper[:-1])
    return hull if hull else records[:1]


def _edge_min_indices(hull: tuple[_CoordRecord, ...], keys: tuple[Point2D, ...]) -> dict[tuple[int, int], int]:
    if len(hull) == 1:
        return {}
    edge_min: dict[tuple[int, int], int] = {}
    for position, start in enumerate(hull):
        next_position = (position + 1) % len(hull)
        end = hull[next_position]
        min_index = min(start.min_index, end.min_index)
        for original_index, key in enumerate(keys):
            if _point_on_segment(key, start.point, end.point):
                min_index = min(min_index, original_index)
        edge_min[(position, next_position)] = min_index
    return edge_min


def _point_on_segment(point: Point2D, start: Point2D, end: Point2D) -> bool:
    if not isclose(_cross(start, end, point), 0.0, rel_tol=0.0, abs_tol=EPS):
        return False
    min_x, max_x = sorted((start[0], end[0]))
    min_y, max_y = sorted((start[1], end[1]))
    return min_x - EPS <= point[0] <= max_x + EPS and min_y - EPS <= point[1] <= max_y + EPS


def _dot(query: Point2D, key: Point2D) -> float:
    return float(query[0]) * float(key[0]) + float(query[1]) * float(key[1])


def _cross(origin: Point2D, a: Point2D, b: Point2D) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def _zero(value: float) -> bool:
    return isclose(value, 0.0, rel_tol=0.0, abs_tol=EPS)
