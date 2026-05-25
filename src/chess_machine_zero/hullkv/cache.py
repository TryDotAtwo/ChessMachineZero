"""Hull-backed KV cache API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar

from chess_machine_zero.hullkv.convex_hull_2d import ConvexHull2D, Point2D


ValueT = TypeVar("ValueT")


@dataclass(frozen=True, slots=True)
class HullKVLookup(Generic[ValueT]):
    index: int
    key: Point2D
    value: ValueT | None
    score: float
    used_dense_scan: bool = False


class HullKVCache(Generic[ValueT]):
    """Convex-hull-backed hardmax lookup over immutable 2D keys."""

    def __init__(self, keys: Iterable[Point2D], values: Iterable[ValueT] | None = None) -> None:
        self.keys: tuple[Point2D, ...] = tuple((float(x), float(y)) for x, y in keys)
        if not self.keys:
            raise ValueError("HullKVCache requires at least one key")
        if values is None:
            self.values: tuple[ValueT | None, ...] = tuple(None for _ in self.keys)
        else:
            self.values = tuple(values)
            if len(self.values) != len(self.keys):
                raise ValueError("values length must equal keys length")
        self._hull = ConvexHull2D(self.keys)

    def hardmax(self, query: Point2D) -> HullKVLookup[ValueT]:
        support = self._hull.support(query)
        return HullKVLookup(
            index=support.index,
            key=support.key,
            value=self.values[support.index],
            score=support.score,
            used_dense_scan=support.used_dense_scan,
        )
