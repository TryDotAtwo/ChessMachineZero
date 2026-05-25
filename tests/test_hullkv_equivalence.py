from __future__ import annotations

from chess_machine_zero.hullkv.cache import HullKVCache
from chess_machine_zero.hullkv.convex_hull_2d import ConvexHull2D
from chess_machine_zero.hullkv.equivalence import dense_hardmax_index, dense_topk_indices
from chess_machine_zero.hullkv.nested_hulls import NestedHullTopK2D


KEYS = (
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (2.0, 2.0),
    (-2.0, 3.0),
    (3.0, -1.0),
    (1.5, 1.0),
    (-1.0, -2.0),
)
QUERIES = (
    (1.0, 2.0),
    (-3.0, 1.0),
    (2.0, -1.0),
    (-1.0, -1.0),
    (0.25, 0.75),
)


def test_convex_hull_support_matches_dense_hardmax_for_queries() -> None:
    hull = ConvexHull2D(KEYS)

    for query in QUERIES:
        result = hull.support(query)
        assert result.index == dense_hardmax_index(query, KEYS)
        assert result.used_dense_scan is False


def test_hull_kv_cache_returns_dense_equivalent_indices_and_values() -> None:
    values = tuple(f"value-{index}" for index in range(len(KEYS)))
    cache = HullKVCache(KEYS, values)

    for query in QUERIES:
        result = cache.hardmax(query)
        expected_index = dense_hardmax_index(query, KEYS)
        assert result.index == expected_index
        assert result.value == values[expected_index]
        assert result.used_dense_scan is False


def test_nested_hulls_topk_matches_dense_topk_order() -> None:
    nested = NestedHullTopK2D(KEYS)

    for query in QUERIES:
        assert nested.topk(query, k=3) == dense_topk_indices(query, KEYS, k=3)
        assert nested.used_dense_scan is False


def test_hull_support_tie_breaks_to_dense_lowest_original_index() -> None:
    keys = ((0.0, 0.0), (1.0, 0.0), (0.5, 0.0), (2.0, 0.0))
    hull = ConvexHull2D(keys)

    assert hull.support((0.0, 1.0)).index == dense_hardmax_index((0.0, 1.0), keys)
