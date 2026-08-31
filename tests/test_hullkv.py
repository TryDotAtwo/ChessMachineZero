import numpy

from vm_compiler.hullkv import build_hull_2d, build_nested_hull_2d, support_topk_2d


KEYS = numpy.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, 0.0],
    ],
    dtype=numpy.float32,
)


def test_hull_keeps_lowest_duplicate_index_and_zero_query_sentinel():
    hull = build_hull_2d(KEYS)

    assert hull.tolist() == [0, 1, 2, 3, 4, 5]


def test_support_lookup_is_exact_and_lowest_index_stable():
    queries = numpy.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=numpy.float32)

    winners = support_topk_2d(queries, KEYS, build_hull_2d(KEYS), k=1)

    assert winners.tolist() == [[1], [1], [0]]


def test_nested_hull_competitors_are_score_then_index_stable():
    queries = numpy.array([[1.0, 0.0], [1.0, 1.0]], dtype=numpy.float32)

    competitors = support_topk_2d(queries, KEYS, build_hull_2d(KEYS), k=2)

    assert competitors.tolist() == [[1, 5], [1, 2]]


def test_nested_layers_match_dense_topk_with_interior_points_and_duplicates():
    random = numpy.random.default_rng(20260828)
    keys = random.normal(size=(96, 2)).astype(numpy.float32)
    keys[95] = keys[7]
    queries = random.normal(size=(128, 2)).astype(numpy.float32)
    queries[0] = 0.0
    k = 4

    candidates = build_nested_hull_2d(keys, layers=k)
    actual = support_topk_2d(queries, keys, candidates, k=k)
    dense = support_topk_2d(queries, keys, numpy.arange(len(keys)), k=k)

    assert numpy.array_equal(actual, dense)


def test_collinear_boundary_tie_preserves_lowest_original_index():
    keys = numpy.array([(0, 0), (1, 0), (1, -1), (1, 1), (-1, 0)], dtype=numpy.float32)
    actual = support_topk_2d(numpy.array([[1, 0]]), keys, build_hull_2d(keys), 1)
    assert actual.tolist() == [[1]]


def test_boundary_duplicates_and_zero_query_match_stable_dense_topk():
    keys = numpy.array([(0, 0), (1, 0), (1, -1), (1, 1), (-1, 0),
                        (1, 0), (0, .5), (0, -.5)], dtype=numpy.float32)
    queries = numpy.array([(1, 0), (0, 0), (-1, 0), (0, 1), (1, 1)], dtype=numpy.float32)
    scores = queries @ keys.T
    for count in range(1, len(keys) + 1):
        expected = numpy.argsort(-scores, axis=1, kind="stable")[:, :count]
        actual = support_topk_2d(queries, keys, build_nested_hull_2d(keys, count), count)
        assert numpy.array_equal(actual, expected)


def test_degenerate_collinear_layer_preserves_perpendicular_query_ties():
    # The collinear middle point at original index 1 must survive layer 1:
    # q=(0,1) ties every point, not just endpoints plus index-0 sentinel.
    keys = numpy.array([(0, 0), (.5, 0), (2, 0), (-2, 0), (.5, 0)], dtype=numpy.float32)
    hull = build_hull_2d(keys)
    actual = support_topk_2d(numpy.array([[0, 1]]), keys, hull, 2)
    assert actual.tolist() == [[0, 1]]
