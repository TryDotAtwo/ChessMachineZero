import numpy

from vm_compiler.relations import build_rule_relations, square_index

RULES = build_rule_relations()


def destinations(relation, source):
    return numpy.flatnonzero(relation[square_index(source)]).tolist()


def indices(*channels):
    return sorted(square_index(channel) for channel in channels)


def test_knight_and_king_relations_are_exact_and_board_bounded():
    rules = RULES

    assert destinations(rules.knight, 21) == indices(13, 33, 42)
    assert destinations(rules.king, 11) == indices(12, 21, 22)
    assert not numpy.diag(rules.knight).any()
    assert not numpy.diag(rules.king).any()
    assert numpy.array_equal(rules.knight, rules.knight.T)
    assert numpy.array_equal(rules.king, rules.king.T)


def test_slider_rays_and_strict_between_relation_are_exact():
    rules = RULES

    assert destinations(rules.bishop_ray, 31) == indices(22, 13, 42, 53, 64, 75, 86)
    assert destinations(rules.rook_ray, 11) == indices(
        12, 13, 14, 15, 16, 17, 18, 21, 31, 41, 51, 61, 71, 81
    )
    assert numpy.flatnonzero(rules.between[square_index(11), square_index(18)]).tolist() == indices(
        12, 13, 14, 15, 16, 17
    )
    assert numpy.flatnonzero(rules.between[square_index(11), square_index(88)]).tolist() == indices(
        22, 33, 44, 55, 66, 77
    )
    assert not rules.between[square_index(11), square_index(23)].any()


def test_white_and_black_pawn_geometry_is_directional_and_exact():
    rules = RULES

    assert destinations(rules.pawn_step[0], 52) == indices(53)
    assert destinations(rules.pawn_double[0], 52) == indices(54)
    assert destinations(rules.pawn_attack[0], 52) == indices(43, 63)
    assert destinations(rules.pawn_step[1], 57) == indices(56)
    assert destinations(rules.pawn_double[1], 57) == indices(55)
    assert destinations(rules.pawn_attack[1], 57) == indices(46, 66)
    assert not rules.pawn_double[0, square_index(53)].any()
    assert not rules.pawn_double[1, square_index(56)].any()


def test_all_relation_tensors_are_binary_and_fixed_shape():
    rules = RULES

    assert rules.king.shape == (64, 64)
    assert rules.knight.shape == (64, 64)
    assert rules.rook_ray.shape == (64, 64)
    assert rules.bishop_ray.shape == (64, 64)
    assert rules.pawn_step.shape == (2, 64, 64)
    assert rules.pawn_double.shape == (2, 64, 64)
    assert rules.pawn_attack.shape == (2, 64, 64)
    assert rules.between.shape == (64, 64, 64)
    for relation in rules:
        assert relation.dtype == numpy.float32
        assert set(numpy.unique(relation)).issubset({0.0, 1.0})
