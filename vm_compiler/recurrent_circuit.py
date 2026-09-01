"""Compile the request/context feedback transition to generic tensor ops only.

The Python code runs offline. The emitted artifact contains no chess opcode:
request membership, append routing, position reconstruction, legal generation
and context selection lower to frozen projections, GEMMs and hardmax gates.
"""

from __future__ import annotations

import numpy as np

from .circuit import Circuit, exact_record
from .compiler import build_position_reconstruction_artifact
from .context import build_context_0
from .legal_circuit import _candidates, compile_legal_set
from .protocol import (
    HISTORY_ROWS,
    LEGAL_ROWS,
    MOVE_ROWS,
    PIECE_CHANNELS,
    STATUS_CHANNELS,
    VOCAB_SIZE,
)
from .state_circuit import EMPTY, build_square_decoder
from .relations import build_rule_relations


def _token_projection(channels):
    projection = np.zeros((VOCAB_SIZE, 1), dtype=np.float32)
    projection[list(channels)] = 1.0
    return projection


def _row_sum(c: Circuit, value: int, name: str | None = None) -> int:
    return c.project(value, np.ones((c.shapes[value][1], 1), dtype=np.float32), name)


def _status(c: Circuit, channel: int, name: str) -> int:
    row = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
    row[0, channel] = 1.0
    return c.constant(name, row)


def _sum(c: Circuit, values, name: str | None = None) -> int:
    values = tuple(values)
    result = values[0]
    for value in values[1:]:
        result = c.add(result, value, name)
    return result


def _all(c: Circuit, *values, name: str | None = None) -> int:
    return c.threshold(_sum(c, values), len(values) - .5, name=name)


def _same_bits(c: Circuit, left: int, right: int, name: str | None = None) -> int:
    different = c.logical_or(
        c.logical_and(left, c.logical_not(right)),
        c.logical_and(c.logical_not(left), right),
    )
    return c.logical_not(different, name=name)


def _concat_rows_many(c: Circuit, values, name: str | None = None) -> int:
    values = tuple(values)
    while len(values) > 255:
        values = tuple(c.concat_rows(values[index:index + 255])
                       for index in range(0, len(values), 255))
    return c.concat_rows(values, name=name)


def _position_prefixes(c: Circuit, position_artifact, position_values):
    """Re-use one event graph and attend only to events available at each ply."""

    final_attention = position_artifact.operations[-1]
    queries, keys, values = (position_values[value] for value in final_attention.inputs)
    streams = (64, 464, 864, 1264, 1664)
    boards = []
    for ply in range(401):
        candidates = list(range(64))
        for start in streams:
            candidates.extend(range(start, start + ply))
        boards.append(c.attention(
            queries,
            keys,
            values,
            min(8, len(candidates)),
            .5,
            candidates,
            name=f"position_after_{ply}_plies",
        ))
    return tuple(boards)


def _castling_rights_by_prefix(c: Circuit, history: int):
    from_rows = c.route(history, 0, HISTORY_ROWS, MOVE_ROWS)
    to_rows = c.route(history, 1, HISTORY_ROWS, MOVE_ROWS)
    touched_from = np.zeros((VOCAB_SIZE, 4), dtype=np.float32)
    touched_to = np.zeros_like(touched_from)
    # White queenside/kingside, then black queenside/kingside.
    for index, (king, rook) in enumerate(((51, 11), (51, 81), (58, 18), (58, 88))):
        touched_from[[king, rook], index] = 1.0
        touched_to[rook, index] = 1.0
    touches = c.threshold(c.add(
        c.project(from_rows, touched_from),
        c.project(to_rows, touched_to),
    ), .5, name="castling_right_touched_per_ply")
    cumulative = c.transpose(c.project(
        c.transpose(touches),
        np.triu(np.ones((400, 400), dtype=np.float32)),
        "castling_touch_prefix_sum",
    ))
    rights_after_moves = c.logical_not(c.threshold(cumulative, .5))
    prefix_rights = c.concat_rows((
        c.constant("initial_castling_rights", np.ones((1, 4), dtype=np.float32)),
        rights_after_moves,
    ), name="castling_rights_by_prefix")
    return prefix_rights, c.route(prefix_rights, 400, 401, name="current_castling_rights")


def _history_zeroing_moves(c: Circuit, history: int, prefix_boards):
    """Detect every pawn move/capture from the actual tensor prefix boards."""

    decoder = build_square_decoder()
    from_rows = c.route(history, 0, HISTORY_ROWS, MOVE_ROWS)
    to_rows = c.route(history, 1, HISTORY_ROWS, MOVE_ROWS)
    from_squares = c.project(from_rows, decoder, "history_source_squares")
    to_squares = c.project(to_rows, decoder, "history_target_squares")
    boards_before = _concat_rows_many(c, prefix_boards[:400],
                                      name="boards_before_history_moves")
    gather = np.tile(np.eye(VOCAB_SIZE, dtype=np.float32), (64, 1))

    def pieces_at(squares, name):
        square_bits = c.reshape(squares, 400 * 64, 1)
        masks = c.project(square_bits, np.ones((1, VOCAB_SIZE), dtype=np.float32))
        selected = c.logical_and(masks, boards_before)
        return c.project(c.reshape(selected, 400, 64 * VOCAB_SIZE), gather, name)

    sources = pieces_at(from_squares, "history_source_pieces")
    targets = pieces_at(to_squares, "history_target_pieces")
    pawn = c.project(sources, _token_projection((
        PIECE_CHANNELS["WHITE_PAWN"], PIECE_CHANNELS["BLACK_PAWN"],
    )), "history_pawn_moves")
    capture = c.project(targets, _token_projection(PIECE_CHANNELS.values()),
                        "history_capture_moves")
    return c.logical_or(pawn, capture, name="history_zeroing_moves")


def _halfmove_clock(c: Circuit, active: int, zeroing: int):
    """Compute active-count minus the last zeroing index using one hardmax."""

    active_count = _row_sum(c, c.transpose(active), "accepted_history_ply_count")
    indicators = c.concat_columns((
        c.constant("no_zeroing_sentinel", [[1.0]]),
        c.transpose(zeroing),
    ), name="zeroing_move_indicators")
    scores = c.project(indicators, np.eye(401, dtype=np.float32) * 4.0,
                       "last_zeroing_active_score")
    bias = -4.0 + np.arange(401, dtype=np.float32)[None, :] / 512.0
    scores = c.bias(scores, bias, "last_zeroing_position_tiebreak")
    selected = c.hardmax(scores, name="last_zeroing_one_hot")
    last_zeroing = c.project(selected, np.arange(401, dtype=np.float32)[:, None],
                             "last_zeroing_ply")
    return c.add(active_count, c.scale(last_zeroing, -1.0), name="halfmove_clock")


def _compact_legal_post_boards(c: Circuit, board: int, legal_rows: int):
    """Apply all 256 compact legal triples in parallel as tensor scatter algebra."""

    decoder = build_square_decoder()
    from_rows = c.route(legal_rows, 0, LEGAL_ROWS, MOVE_ROWS)
    to_rows = c.route(legal_rows, 1, LEGAL_ROWS, MOVE_ROWS)
    result_rows = c.route(legal_rows, 2, LEGAL_ROWS, MOVE_ROWS)
    from_squares = c.project(from_rows, decoder, "future_source_squares")
    to_squares = c.project(to_rows, decoder, "future_target_squares")
    source_pieces = c.matmul(from_squares, board, name="future_source_pieces")
    target_pieces = c.matmul(to_squares, board, name="future_target_pieces")
    pawn = c.project(source_pieces, _token_projection((
        PIECE_CHANNELS["WHITE_PAWN"], PIECE_CHANNELS["BLACK_PAWN"],
    )), "future_pawn_moves")
    capture = c.project(target_pieces, _token_projection(PIECE_CHANNELS.values()),
                        "future_capture_moves")
    nonzeroing = c.logical_not(c.logical_or(pawn, capture), name="future_nonzeroing_moves")

    board_flat = c.reshape(board, 1, 64 * VOCAB_SIZE)
    tiled = c.reshape(c.matmul(
        c.constant("legal_slot_units", np.ones((256, 1), dtype=np.float32)),
        board_flat,
    ), 256 * 64, VOCAB_SIZE, name="future_tiled_boards")
    source_bits = c.reshape(from_squares, 256 * 64, 1)
    target_bits = c.reshape(to_squares, 256 * 64, 1)
    source_mask = c.project(source_bits, np.ones((1, VOCAB_SIZE), dtype=np.float32))
    target_mask = c.project(target_bits, np.ones((1, VOCAB_SIZE), dtype=np.float32))
    selected_source = c.logical_and(source_mask, tiled)
    empty_projection = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
    empty_projection[0, EMPTY] = 1.0
    cleared = _sum(c, (
        tiled,
        c.scale(selected_source, -1.0),
        c.project(source_bits, empty_projection),
    ), "future_source_cleared_boards")

    repeat = np.zeros((256 * 64, 256), dtype=np.float32)
    repeat[np.arange(256 * 64), np.arange(256).repeat(64)] = 1.0
    repeated_results = c.matmul(c.constant("legal_result_row_repeat", repeat), result_rows,
                                name="future_repeated_result_pieces")
    selected_target = c.logical_and(target_mask, cleared)
    written_target = c.logical_and(target_mask, repeated_results)
    post_rows = _sum(c, (cleared, c.scale(selected_target, -1.0), written_target),
                     "future_normal_post_move_boards")

    # The move token already identifies every special relocation. Apply the
    # extra rook/EP square delta with frozen lookup matrices; no runtime branch
    # or chess-specific opcode is introduced.
    special_patterns = []
    for color, rank in ((0, 0), (1, 7)):
        king_piece = PIECE_CHANNELS["BLACK_KING" if color else "WHITE_KING"]
        rook_piece = PIECE_CHANNELS["BLACK_ROOK" if color else "WHITE_ROOK"]
        for king_file, rook_file, rook_target_file in ((2, 0, 3), (6, 7, 5)):
            special_patterns.append((
                (4 * 8 + rank, king_file * 8 + rank, king_piece),
                ((rook_file * 8 + rank, rook_piece, EMPTY),
                 (rook_target_file * 8 + rank, EMPTY, rook_piece)),
            ))
    for color in range(2):
        rank = 4 if color == 0 else 3
        direction = 1 if color == 0 else -1
        pawn = PIECE_CHANNELS["WHITE_PAWN" if color == 0 else "BLACK_PAWN"]
        victim_piece = PIECE_CHANNELS["BLACK_PAWN" if color == 0 else "WHITE_PAWN"]
        for source_file in range(8):
            for target_file in (source_file - 1, source_file + 1):
                if 0 <= target_file < 8:
                    special_patterns.append((
                        (source_file * 8 + rank,
                         target_file * 8 + rank + direction, pawn),
                        ((target_file * 8 + rank, victim_piece, EMPTY),),
                    ))

    match_weights = [np.zeros((VOCAB_SIZE, len(special_patterns)), dtype=np.float32)
                     for _ in range(MOVE_ROWS)]
    deltas = np.zeros((len(special_patterns), 64 * VOCAB_SIZE), dtype=np.float32)
    for pattern_index, (triple, changes) in enumerate(special_patterns):
        for slot, token in enumerate((
            (triple[0] // 8 + 1) * 10 + triple[0] % 8 + 1,
            (triple[1] // 8 + 1) * 10 + triple[1] % 8 + 1,
            triple[2],
        )):
            match_weights[slot][token, pattern_index] = 1.0
        for square, before, after in changes:
            deltas[pattern_index, square * VOCAB_SIZE + before] -= 1.0
            deltas[pattern_index, square * VOCAB_SIZE + after] += 1.0
    special_match = c.threshold(_sum(c, (
        c.project(rows, weights)
        for rows, weights in zip((from_rows, to_rows, result_rows), match_weights)
    )), 2.5, name="future_special_move_match")
    special_delta = c.matmul(special_match, c.constant("future_special_board_deltas", deltas),
                             name="future_special_board_delta")
    post_rows = c.add(post_rows, c.reshape(special_delta, 256 * 64, VOCAB_SIZE),
                      name="future_post_move_boards")
    return c.reshape(post_rows, 256, 64 * VOCAB_SIZE), nonzeroing, from_rows, to_rows


def _reply_existence_patterns(relations):
    """Collapse result/color duplicates that cannot affect move existence."""

    unique = {}
    for candidate in _candidates(relations):
        if candidate.mode == "ep":
            # This helper is consumed only for a preceding non-zeroing move.
            # Such a move cannot create an en-passant right; any older right has
            # expired. EP candidates are therefore impossible in this slice.
            continue
        color = -1 if candidate.mode == "ordinary" else candidate.color
        key = (candidate.source, candidate.target, candidate.kind, candidate.mode,
               color, candidate.rook_source, candidate.rook_target)
        unique.setdefault(key, candidate)
    return tuple(unique.values())


def _batched_reply_exists(c: Circuit, post_boards: int, reply_black: int,
                          castling_rights: int):
    """Exact legal-reply existence for 256 post-move boards in parallel.

    The implementation is factorized into `[256, candidate]` predicates and
    `[256, 64, 64]` ray/pin relations. It never materializes a
    `[256, candidate, 64]` tensor and lowers only to generic frozen operations.
    """

    state_count = c.shapes[post_boards][0]
    if state_count != 256 or c.shapes[post_boards][1] != 64 * VOCAB_SIZE:
        raise ValueError("reply existence expects 256 flattened one-hot boards")
    relations = build_rule_relations()
    candidates = _reply_existence_patterns(relations)
    candidate_count = len(candidates)
    eye = np.eye(64, dtype=np.float32)
    sources = np.stack([eye[p.source] for p in candidates])
    targets = np.stack([eye[p.target] for p in candidates])

    board_rows = c.reshape(post_boards, state_count * 64, VOCAB_SIZE,
                           name="future_reply_board_rows")
    planes = []
    piece_names = ("PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING")
    for color in range(2):
        color_planes = []
        for kind in range(6):
            channel = PIECE_CHANNELS[("BLACK_" if color else "WHITE_") + piece_names[kind]]
            value = c.project(
                board_rows,
                _token_projection((channel,)),
                name=f"future_reply_piece_{color}_{kind}",
            )
            color_planes.append(c.reshape(value, state_count, 64))
        planes.append(tuple(color_planes))
    own = tuple(c.select(reply_black, planes[1][kind], planes[0][kind],
                         name=f"future_reply_own_kind_{kind}") for kind in range(6))
    enemy = tuple(c.select(reply_black, planes[0][kind], planes[1][kind],
                           name=f"future_reply_enemy_kind_{kind}") for kind in range(6))
    own_occupancy = _sum(c, own, "future_reply_own_occupancy")
    enemy_occupancy = _sum(c, enemy, "future_reply_enemy_occupancy")
    occupancy = c.add(own_occupancy, enemy_occupancy, "future_reply_occupancy")
    own_king = own[5]

    source_kind_matches = []
    for kind in range(6):
        kind_mask = np.array([[float(p.kind == kind)] for p in candidates],
                             dtype=np.float32)
        source_kind_matches.append(c.matmul(
            own[kind],
            c.constant(f"future_reply_sources_kind_{kind}", (sources * kind_mask).T),
        ))
    source_matches = _sum(c, source_kind_matches, "future_reply_source_matches")
    target_matrix = c.constant("future_reply_targets", targets.T)
    target_own = c.matmul(own_occupancy, target_matrix, name="future_reply_target_own")
    target_enemy = c.matmul(enemy_occupancy, target_matrix, name="future_reply_target_enemy")
    target_king = c.matmul(enemy[5], target_matrix, name="future_reply_target_king")

    paths = np.stack([relations.between[p.source, p.target].copy() for p in candidates])
    for index, candidate in enumerate(candidates):
        if candidate.mode == "castle":
            paths[index] = relations.between[candidate.source, candidate.rook_source]
    clear_path = c.logical_not(c.threshold(c.matmul(
        occupancy, c.constant("future_reply_paths", paths.T)), .5),
        name="future_reply_clear_path")

    flags = {
        mode: np.array([[float(p.mode == mode) for p in candidates]], dtype=np.float32)
        for mode in ("ordinary", "quiet", "capture", "castle")
    }
    right_color = c.select(
        reply_black,
        c.constant("future_reply_black_patterns", np.array([[
            float(p.mode == "ordinary" or p.color == 1) for p in candidates
        ]], dtype=np.float32)),
        c.constant("future_reply_white_patterns", np.array([[
            float(p.mode == "ordinary" or p.color == 0) for p in candidates
        ]], dtype=np.float32)),
        name="future_reply_right_color",
    )
    pawn_target = _sum(c, (
        c.logical_and(c.constant("future_reply_quiet", flags["quiet"]),
                      c.logical_not(target_enemy)),
        c.logical_and(c.constant("future_reply_capture", flags["capture"]), target_enemy),
        c.constant("future_reply_non_pawns", flags["ordinary"] + flags["castle"]),
    ), "future_reply_pawn_target")

    castle_routes = np.zeros((4, candidate_count), dtype=np.float32)
    rook_sources = np.zeros((64, candidate_count), dtype=np.float32)
    castle_paths = np.zeros((64, candidate_count), dtype=np.float32)
    for index, candidate in enumerate(candidates):
        if candidate.mode != "castle":
            continue
        right_index = candidate.color * 2 + int(candidate.target // 8 == 6)
        castle_routes[right_index, index] = 1.0
        rook_sources[candidate.rook_source, index] = 1.0
        castle_paths[:, index] = relations.between[candidate.source, candidate.target]
        castle_paths[[candidate.source, candidate.target], index] = 1.0
    non_castle = 1.0 - flags["castle"]
    rights = c.add(c.matmul(castling_rights, c.constant(
        "future_reply_castle_routes", castle_routes)),
        c.constant("future_reply_non_castle_rights", non_castle),
        name="future_reply_castling_right")
    rook_present = c.add(c.matmul(own[3], c.constant(
        "future_reply_rook_sources", rook_sources)),
        c.constant("future_reply_non_castle_rook", non_castle),
        name="future_reply_castling_rook_present")

    # Full enemy attack map for possible king destinations. Removing the old
    # king square is sufficient: a captured endpoint never attacks itself and
    # ray-interior masks exclude both endpoints.
    without_king = c.add(occupancy, c.scale(own_king, -1.0),
                         name="future_reply_occupancy_without_king")
    all_between_projection = relations.between.transpose(2, 1, 0).reshape(64, 4096)
    all_blockers = c.reshape(c.matmul(
        without_king,
        c.constant("future_reply_all_between_projection", all_between_projection),
    ), state_count * 64, 64, name="future_reply_all_ray_blockers")
    all_clear = c.logical_not(c.threshold(all_blockers, .5),
                              name="future_reply_all_clear_rays")
    state_repeat = np.zeros((state_count * 64, state_count), dtype=np.float32)
    state_repeat[np.arange(state_count * 64), np.arange(state_count).repeat(64)] = 1.0
    repeat_states = c.constant("future_reply_state_repeat", state_repeat)
    repeated_enemy = tuple(c.matmul(repeat_states, plane,
                                    name=f"future_reply_repeated_enemy_{kind}")
                           for kind, plane in enumerate(enemy))
    geometries = (None, relations.knight, relations.bishop_ray, relations.rook_ray,
                  relations.bishop_ray + relations.rook_ray, relations.king)
    all_attacks = []
    for kind in range(6):
        if kind == 0:
            geometry = c.select(
                reply_black,
                c.constant("future_reply_white_pawn_attack_map",
                           np.tile(relations.pawn_attack[0].T, (state_count, 1))),
                c.constant("future_reply_black_pawn_attack_map",
                           np.tile(relations.pawn_attack[1].T, (state_count, 1))),
            )
        else:
            geometry = c.constant(
                f"future_reply_attack_map_kind_{kind}",
                np.tile(geometries[kind].T, (state_count, 1)),
            )
        all_attacks.append(c.logical_and(geometry, repeated_enemy[kind]))
    all_direct = _sum(c, (all_attacks[kind] for kind in (0, 1, 5)))
    all_rays = _sum(c, (all_attacks[kind] for kind in (2, 3, 4)))
    attacked = c.reshape(c.threshold(_row_sum(c, _sum(c, (
        all_direct, c.logical_and(all_rays, all_clear),
    ))), .5), state_count, 64, name="future_reply_attacked_squares")
    king_safe = c.logical_not(c.matmul(attacked, target_matrix),
                              name="future_reply_king_safe")
    castle_safe = c.logical_not(c.threshold(c.matmul(
        attacked, c.constant("future_reply_castle_paths", castle_paths)), .5),
        name="future_reply_castle_path_safe")

    # Current checkers, single-check evasion targets, and pinned-source target
    # constraints give exact non-king safety without per-candidate board cubes.
    between_projection = relations.between.transpose(1, 2, 0).reshape(64, 4096)
    between_by_source = c.reshape(c.matmul(
        own_king, c.constant("future_reply_between_by_source", between_projection),
    ), state_count * 64, 64, name="future_reply_between_source_attacker")
    blocker_counts = c.grouped_matmul(
        occupancy, between_by_source, state_count,
        name="future_reply_king_ray_blocker_counts",
    )
    single_attacks = []
    for kind in range(6):
        if kind == 0:
            geometry = c.select(
                reply_black,
                c.matmul(own_king, c.constant(
                    "future_reply_white_pawn_checker_geometry", relations.pawn_attack[0].T)),
                c.matmul(own_king, c.constant(
                    "future_reply_black_pawn_checker_geometry", relations.pawn_attack[1].T)),
            )
        else:
            geometry = c.matmul(own_king, c.constant(
                f"future_reply_checker_geometry_kind_{kind}", geometries[kind].T))
        single_attacks.append(c.logical_and(geometry, enemy[kind]))
    direct_checkers = _sum(c, (single_attacks[kind] for kind in (0, 1, 5)))
    ray_attackers = _sum(c, (single_attacks[kind] for kind in (2, 3, 4)))
    clear_current_rays = c.logical_not(c.threshold(blocker_counts, .5))
    ray_checkers = c.logical_and(ray_attackers, clear_current_rays)
    checkers = _sum(c, (direct_checkers, ray_checkers), "future_reply_checkers")
    checker_count = _row_sum(c, checkers, "future_reply_checker_count")
    has_check = c.threshold(checker_count, .5)
    double_check = c.threshold(checker_count, 1.5)
    single_check = c.logical_and(has_check, c.logical_not(double_check))

    between_by_attacker = c.reshape(c.matmul(
        own_king,
        c.constant("future_reply_between_by_attacker", relations.between.reshape(64, 4096)),
    ), state_count * 64, 64, name="future_reply_between_attacker_target")
    evasion_between = c.grouped_matmul(
        checkers, between_by_attacker, state_count,
        name="future_reply_single_checker_between_squares",
    )
    evasion_targets = c.threshold(c.add(evasion_between, checkers), .5)
    target_evasion = c.matmul(c.select(
        c.logical_not(has_check),
        c.constant("future_reply_all_evasion_targets", np.ones((1, 64), dtype=np.float32)),
        c.select(single_check, evasion_targets,
                 c.constant("future_reply_no_double_check_evasion", np.zeros((1, 64), dtype=np.float32))),
    ), target_matrix, name="future_reply_target_evasion")

    exactly_one_blocker = c.logical_and(
        c.threshold(blocker_counts, .5),
        c.logical_not(c.threshold(blocker_counts, 1.5)),
    )
    sole_slider = c.logical_and(ray_attackers, exactly_one_blocker,
                                name="future_reply_sole_blocker_sliders")
    repeated_sole_slider = c.matmul(repeat_states, sole_slider)
    pin_sources = c.logical_and(between_by_source, repeated_sole_slider,
                                name="future_reply_pin_source_attacker")
    allowed_pin_targets = c.threshold(c.add(
        between_by_attacker,
        c.constant("future_reply_pin_attacker_targets",
                   np.tile(np.eye(64, dtype=np.float32), (state_count, 1))),
    ), .5)
    invalid_pin_pairs = c.threshold(c.grouped_matmul(
        pin_sources, c.logical_not(allowed_pin_targets), state_count,
        name="future_reply_invalid_pin_pair_counts",
    ), .5, name="future_reply_invalid_pin_pairs")
    pair_decoder = np.zeros((4096, candidate_count), dtype=np.float32)
    for index, candidate in enumerate(candidates):
        pair_decoder[candidate.source * 64 + candidate.target, index] = 1.0
    invalid_pin = c.matmul(
        c.reshape(invalid_pin_pairs, state_count, 4096),
        c.constant("future_reply_candidate_pair_decoder", pair_decoder),
        name="future_reply_candidate_invalid_pin",
    )
    non_king_safe = _all(c, target_evasion, c.logical_not(invalid_pin),
                         name="future_reply_non_king_safe")
    is_king = c.constant("future_reply_king_candidates", np.array([[
        float(candidate.kind == 5) for candidate in candidates
    ]], dtype=np.float32))
    safe = c.select(is_king, king_safe, non_king_safe, name="future_reply_safe")

    legal = _all(
        c, right_color, source_matches, c.logical_not(target_own),
        c.logical_not(target_king), clear_path, pawn_target, rights,
        rook_present, safe, castle_safe, name="future_reply_legal_candidates",
    )
    return c.threshold(_row_sum(c, legal), .5, name="future_has_legal_reply")


def _insufficient_material(c: Circuit, board: int):
    """Exact python-chess material predicate for standard-chess positions."""

    def count(channels, name):
        squares = c.project(board, _token_projection(channels), f"{name}_squares")
        return _row_sum(c, c.transpose(squares), name)

    major_or_pawn = count((
        PIECE_CHANNELS["WHITE_PAWN"], PIECE_CHANNELS["BLACK_PAWN"],
        PIECE_CHANNELS["WHITE_ROOK"], PIECE_CHANNELS["BLACK_ROOK"],
        PIECE_CHANNELS["WHITE_QUEEN"], PIECE_CHANNELS["BLACK_QUEEN"],
    ), "pawn_rook_queen_count")
    no_major_or_pawn = c.logical_not(c.threshold(major_or_pawn, .5))

    knight_count = count((PIECE_CHANNELS["WHITE_KNIGHT"],
                          PIECE_CHANNELS["BLACK_KNIGHT"]), "knight_count")
    bishop_squares = c.project(board, _token_projection((
        PIECE_CHANNELS["WHITE_BISHOP"], PIECE_CHANNELS["BLACK_BISHOP"],
    )), "bishop_squares")
    bishop_count = _row_sum(c, c.transpose(bishop_squares), "bishop_count")
    has_knight = c.threshold(knight_count, .5)
    multiple_knights = c.threshold(knight_count, 1.5)
    exactly_one_knight = c.logical_and(has_knight, c.logical_not(multiple_knights))
    no_knight = c.logical_not(has_knight)
    no_bishop = c.logical_not(c.threshold(bishop_count, .5))

    dark = np.array([[float((square // 8 + square % 8) % 2 == 0)]
                     for square in range(64)], dtype=np.float32)
    light = 1.0 - dark
    dark_count = c.project(c.transpose(bishop_squares), dark, "dark_square_bishop_count")
    light_count = c.project(c.transpose(bishop_squares), light, "light_square_bishop_count")
    all_bishops_same_color = c.logical_or(
        c.logical_not(c.threshold(dark_count, .5)),
        c.logical_not(c.threshold(light_count, .5)),
        name="all_bishops_same_square_color",
    )
    minor_material_insufficient = c.logical_or(
        _all(c, exactly_one_knight, no_bishop),
        _all(c, no_knight, all_bishops_same_color),
    )
    return c.logical_and(no_major_or_pawn, minor_material_insufficient,
                         name="insufficient_material")


def _effective_ep_by_prefix(c: Circuit, history: int, prefix_boards):
    """Return python-chess-style legal EP target keys for all 401 prefixes."""

    state_count = 401
    candidates = []
    for color in range(2):
        rank = 4 if color == 0 else 3
        direction = 1 if color == 0 else -1
        for source_file in range(8):
            for target_file in (source_file - 1, source_file + 1):
                if 0 <= target_file < 8:
                    source = source_file * 8 + rank
                    target = target_file * 8 + rank + direction
                    victim = target_file * 8 + rank
                    candidates.append((color, source, target, victim))
    candidate_count = len(candidates)

    def channel(square):
        return (square // 8 + 1) * 10 + square % 8 + 1

    history_rows = tuple(c.route(history, slot, HISTORY_ROWS, MOVE_ROWS)
                         for slot in range(MOVE_ROWS))
    pad = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
    pad[0, 0] = 1.0
    previous = tuple(c.concat_rows((c.constant("ep_prefix_pad", pad), rows))
                     for rows in history_rows)
    match_projections = [np.zeros((VOCAB_SIZE, candidate_count), dtype=np.float32)
                         for _ in range(MOVE_ROWS)]
    sources = np.zeros((64, candidate_count), dtype=np.float32)
    targets = np.zeros_like(sources)
    victims = np.zeros_like(sources)
    colors = np.zeros((1, candidate_count), dtype=np.float32)
    for index, (color, source, target, victim) in enumerate(candidates):
        old_rank = 6 if color == 0 else 1
        expected = (channel((target // 8) * 8 + old_rank), channel(victim),
                    PIECE_CHANNELS["BLACK_PAWN" if color == 0 else "WHITE_PAWN"])
        for slot, token in enumerate(expected):
            match_projections[slot][token, index] = 1.0
        sources[source, index] = targets[target, index] = victims[victim, index] = 1.0
        colors[0, index] = color
    previous_match = c.threshold(_sum(c, (
        c.project(rows, projection)
        for rows, projection in zip(previous, match_projections)
    )), 2.5, name="effective_ep_previous_double_push")

    boards = _concat_rows_many(c, prefix_boards, name="effective_ep_prefix_boards")
    piece_planes = {}
    for color in range(2):
        for kind in range(6):
            plane = c.project(boards, _token_projection((96 + color * 6 + kind,)))
            piece_planes[color, kind] = c.reshape(
                plane, state_count, 64, name=f"ep_prefix_piece_{color}_{kind}"
            )
    occupancy = _sum(c, piece_planes.values(), "ep_prefix_occupancy")
    empty = c.reshape(c.project(boards, _token_projection((EMPTY,))), state_count, 64,
                      name="ep_prefix_empty_squares")
    prefix_black = c.constant("side_to_move_by_prefix",
                              (np.arange(state_count) % 2).astype(np.float32)[:, None])
    right_color = _same_bits(c, prefix_black, c.constant("ep_candidate_colors", colors),
                             name="effective_ep_right_color")
    source_pawn = c.add(
        c.matmul(piece_planes[0, 0], c.constant("white_ep_sources", sources * (1.0 - colors))),
        c.matmul(piece_planes[1, 0], c.constant("black_ep_sources", sources * colors)),
        name="effective_ep_source_pawn",
    )
    target_empty = c.matmul(empty, c.constant("effective_ep_targets", targets),
                            name="effective_ep_target_empty")

    own_king = c.select(prefix_black, piece_planes[1, 5], piece_planes[0, 5],
                        name="ep_prefix_moving_king")
    enemy = [c.select(prefix_black, piece_planes[0, kind], piece_planes[1, kind])
             for kind in range(6)]

    repeat = np.zeros((state_count * candidate_count, state_count), dtype=np.float32)
    repeat[np.arange(state_count * candidate_count),
           np.arange(state_count).repeat(candidate_count)] = 1.0
    repeat_value = c.constant("ep_state_candidate_repeat", repeat)
    repeated_occupancy = c.matmul(repeat_value, occupancy)
    repeated_king = c.matmul(repeat_value, own_king)
    repeated_enemy = [c.matmul(repeat_value, plane) for plane in enemy]
    source_masks = c.constant("ep_candidate_source_masks",
                              np.tile(sources.T, (state_count, 1)))
    target_masks = c.constant("ep_candidate_target_masks",
                              np.tile(targets.T, (state_count, 1)))
    victim_masks = c.constant("ep_candidate_victim_masks",
                              np.tile(victims.T, (state_count, 1)))
    removed_source = c.logical_and(source_masks, repeated_occupancy)
    removed_victim = c.logical_and(victim_masks, repeated_occupancy)
    added_target = c.logical_and(target_masks, c.logical_not(repeated_occupancy))
    post_occupancy = _sum(c, (
        repeated_occupancy,
        c.scale(removed_source, -1.0),
        c.scale(removed_victim, -1.0),
        added_target,
    ), "effective_ep_post_occupancy")
    repeated_enemy[0] = c.add(
        repeated_enemy[0],
        c.scale(c.logical_and(victim_masks, repeated_enemy[0]), -1.0),
        name="effective_ep_enemy_pawns_after_capture",
    )

    relations = build_rule_relations()
    geometries = (None, relations.knight, relations.bishop_ray, relations.rook_ray,
                  relations.bishop_ray + relations.rook_ray, relations.king)
    direct, rays = [], []
    for kind in range(6):
        if kind == 0:
            geometry = c.select(
                prefix_black,
                c.project(own_king, relations.pawn_attack[0].T),
                c.project(own_king, relations.pawn_attack[1].T),
            )
        else:
            geometry = c.project(own_king, geometries[kind].T)
        repeated_geometry = c.matmul(repeat_value, geometry)
        attacks = c.logical_and(repeated_geometry, repeated_enemy[kind])
        (direct if kind in (0, 1, 5) else rays).append(attacks)

    # Per-prefix right operands are [blocker, attacker]. Grouped GEMM applies
    # all 28 candidate occupancies without materializing [401,28,64,64].
    between_projection = relations.between.transpose(1, 2, 0).reshape(64, 4096)
    between_by_king = c.reshape(
        c.project(own_king, between_projection), state_count * 64, 64,
        name="effective_ep_between_king_and_attackers",
    )
    blockers = c.grouped_matmul(
        post_occupancy, between_by_king, state_count,
        name="effective_ep_ray_blocker_counts",
    )
    clear_rays = c.logical_not(c.threshold(blockers, .5))
    unsafe = c.threshold(_row_sum(c, _sum(c, (
        *direct,
        *(c.logical_and(ray, clear_rays) for ray in rays),
    ))), .5, name="effective_ep_king_unsafe")
    safe = c.logical_not(c.reshape(unsafe, state_count, candidate_count),
                         name="effective_ep_king_safe")
    legal = _all(c, right_color, previous_match, source_pawn, target_empty, safe,
                 name="effective_legal_ep_candidates")
    ep_targets = c.threshold(c.matmul(legal, c.constant("effective_ep_target_decoder", targets.T)),
                             .5, name="effective_ep_target_key")
    has_ep = c.threshold(_row_sum(c, ep_targets), .5, name="has_effective_legal_ep")
    return ep_targets, has_ep


def _automatic_draws(c: Circuit, history: int, active: int, board: int,
                     generated, position_artifact, position_values):
    prefix_boards = _position_prefixes(c, position_artifact, position_values)
    prefix_flat = c.reshape(_concat_rows_many(c, prefix_boards), 401, 64 * VOCAB_SIZE,
                            name="position_keys_by_prefix")
    current_flat = c.reshape(board, 1, 64 * VOCAB_SIZE)
    board_equal = c.threshold(c.matmul(prefix_flat, c.transpose(current_flat)), 63.5,
                              name="current_board_equal_by_prefix")

    prefix_rights, current_rights = _castling_rights_by_prefix(c, history)
    signed_prefix_rights = c.bias(c.scale(prefix_rights, 2.0), -1.0)
    signed_current_rights = c.bias(c.scale(current_rights, 2.0), -1.0)
    rights_equal = c.threshold(c.matmul(
        signed_prefix_rights, c.transpose(signed_current_rights)), 3.5,
        name="current_castling_rights_equal_by_prefix")
    prefix_black = c.constant("side_to_move_by_prefix",
                              (np.arange(401) % 2).astype(np.float32)[:, None])
    side_equal = _same_bits(c, prefix_black, generated.black_to_move,
                            name="current_side_equal_by_prefix")
    prefix_present = c.concat_rows((
        c.constant("initial_position_present", [[1.0]]), active,
    ), name="present_position_prefixes")
    prefix_ep, prefix_has_ep = _effective_ep_by_prefix(c, history, prefix_boards)
    next_prefix_present = c.concat_rows((
        c.route(prefix_present, 1, 401),
        c.constant("position_prefix_end", [[0.0]]),
    ))
    current_prefix = c.add(prefix_present, c.scale(next_prefix_present, -1.0),
                           name="current_position_prefix")
    current_ep = c.matmul(c.transpose(current_prefix), prefix_ep,
                          name="current_effective_ep_target")
    current_has_ep = c.matmul(c.transpose(current_prefix), prefix_has_ep,
                              name="current_has_effective_ep")
    same_ep_target = c.threshold(c.matmul(prefix_ep, c.transpose(current_ep)), .5)
    both_without_ep = c.logical_not(c.logical_or(prefix_has_ep, current_has_ep))
    ep_equal = c.logical_or(same_ep_target, both_without_ep,
                            name="current_effective_ep_equal_by_prefix")
    current_matches = _all(c, board_equal, rights_equal, side_equal, ep_equal, prefix_present,
                           name="current_transposition_key_matches")
    current_occurrences = _row_sum(c, c.transpose(current_matches),
                                   "current_position_occurrences")
    current_threefold = c.threshold(current_occurrences, 2.5,
                                    name="current_threefold_repetition")

    post_flat, nonzeroing, future_from, future_to = _compact_legal_post_boards(
        c, board, generated.rows)
    future_board_equal = c.threshold(c.matmul(post_flat, c.transpose(prefix_flat)), 63.5,
                                     name="future_board_equal_by_prefix")

    touched_from = np.zeros((VOCAB_SIZE, 4), dtype=np.float32)
    touched_to = np.zeros_like(touched_from)
    for index, (king, rook) in enumerate(((51, 11), (51, 81), (58, 18), (58, 88))):
        touched_from[[king, rook], index] = 1.0
        touched_to[rook, index] = 1.0
    future_touches = c.threshold(c.add(
        c.project(future_from, touched_from), c.project(future_to, touched_to)), .5)
    future_rights = c.logical_and(current_rights, c.logical_not(future_touches),
                                  name="future_castling_rights")
    signed_future_rights = c.bias(c.scale(future_rights, 2.0), -1.0)
    future_rights_equal = c.threshold(c.matmul(
        signed_future_rights, c.transpose(signed_prefix_rights)), 3.5,
        name="future_castling_rights_equal_by_prefix")
    future_black = c.logical_not(generated.black_to_move, name="future_side_to_move")
    future_has_reply = _batched_reply_exists(
        c, post_flat, future_black, future_rights,
    )
    future_side_equal = c.transpose(_same_bits(c, prefix_black, future_black))
    future_matches = _all(
        c,
        future_board_equal,
        future_rights_equal,
        future_side_equal,
        c.transpose(c.logical_not(prefix_has_ep)),
        c.transpose(prefix_present),
        name="future_transposition_key_matches",
    )
    future_occurrences = c.project(
        future_matches, np.ones((401, 1), dtype=np.float32),
        "future_position_prior_occurrences",
    )
    reaches_third = c.threshold(future_occurrences, 1.5)
    repeating_legal = _all(c, reaches_third, generated.present, nonzeroing,
                           name="future_legal_repetition_candidates")
    future_threefold = c.threshold(
        _row_sum(c, c.transpose(repeating_legal)), .5,
        name="next_move_threefold_repetition",
    )

    zeroing = _history_zeroing_moves(c, history, prefix_boards)
    halfmove_clock = _halfmove_clock(c, active, zeroing)
    at_least_99 = c.threshold(halfmove_clock, 98.5, name="halfmove_clock_at_least_99")
    at_least_100 = c.threshold(halfmove_clock, 99.5, name="halfmove_clock_at_least_100")
    nonzeroing_legal = _all(c, nonzeroing, generated.present, future_has_reply,
                            name="nonzeroing_legal_move_with_reply")
    has_nonzeroing_legal = c.threshold(
        _row_sum(c, c.transpose(nonzeroing_legal)), .5,
        name="has_nonzeroing_legal_move",
    )
    fifty_claim = c.logical_or(
        at_least_100,
        c.logical_and(at_least_99, has_nonzeroing_legal),
        name="automatic_fifty_move_claim",
    )
    insufficient = _insufficient_material(c, board)
    draw = c.logical_or(
        insufficient,
        c.logical_or(fifty_claim, c.logical_or(current_threefold, future_threefold)),
        name="automatic_or_material_draw",
    )
    return draw, {
        "halfmove_clock": halfmove_clock,
        "current_position_occurrences": current_occurrences,
        "current_threefold_repetition": current_threefold,
        "next_move_threefold_repetition": future_threefold,
        "future_has_legal_reply": future_has_reply,
        "automatic_fifty_move_claim": fifty_claim,
        "insufficient_material": insufficient,
    }


def build_recurrent_artifact():
    """Return a full recurrent Circuit and its `[B,2045,128]` output value."""

    c = Circuit()
    request = c.route(c.input, 0, MOVE_ROWS, name="requested_move")
    history = c.route(c.input, MOVE_ROWS, MOVE_ROWS + HISTORY_ROWS, name="previous_history")
    legal_start = MOVE_ROWS + HISTORY_ROWS
    incoming_legal = c.route(c.input, legal_start, legal_start + LEGAL_ROWS,
                             name="previous_legal_set")
    status_start = legal_start + LEGAL_ROWS
    incoming_status = c.route(c.input, status_start, status_start + 1,
                              name="previous_status")
    service = c.route(c.input, status_start + 1, 2048, name="reserved_service_rows")

    # A request is accepted only when its exact three one-hot rows equal one of
    # the 256 incoming legal triples. Padding triples are excluded explicitly.
    legal_slots = c.reshape(incoming_legal, 256, MOVE_ROWS * VOCAB_SIZE,
                            name="previous_legal_triples")
    request_flat = c.reshape(request, 1, MOVE_ROWS * VOCAB_SIZE,
                             name="requested_move_flat")
    match_score = c.matmul(legal_slots, c.transpose(request_flat),
                           name="request_legal_match_score")
    exact_match = c.threshold(match_score, 2.5, name="request_exact_legal_match")
    legal_from_rows = c.route(incoming_legal, 0, LEGAL_ROWS, MOVE_ROWS,
                              name="legal_from_rows")
    present = c.project(
        legal_from_rows,
        _token_projection(
            channel for channel in range(VOCAB_SIZE)
            if 1 <= channel // 10 <= 8 and 1 <= channel % 10 <= 8
        ),
        name="legal_slot_present",
    )
    matched = c.logical_and(exact_match, present, name="present_exact_match")
    membership = c.threshold(
        _row_sum(c, c.transpose(matched), "request_match_count"),
        .5,
        name="request_is_legal",
    )

    # History is an inductively trusted prefix of complete triples. The first
    # free triple is selected by a fixed prefix edge, then the request replaces
    # exactly its three PAD rows through outer products.
    history_from = c.route(history, 0, HISTORY_ROWS, MOVE_ROWS,
                           name="history_from_rows")
    active = c.project(history_from, _token_projection(
        channel for channel in range(VOCAB_SIZE)
        if 1 <= channel // 10 <= 8 and 1 <= channel % 10 <= 8
    ), name="active_history_plies")
    active_count = _row_sum(c, c.transpose(active), "history_ply_count")
    full = c.threshold(active_count, 399.5, name="history_is_full")
    terminal = c.threshold(
        c.project(incoming_status, _token_projection((
            STATUS_CHANNELS["WHITE_WIN"],
            STATUS_CHANNELS["BLACK_WIN"],
            STATUS_CHANNELS["DRAW"],
        )), "incoming_terminal_status"),
        .5,
        name="incoming_is_terminal",
    )
    accepted = c.logical_and(
        membership,
        c.logical_and(c.logical_not(full), c.logical_not(terminal)),
        name="request_is_accepted",
    )
    previous_active = c.concat_rows((
        c.constant("history_prefix_origin", [[1.0]]),
        c.route(active, 0, 399),
    ), name="previous_history_activity")
    first_free = c.logical_and(c.logical_not(active), previous_active,
                               name="first_free_history_slot")
    append_slot = c.logical_and(first_free, accepted, name="accepted_append_slot")
    append_payload = c.matmul(append_slot, request_flat, name="appended_move_payload")
    pad_triple = np.zeros((1, MOVE_ROWS * VOCAB_SIZE), dtype=np.float32)
    pad_triple[0, [0, VOCAB_SIZE, 2 * VOCAB_SIZE]] = 1.0
    removed_padding = c.matmul(
        append_slot,
        c.constant("history_pad_triple", pad_triple),
        name="removed_history_padding",
    )
    history_delta = c.reshape(
        c.add(append_payload, c.scale(removed_padding, -1.0)),
        HISTORY_ROWS,
        VOCAB_SIZE,
        name="history_append_delta",
    )
    next_history = c.add(history, history_delta, name="next_history")
    next_active = c.add(active, append_slot, name="accepted_history_activity")

    # The same accepted-history tensor is fed to the already verified position
    # and legal-set circuits. No board or move decoder enters the runtime.
    next_source = c.concat_rows((
        request, next_history, incoming_legal, incoming_status, service,
    ), name="accepted_history_source")
    position_artifact = build_position_reconstruction_artifact()
    position_values = c.include_graph_values(position_artifact, next_source)
    board = position_values[position_artifact.operations[-1].outputs[0]]
    generated = compile_legal_set(c, board, next_source)
    automatic_draw, draw_diagnostics = _automatic_draws(
        c, next_history, next_active, board, generated, position_artifact, position_values,
    )

    # An empty generated set is either mate or stalemate. `attacked_squares`
    # and `king_row` are outputs of the same frozen legality circuit, so winner
    # selection is still tensor arithmetic and contains no runtime chess branch.
    has_legal_reply = c.threshold(generated.count, .5, name="has_legal_reply")
    no_legal_reply = c.logical_not(has_legal_reply, name="no_legal_reply")
    side_to_move_in_check = c.threshold(
        c.matmul(generated.king_row, generated.attacked_squares,
                 name="next_king_attack_score"),
        .5,
        name="next_side_is_in_check",
    )
    checkmate_winner = c.select(
        generated.black_to_move,
        _status(c, STATUS_CHANNELS["WHITE_WIN"], "status_white_win"),
        _status(c, STATUS_CHANNELS["BLACK_WIN"], "status_black_win"),
        name="checkmate_winner_status",
    )
    terminal_board_status = c.select(
        side_to_move_in_check,
        checkmate_winner,
        _status(c, STATUS_CHANNELS["DRAW"], "status_stalemate_draw"),
        name="mate_or_stalemate_status",
    )
    accepted_status = c.select(
        no_legal_reply,
        terminal_board_status,
        c.select(
            automatic_draw,
            _status(c, STATUS_CHANNELS["DRAW"], "status_automatic_draw"),
            _status(c, STATUS_CHANNELS["OK"], "status_ok"),
        ),
        name="accepted_request_status",
    )
    accepted_terminal = c.logical_or(no_legal_reply, automatic_draw,
                                     name="accepted_position_is_terminal")
    empty_legal = np.zeros((LEGAL_ROWS, VOCAB_SIZE), dtype=np.float32)
    empty_legal[:, 0] = 1.0
    accepted_legal = c.select(
        accepted_terminal,
        c.constant("terminal_empty_legal_set", empty_legal),
        generated.rows,
        name="accepted_next_legal_set",
    )
    next_legal = c.select(accepted, accepted_legal, incoming_legal,
                          name="selected_next_legal_set")
    active_status = c.select(
        full,
        _status(c, STATUS_CHANNELS["HISTORY_OVERFLOW"], "status_history_overflow"),
        c.select(
            accepted,
            accepted_status,
            _status(c, STATUS_CHANNELS["ILLEGAL_MOVE"], "status_illegal_move"),
        ),
        name="active_transition_status",
    )
    next_status = c.select(terminal, incoming_status, active_status,
                           name="selected_transition_status")
    output = c.concat_rows((next_history, next_legal, next_status, service),
                           name="next_context")
    c.named_outputs.update(
        request_is_legal=membership,
        request_is_accepted=accepted,
        history_ply_count=active_count,
        history_is_full=full,
        next_history=next_history,
        next_legal_set=next_legal,
        has_legal_reply=has_legal_reply,
        next_side_is_in_check=side_to_move_in_check,
        next_status=next_status,
        next_context=output,
        automatic_claimable_draw=automatic_draw,
        **draw_diagnostics,
    )
    c.tensors.append(exact_record("context_0", build_context_0()))
    return c, output
