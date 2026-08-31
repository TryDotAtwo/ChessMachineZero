"""Compile reusable legal-move rules to generic frozen tensor operations.

This is a legal-set subgraph, NOT a recurrent transition or terminal adjudicator.
History must be valid. Python enumerates only position-independent geometry;
occupancy, turn, rights, king safety and stable compaction execute as tensors.
"""

from dataclasses import dataclass

import numpy as np

from .circuit import Circuit
from .compaction import stable_compact
from .compiler import build_position_reconstruction_artifact
from .relations import build_rule_relations


def _channel(square):
    return (square // 8 + 1) * 10 + square % 8 + 1


@dataclass(frozen=True)
class Candidate:
    source: int
    target: int
    kind: int
    result: int
    color: int
    mode: str = "ordinary"
    rook_source: int = -1
    rook_target: int = -1
    victim: int = -1


@dataclass(frozen=True)
class LegalSet:
    rows: int
    count: int
    overflow: int
    present: int
    black_to_move: int
    king_row: int
    attacked_squares: int


def _candidates(relations):
    candidates = []
    for color in range(2):
        for kind, geometry in ((1, relations.knight), (2, relations.bishop_ray),
                               (3, relations.rook_ray), (4, relations.bishop_ray + relations.rook_ray),
                               (5, relations.king)):
            for source, target in np.argwhere(geometry):
                candidates.append(Candidate(int(source), int(target), kind, kind, color))
        for mode, geometry in (("quiet", relations.pawn_step[color] + relations.pawn_double[color]),
                               ("capture", relations.pawn_attack[color])):
            for source, target in np.argwhere(geometry):
                if source % 8 not in range(1, 7):
                    continue
                results = (1, 2, 3, 4) if target % 8 in (0, 7) else (0,)
                candidates.extend(Candidate(int(source), int(target), 0, result, color, mode) for result in results)
        rank = 4 if color == 0 else 3
        direction = 1 if color == 0 else -1
        for file in range(8):
            for target_file in (file - 1, file + 1):
                if 0 <= target_file < 8:
                    candidates.append(Candidate(file * 8 + rank, target_file * 8 + rank + direction,
                                                0, 0, color, "ep", victim=target_file * 8 + rank))
        rank = 0 if color == 0 else 7
        for king_file, rook_file, rook_target_file in ((2, 0, 3), (6, 7, 5)):
            candidates.append(Candidate(4 * 8 + rank, king_file * 8 + rank, 5, 5, color,
                                        "castle", rook_file * 8 + rank, rook_target_file * 8 + rank))
    # Duplicate triples can have mutually exclusive source kinds (queen move vs
    # pawn promotion). Only one can survive for a valid board.
    return tuple(sorted(candidates, key=lambda c: (_channel(c.source), _channel(c.target),
                                                   96 + 6 * c.color + c.result, c.kind, c.mode)))


def _sum(c, values, name=None):
    values = tuple(values)
    value = values[0]
    for next_value in values[1:]:
        value = c.add(value, next_value, name)
    return value


def _column(c, value, index, name=None):
    weights = np.zeros((c.shapes[value][1], 1), dtype=np.float32)
    weights[index] = 1
    return c.project(value, weights, name)


def _rowsum(c, value, name=None):
    return c.project(value, np.ones((c.shapes[value][1], 1)), name)


def _all(c, *values):
    return c.threshold(_sum(c, values), len(values) - .5)


def build_legal_artifact():
    c = Circuit()
    board = c.include_graph(build_position_reconstruction_artifact())
    result = compile_legal_set(c, board, c.input)
    c.named_outputs.update(legal_set=result.rows, legal_count=result.count,
                           legal_capacity_overflow=result.overflow, legal_slots_present=result.present,
                           black_to_move=result.black_to_move)
    return c, result.rows


def compile_legal_set(c: Circuit, board: int, source: int):
    relations = build_rule_relations()
    candidates = _candidates(relations)
    count = len(candidates)
    eye = np.eye(64, dtype=np.float32)
    sources = np.stack([eye[p.source] for p in candidates])
    targets = np.stack([eye[p.target] for p in candidates])
    source_matrix = c.constant("candidate_source_squares", sources)
    target_matrix = c.constant("candidate_target_squares", targets)
    flags = {mode: np.array([[float(p.mode == mode)] for p in candidates], dtype=np.float32)
             for mode in ("ordinary", "quiet", "capture", "ep", "castle")}
    is_ep = c.constant("en_passant_candidates", flags["ep"])
    not_castle = c.constant("non_castling_candidates", 1 - flags["castle"])

    history = tuple(c.route(source, 3 + slot, 1203 + slot, 3) for slot in range(3))
    occupied_token = np.zeros((128, 1), dtype=np.float32)
    occupied_token[[_channel(s) for s in range(64)]] = 1
    active = c.project(history[0], occupied_token, "active_history_plies")
    parity = np.array([[1 if index % 2 == 0 else -1] for index in range(400)], dtype=np.float32)
    black = c.project(c.transpose(active), parity, "black_to_move")
    candidate_colors = np.array([[p.color] for p in candidates], dtype=np.float32)
    right_color = c.select(black, c.constant("black_candidates", candidate_colors),
                           c.constant("white_candidates", 1 - candidate_colors))

    piece_sets = []
    for color in range(2):
        projection = np.zeros((128, 6), dtype=np.float32)
        projection[96 + color * 6:102 + color * 6] = np.eye(6)
        piece_sets.append(c.project(board, projection))
    own = c.select(black, piece_sets[1], piece_sets[0], "side_to_move_pieces")
    enemy = c.select(black, piece_sets[0], piece_sets[1], "opponent_pieces")
    own_occupancy, enemy_occupancy = _rowsum(c, own), _rowsum(c, enemy)
    occupancy = c.add(own_occupancy, enemy_occupancy, "board_occupancy")
    own_king = _column(c, own, 5, "moving_side_king")
    enemy_columns = tuple(c.transpose(_column(c, enemy, kind)) for kind in range(6))
    king_row = c.transpose(own_king)

    selected_sources = c.matmul(source_matrix, own)
    required_kind = np.eye(6, dtype=np.float32)[[p.kind for p in candidates]]
    source_matches = _rowsum(c, c.logical_and(selected_sources, c.constant("required_source_kind", required_kind)))
    target_own = c.matmul(target_matrix, own_occupancy)
    target_enemy = c.matmul(target_matrix, enemy_occupancy)
    target_king = c.matmul(target_matrix, _column(c, enemy, 5))
    target_empty = c.logical_not(c.add(target_own, target_enemy))

    paths = np.stack([relations.between[p.source, p.target].copy() for p in candidates])
    for index, p in enumerate(candidates):
        if p.mode == "castle":
            paths[index] = relations.between[p.source, p.rook_source]
    clear_path = c.logical_not(c.threshold(c.matmul(c.constant("candidate_between_squares", paths), occupancy), .5))

    # Last used history triple; no gather/argmax index leaves the tensor graph.
    future_active = c.concat_rows((c.route(active, 1, 400), c.constant("zero_row", [[0.]])))
    last_mask = c.add(active, c.scale(future_active, -1.))
    previous = tuple(c.matmul(c.transpose(last_mask), rows) for rows in history)
    previous_projections = [np.zeros((128, count), dtype=np.float32) for _ in range(3)]
    ep_victims = np.zeros((count, 64), dtype=np.float32)
    for index, p in enumerate(candidates):
        if p.mode == "ep":
            old_rank = 6 if p.color == 0 else 1
            previous_triple = (_channel((p.target // 8) * 8 + old_rank), _channel(p.victim),
                               102 if p.color == 0 else 96)
            for slot, channel in enumerate(previous_triple):
                previous_projections[slot][channel, index] = 1
            ep_victims[index, p.victim] = 1
    ep_match = c.threshold(c.transpose(_sum(c, (
        c.project(value, projection) for value, projection in zip(previous, previous_projections)))), 2.5)
    pawn_target = _sum(c, (
        c.logical_and(c.constant("pawn_quiet_candidates", flags["quiet"]), target_empty),
        c.logical_and(c.constant("pawn_capture_candidates", flags["capture"]), target_enemy),
        _all(c, is_ep, ep_match, target_empty),
        c.constant("non_pawn_candidates", flags["ordinary"] + flags["castle"]),
    ))

    # Rights are lost after a king/rook source event or a capture on its corner.
    castles = [p for p in candidates if p.mode == "castle"]
    touched_from = np.zeros((128, 4), dtype=np.float32)
    touched_to = np.zeros_like(touched_from)
    castle_routes = np.zeros((4, count), dtype=np.float32)
    rook_sources = np.zeros((count, 64), dtype=np.float32)
    rook_targets = np.zeros_like(rook_sources)
    castle_king_path = np.zeros_like(rook_sources)
    for castle_index, castle in enumerate(castles):
        touched_from[[_channel(castle.source), _channel(castle.rook_source)], castle_index] = 1
        touched_to[_channel(castle.rook_source), castle_index] = 1
    for index, p in enumerate(candidates):
        if p.mode == "castle":
            castle_routes[castles.index(p), index] = 1
            rook_sources[index, p.rook_source] = rook_targets[index, p.rook_target] = 1
            castle_king_path[index] = relations.between[p.source, p.target]
            castle_king_path[index, [p.source, p.target]] = 1
    touched = c.add(c.project(history[0], touched_from), c.project(history[1], touched_to))
    touches = c.transpose(c.project(c.transpose(touched), np.ones((400, 1))))
    rights = c.logical_not(c.threshold(c.transpose(c.project(touches, castle_routes)), .5))
    rook_present = c.add(c.matmul(c.constant("castling_rook_sources", rook_sources), _column(c, own, 3)), not_castle)

    # Invalid candidates may have meaningless occupancy; the source/geometry
    # gates reject them. Eligible candidates get exact ep/rook relocation too.
    post_occupancy = _sum(c, (
        c.transpose(occupancy), c.constant("clear_candidate_source", -sources),
        c.logical_and(target_empty, target_matrix), c.constant("clear_ep_victim", -ep_victims),
        c.constant("castling_rook_delta", rook_targets - rook_sources),
    ), "candidate_post_move_occupancy")
    enemy_alive = c.constant("uncaptured_enemy_squares", 1 - targets - ep_victims)
    between_king = c.reshape(c.project(king_row, relations.between.transpose(1, 0, 2).reshape(64, 4096)), 64, 64)
    clear_king_rays = c.logical_not(c.threshold(c.matmul(post_occupancy, c.transpose(between_king)), .5))

    geometries = (None, relations.knight, relations.bishop_ray, relations.rook_ray,
                  relations.bishop_ray + relations.rook_ray, relations.king)
    single_attacks, all_attacks = [], []
    for kind in range(6):
        if kind == 0:
            single_geometry = c.select(black, c.project(king_row, relations.pawn_attack[0].T),
                                       c.project(king_row, relations.pawn_attack[1].T))
            all_geometry = c.select(black, c.constant("white_pawn_attack_geometry", relations.pawn_attack[0].T),
                                    c.constant("black_pawn_attack_geometry", relations.pawn_attack[1].T))
        else:
            single_geometry = c.project(king_row, geometries[kind].T)
            all_geometry = c.constant(f"attack_geometry_kind_{kind}", geometries[kind].T)
        single_attacks.append(c.logical_and(single_geometry, enemy_columns[kind]))
        all_attacks.append(c.logical_and(all_geometry, enemy_columns[kind]))
    direct = _sum(c, (single_attacks[kind] for kind in (0, 1, 5)))
    rays = _sum(c, (single_attacks[kind] for kind in (2, 3, 4)))
    unsafe = c.threshold(_rowsum(c, c.logical_and(
        enemy_alive, c.add(direct, c.logical_and(rays, clear_king_rays)))), .5)
    ordinary_safe = c.logical_not(unsafe)

    # Remove the old king square before testing destinations. Captured pieces
    # never attack their own square; ray interiors exclude the target endpoint.
    without_king = c.add(occupancy, c.scale(own_king, -1.))
    all_blockers = c.reshape(c.project(c.transpose(without_king),
                                      relations.between.transpose(2, 1, 0).reshape(64, 4096)), 64, 64)
    all_clear = c.logical_not(c.threshold(all_blockers, .5))
    all_direct = _sum(c, (all_attacks[kind] for kind in (0, 1, 5)))
    all_rays = _sum(c, (all_attacks[kind] for kind in (2, 3, 4)))
    attacked = c.threshold(_rowsum(c, c.add(all_direct, c.logical_and(all_rays, all_clear))),
                           .5, name="opponent_attacked_squares_without_king")
    king_safe = c.logical_not(c.matmul(target_matrix, attacked))
    is_king = c.constant("king_candidates", np.array([[p.kind == 5] for p in candidates], dtype=np.float32))
    safe = c.select(is_king, king_safe, ordinary_safe)
    castle_safe = c.logical_not(c.threshold(c.matmul(
        c.constant("castling_king_path", castle_king_path), attacked), .5))

    legal = _all(c, right_color, source_matches, c.logical_not(target_own), c.logical_not(target_king),
                 clear_path, pawn_target, rights, rook_present, safe, castle_safe)
    payload = np.zeros((count, 384), dtype=np.float32)
    for index, p in enumerate(candidates):
        payload[index, _channel(p.source)] = 1
        payload[index, 128 + _channel(p.target)] = 1
        payload[index, 256 + 96 + 6 * p.color + p.result] = 1
    compacted = stable_compact(c, legal, c.constant("candidate_move_triples", payload), capacity=256)
    # Unused slots contain three PAD rows, not all-zero vocabulary vectors.
    pad = np.zeros((1, 384), dtype=np.float32)
    pad[0, [0, 128, 256]] = 1
    padding = c.project(c.logical_not(compacted.present), pad, "unused_slot_padding")
    rows = c.reshape(c.add(compacted.rows, padding), 768, 128, "legal_set")
    return LegalSet(rows, compacted.count, compacted.overflow, compacted.present, black, king_row, attacked)
