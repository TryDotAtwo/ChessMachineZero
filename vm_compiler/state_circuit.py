"""Tensor reference circuits for history-to-position compilation."""

from __future__ import annotations

import numpy

from .protocol import (
    HISTORY_ROWS,
    MOVE_ROWS,
    PIECE_CHANNELS,
    VOCAB_SIZE,
)
from .relations import square_index

EMPTY = 95
CASTLING_EVENTS = (
    (51, 71, PIECE_CHANNELS["WHITE_KING"], 81, EMPTY, 61, PIECE_CHANNELS["WHITE_ROOK"]),
    (51, 31, PIECE_CHANNELS["WHITE_KING"], 11, EMPTY, 41, PIECE_CHANNELS["WHITE_ROOK"]),
    (58, 78, PIECE_CHANNELS["BLACK_KING"], 88, EMPTY, 68, PIECE_CHANNELS["BLACK_ROOK"]),
    (58, 38, PIECE_CHANNELS["BLACK_KING"], 18, EMPTY, 48, PIECE_CHANNELS["BLACK_ROOK"]),
)


def build_square_decoder() -> numpy.ndarray:
    decoder = numpy.zeros((VOCAB_SIZE, 64), dtype=numpy.float32)
    for file_index in range(1, 9):
        for rank_index in range(1, 9):
            channel = file_index * 10 + rank_index
            decoder[channel, square_index(channel)] = 1.0
    return decoder


def build_castling_derived_events() -> numpy.ndarray:
    table = numpy.zeros((4, 7, VOCAB_SIZE), dtype=numpy.float32)
    for pattern_index, pattern in enumerate(CASTLING_EVENTS):
        table[pattern_index, numpy.arange(7), pattern] = 1.0
    return table


def build_initial_piece_state() -> numpy.ndarray:
    board = numpy.zeros((64, VOCAB_SIZE), dtype=numpy.float32)
    board[:, EMPTY] = 1.0
    back_rank = ("ROOK", "KNIGHT", "BISHOP", "QUEEN", "KING", "BISHOP", "KNIGHT", "ROOK")
    for file_index, piece in enumerate(back_rank, start=1):
        placements = (
            (file_index * 10 + 1, PIECE_CHANNELS[f"WHITE_{piece}"]),
            (file_index * 10 + 2, PIECE_CHANNELS["WHITE_PAWN"]),
            (file_index * 10 + 7, PIECE_CHANNELS["BLACK_PAWN"]),
            (file_index * 10 + 8, PIECE_CHANNELS[f"BLACK_{piece}"]),
        )
        for square, channel in placements:
            row = square_index(square)
            board[row, EMPTY] = 0.0
            board[row, channel] = 1.0
    return board


def normal_move_transition(board: numpy.ndarray, move: numpy.ndarray) -> numpy.ndarray:
    state = numpy.asarray(board, dtype=numpy.float32)
    request = numpy.asarray(move, dtype=numpy.float32)
    if state.shape != (64, VOCAB_SIZE) or request.shape != (MOVE_ROWS, VOCAB_SIZE):
        raise ValueError("normal transition tensor shapes are invalid")
    decoder = build_square_decoder()
    source = request[0] @ decoder
    destination = request[1] @ decoder
    if float(source @ destination) != 0.0:
        raise ValueError("normal transition source and destination must differ")
    moving_piece = request[2]
    empty_piece = numpy.zeros(VOCAB_SIZE, dtype=numpy.float32)
    empty_piece[EMPTY] = 1.0
    keep = 1.0 - source - destination
    return (
        keep[:, None] * state
        + source[:, None] * empty_piece[None, :]
        + destination[:, None] * moving_piece[None, :]
    ).astype(numpy.float32)


def reconstruct_position_from_history(history: numpy.ndarray) -> numpy.ndarray:
    """Dense development oracle for one latest-event hard-attention circuit."""

    rows = numpy.asarray(history, dtype=numpy.float32)
    if rows.shape != (HISTORY_ROWS, VOCAB_SIZE):
        raise ValueError("history must have shape [1200,128]")
    moves = rows.reshape(-1, MOVE_ROWS, VOCAB_SIZE)
    active = moves[:, 0, 0] == 0.0
    moves = moves[active]
    decoder = build_square_decoder()
    source_targets = moves[:, 0] @ decoder
    destination_targets = moves[:, 1] @ decoder
    derived_targets, derived_values, derived_times = _derive_special_events(moves)
    event_targets = numpy.concatenate(
        (
            numpy.eye(64, dtype=numpy.float32),
            source_targets,
            destination_targets,
            derived_targets,
        ),
        axis=0,
    )
    empty = numpy.zeros((moves.shape[0], VOCAB_SIZE), dtype=numpy.float32)
    empty[:, EMPTY] = 1.0
    event_values = numpy.concatenate(
        (build_initial_piece_state(), empty, moves[:, 2], derived_values), axis=0
    )
    event_times = numpy.concatenate(
        (
            numpy.zeros(64, dtype=numpy.float32),
            numpy.arange(1, moves.shape[0] + 1, dtype=numpy.float32),
            numpy.arange(1, moves.shape[0] + 1, dtype=numpy.float32),
            derived_times,
        )
    )
    matches = numpy.eye(64, dtype=numpy.float32) @ event_targets.T
    scores = numpy.where(matches == 1.0, event_times[None, :], -numpy.inf)
    return event_values[numpy.argmax(scores, axis=1)]


def _derive_special_events(moves: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    empty_value = numpy.zeros(VOCAB_SIZE, dtype=numpy.float32)
    empty_value[EMPTY] = 1.0
    targets: list[numpy.ndarray] = []
    values: list[numpy.ndarray] = []
    times: list[float] = []

    castle_events = {
        tuple(pattern[:3]): ((pattern[3], pattern[4]), (pattern[5], pattern[6]))
        for pattern in CASTLING_EVENTS
    }
    decoded = numpy.argmax(moves, axis=2)
    for move_index, native in enumerate(decoded):
        event_time = float(move_index + 1)
        for square, piece in castle_events.get(tuple(int(x) for x in native), ()):
            targets.append(numpy.eye(64, dtype=numpy.float32)[square_index(square)])
            value = empty_value.copy()
            if piece != EMPTY:
                value[:] = 0.0
                value[piece] = 1.0
            values.append(value)
            times.append(event_time)

        if move_index == 0:
            continue
        source, destination, piece = (int(x) for x in native)
        previous_source, previous_destination, previous_piece = (
            int(x) for x in decoded[move_index - 1]
        )
        direction = 1 if piece == PIECE_CHANNELS["WHITE_PAWN"] else -1
        opponent = (
            PIECE_CHANNELS["BLACK_PAWN"]
            if direction == 1
            else PIECE_CHANNELS["WHITE_PAWN"]
        )
        is_diagonal_pawn = (
            piece in (PIECE_CHANNELS["WHITE_PAWN"], PIECE_CHANNELS["BLACK_PAWN"])
            and abs(source // 10 - destination // 10) == 1
            and destination % 10 - source % 10 == direction
        )
        previous_was_matching_double = (
            previous_piece == opponent
            and previous_destination // 10 == destination // 10
            and previous_destination % 10 == source % 10
            and abs(previous_destination % 10 - previous_source % 10) == 2
        )
        if is_diagonal_pawn and previous_was_matching_double:
            captured = previous_destination
            targets.append(numpy.eye(64, dtype=numpy.float32)[square_index(captured)])
            values.append(empty_value.copy())
            times.append(event_time)

    if not targets:
        return (
            numpy.zeros((0, 64), dtype=numpy.float32),
            numpy.zeros((0, VOCAB_SIZE), dtype=numpy.float32),
            numpy.zeros(0, dtype=numpy.float32),
        )
    return (
        numpy.stack(targets),
        numpy.stack(values),
        numpy.asarray(times, dtype=numpy.float32),
    )


def piece_at(board: numpy.ndarray, square: int) -> int:
    return int(numpy.argmax(board[square_index(square)]))
