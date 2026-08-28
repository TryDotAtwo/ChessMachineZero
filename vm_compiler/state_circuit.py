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


def build_square_decoder() -> numpy.ndarray:
    decoder = numpy.zeros((VOCAB_SIZE, 64), dtype=numpy.float32)
    for file_index in range(1, 9):
        for rank_index in range(1, 9):
            channel = file_index * 10 + rank_index
            decoder[channel, square_index(channel)] = 1.0
    return decoder


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
    event_targets = numpy.concatenate(
        (numpy.eye(64, dtype=numpy.float32), source_targets, destination_targets), axis=0
    )
    empty = numpy.zeros((moves.shape[0], VOCAB_SIZE), dtype=numpy.float32)
    empty[:, EMPTY] = 1.0
    event_values = numpy.concatenate((build_initial_piece_state(), empty, moves[:, 2]), axis=0)
    event_times = numpy.concatenate(
        (
            numpy.zeros(64, dtype=numpy.float32),
            numpy.arange(1, moves.shape[0] + 1, dtype=numpy.float32),
            numpy.arange(1, moves.shape[0] + 1, dtype=numpy.float32),
        )
    )
    matches = numpy.eye(64, dtype=numpy.float32) @ event_targets.T
    scores = numpy.where(matches == 1.0, event_times[None, :], -numpy.inf)
    return event_values[numpy.argmax(scores, axis=1)]


def piece_at(board: numpy.ndarray, square: int) -> int:
    return int(numpy.argmax(board[square_index(square)]))
