"""Tensor reference circuits for history-to-position compilation."""

from __future__ import annotations

import numpy

from .protocol import (
    HISTORY_ROWS,
    INPUT_ROWS,
    LEGAL_ROWS,
    MOVE_ROWS,
    STATUS_ROWS,
    VOCAB_SIZE,
)
from .relations import square_index

EMPTY = 95
WORKSPACE_INPUT_START = MOVE_ROWS + HISTORY_ROWS + LEGAL_ROWS + STATUS_ROWS
PIECE_CHANNELS = {
    "WHITE_PAWN": 96,
    "WHITE_KNIGHT": 97,
    "WHITE_BISHOP": 98,
    "WHITE_ROOK": 99,
    "WHITE_QUEEN": 100,
    "WHITE_KING": 101,
    "BLACK_PAWN": 102,
    "BLACK_KNIGHT": 103,
    "BLACK_BISHOP": 104,
    "BLACK_ROOK": 105,
    "BLACK_QUEEN": 106,
    "BLACK_KING": 107,
}


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


def build_initial_workspace_bias() -> numpy.ndarray:
    """Frozen additive injection of the initial board into hidden service rows."""

    bias = numpy.zeros((INPUT_ROWS, VOCAB_SIZE), dtype=numpy.float32)
    workspace = slice(WORKSPACE_INPUT_START, WORKSPACE_INPUT_START + 64)
    bias[workspace, 0] = -1.0
    bias[workspace] += build_initial_piece_state()
    return bias


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
    moving_piece = source @ state
    empty_piece = numpy.zeros(VOCAB_SIZE, dtype=numpy.float32)
    empty_piece[EMPTY] = 1.0
    keep = 1.0 - source - destination
    return (
        keep[:, None] * state
        + source[:, None] * empty_piece[None, :]
        + destination[:, None] * moving_piece[None, :]
    ).astype(numpy.float32)


def piece_at(board: numpy.ndarray, square: int) -> int:
    return int(numpy.argmax(board[square_index(square)]))
