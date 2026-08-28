"""Fixed numeric language shared by players and the frozen VM."""

from __future__ import annotations

import numpy

VOCAB_SIZE = 128
MOVE_ROWS = 3
HISTORY_ROWS = 1200
LEGAL_ROWS = 768
STATUS_ROWS = 1
SERVICE_ROWS = 76
CONTEXT_ROWS = HISTORY_ROWS + LEGAL_ROWS + STATUS_ROWS + SERVICE_ROWS
INPUT_ROWS = MOVE_ROWS + CONTEXT_ROWS

STATUS_CHANNELS = {
    "OK": 89,
    "ILLEGAL_MOVE": 90,
    "WHITE_WIN": 91,
    "BLACK_WIN": 92,
    "DRAW": 93,
    "HISTORY_OVERFLOW": 94,
}
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


def _is_square_channel(channel: int) -> bool:
    return (
        isinstance(channel, int)
        and 1 <= channel // 10 <= 8
        and 1 <= channel % 10 <= 8
    )


def encode_move_one_hot(
    source_square: int, destination_square: int, result_piece: int
) -> numpy.ndarray:
    """Development helper; production consumes the resulting tensor directly."""

    if not _is_square_channel(source_square):
        raise ValueError(f"invalid source square channel: {source_square}")
    if not _is_square_channel(destination_square):
        raise ValueError(f"invalid destination square channel: {destination_square}")
    if result_piece not in PIECE_CHANNELS.values():
        raise ValueError(f"invalid result piece channel: {result_piece}")

    move = numpy.zeros((MOVE_ROWS, VOCAB_SIZE), dtype=numpy.float32)
    move[0, source_square] = 1.0
    move[1, destination_square] = 1.0
    move[2, result_piece] = 1.0
    return move
