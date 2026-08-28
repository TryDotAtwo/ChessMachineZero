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


def _is_square_channel(channel: int) -> bool:
    return (
        isinstance(channel, int)
        and 1 <= channel // 10 <= 8
        and 1 <= channel % 10 <= 8
    )


def encode_move_one_hot(
    source_square: int, destination_square: int, promotion: int = 0
) -> numpy.ndarray:
    """Development helper; production consumes the resulting tensor directly."""

    if not _is_square_channel(source_square):
        raise ValueError(f"invalid source square channel: {source_square}")
    if not _is_square_channel(destination_square):
        raise ValueError(f"invalid destination square channel: {destination_square}")
    if promotion not in range(5):
        raise ValueError(f"invalid promotion channel: {promotion}")

    move = numpy.zeros((MOVE_ROWS, VOCAB_SIZE), dtype=numpy.float32)
    move[0, source_square] = 1.0
    move[1, destination_square] = 1.0
    move[2, promotion] = 1.0
    return move
