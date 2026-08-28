"""Compile the immutable initial recurrent context from the numeric protocol."""

from __future__ import annotations

import numpy

from .protocol import (
    CONTEXT_ROWS,
    HISTORY_ROWS,
    LEGAL_ROWS,
    PIECE_CHANNELS,
    STATUS_CHANNELS,
    VOCAB_SIZE,
)


def initial_legal_moves() -> list[tuple[int, int, int]]:
    moves = []
    for file_index in range(1, 9):
        source = file_index * 10 + 2
        piece = PIECE_CHANNELS["WHITE_PAWN"]
        moves.extend(((source, source + 1, piece), (source, source + 2, piece)))
    knight = PIECE_CHANNELS["WHITE_KNIGHT"]
    moves.extend(((21, 13, knight), (21, 33, knight), (71, 63, knight), (71, 83, knight)))
    return sorted(moves)


def build_context_0() -> numpy.ndarray:
    context = numpy.zeros((CONTEXT_ROWS, VOCAB_SIZE), dtype=numpy.float32)
    context[:, 0] = 1.0

    legal_start = HISTORY_ROWS
    for move_index, move in enumerate(initial_legal_moves()):
        for slot, channel in enumerate(move):
            row = legal_start + move_index * 3 + slot
            context[row, 0] = 0.0
            context[row, channel] = 1.0

    status_row = HISTORY_ROWS + LEGAL_ROWS
    context[status_row, 0] = 0.0
    context[status_row, STATUS_CHANNELS["OK"]] = 1.0
    return context
