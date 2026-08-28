"""Compile the immutable initial recurrent context from the numeric protocol."""

from __future__ import annotations

import numpy

from .protocol import (
    CONTEXT_ROWS,
    HISTORY_ROWS,
    LEGAL_ROWS,
    STATUS_CHANNELS,
    VOCAB_SIZE,
)


def initial_legal_moves() -> list[tuple[int, int, int]]:
    moves = []
    for file_index in range(1, 9):
        source = file_index * 10 + 2
        moves.extend(((source, source + 1, 0), (source, source + 2, 0)))
    moves.extend(((21, 13, 0), (21, 33, 0), (71, 63, 0), (71, 83, 0)))
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
