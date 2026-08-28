import numpy

from vm_compiler.context import build_context_0, initial_legal_moves
from vm_compiler.protocol import HISTORY_ROWS, LEGAL_ROWS, PIECE_CHANNELS, STATUS_CHANNELS

WP = PIECE_CHANNELS["WHITE_PAWN"]
WN = PIECE_CHANNELS["WHITE_KNIGHT"]


EXPECTED_INITIAL_LEGAL = [
    (12, 13, WP), (12, 14, WP),
    (21, 13, WN), (21, 33, WN),
    (22, 23, WP), (22, 24, WP),
    (32, 33, WP), (32, 34, WP),
    (42, 43, WP), (42, 44, WP),
    (52, 53, WP), (52, 54, WP),
    (62, 63, WP), (62, 64, WP),
    (71, 63, WN), (71, 83, WN),
    (72, 73, WP), (72, 74, WP),
    (82, 83, WP), (82, 84, WP),
]


def active_channels(rows):
    return numpy.argmax(rows, axis=1).tolist()


def test_initial_legal_set_is_exact_native_sorted_move_language():
    assert initial_legal_moves() == EXPECTED_INITIAL_LEGAL


def test_context_0_is_a_complete_hard_one_hot_recurrent_tensor():
    context = build_context_0()

    assert context.shape == (2045, 128)
    assert context.dtype == numpy.float32
    assert numpy.array_equal(context.sum(axis=1), numpy.ones(2045, dtype=numpy.float32))
    assert set(numpy.unique(context)).issubset({0.0, 1.0})


def test_context_0_contains_history_legal_status_and_service_without_board_rows():
    context = build_context_0()
    legal_start = HISTORY_ROWS
    status_row = HISTORY_ROWS + LEGAL_ROWS

    assert active_channels(context[:HISTORY_ROWS]) == [0] * HISTORY_ROWS
    assert active_channels(context[legal_start : legal_start + 60]) == [
        channel for move in EXPECTED_INITIAL_LEGAL for channel in move
    ]
    assert active_channels(context[legal_start + 60 : status_row]) == [0] * (LEGAL_ROWS - 60)
    assert active_channels(context[status_row : status_row + 1]) == [STATUS_CHANNELS["OK"]]
    assert active_channels(context[status_row + 1 :]) == [0] * 76
