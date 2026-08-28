import numpy

from vm_compiler.context import build_context_0
from vm_compiler.oracle import transition_oracle
from vm_compiler.protocol import HISTORY_ROWS, LEGAL_ROWS, PIECE_CHANNELS, STATUS_CHANNELS, encode_move_one_hot

WP = PIECE_CHANNELS["WHITE_PAWN"]
WN = PIECE_CHANNELS["WHITE_KNIGHT"]
WQ = PIECE_CHANNELS["WHITE_QUEEN"]
BP = PIECE_CHANNELS["BLACK_PAWN"]
BN = PIECE_CHANNELS["BLACK_KNIGHT"]
BQ = PIECE_CHANNELS["BLACK_QUEEN"]


def channels(rows):
    return numpy.argmax(rows, axis=1).tolist()


def test_e2e4_appends_history_and_emits_exact_sorted_black_legal_set():
    result = transition_oracle(build_context_0(), encode_move_one_hot(52, 54, WP))
    expected_black = [
        (17, 15, BP), (17, 16, BP),
        (27, 25, BP), (27, 26, BP),
        (28, 16, BN), (28, 36, BN),
        (37, 35, BP), (37, 36, BP),
        (47, 45, BP), (47, 46, BP),
        (57, 55, BP), (57, 56, BP),
        (67, 65, BP), (67, 66, BP),
        (77, 75, BP), (77, 76, BP),
        (78, 66, BN), (78, 86, BN),
        (87, 85, BP), (87, 86, BP),
    ]

    assert channels(result[:6]) == [52, 54, WP, 0, 0, 0]
    assert channels(result[HISTORY_ROWS : HISTORY_ROWS + 60]) == [
        channel for move in expected_black for channel in move
    ]
    assert channels(result[HISTORY_ROWS + LEGAL_ROWS : HISTORY_ROWS + LEGAL_ROWS + 1]) == [
        STATUS_CHANNELS["OK"]
    ]


def test_illegal_move_preserves_history_and_legal_set_and_sets_status():
    before = build_context_0()
    result = transition_oracle(before, encode_move_one_hot(52, 55, WP))
    status_row = HISTORY_ROWS + LEGAL_ROWS

    assert numpy.array_equal(result[:status_row], before[:status_row])
    assert channels(result[status_row : status_row + 1]) == [STATUS_CHANNELS["ILLEGAL_MOVE"]]


def test_fools_mate_emits_black_win_and_empty_legal_set():
    context = build_context_0()
    for move in [(62, 63, WP), (57, 55, BP), (72, 74, WP), (48, 84, BQ)]:
        context = transition_oracle(context, encode_move_one_hot(*move))

    legal = context[HISTORY_ROWS : HISTORY_ROWS + LEGAL_ROWS]
    status = context[HISTORY_ROWS + LEGAL_ROWS]
    assert channels(legal) == [0] * LEGAL_ROWS
    assert int(numpy.argmax(status)) == STATUS_CHANNELS["BLACK_WIN"]


def test_legal_geometry_with_wrong_declared_piece_is_illegal():
    result = transition_oracle(build_context_0(), encode_move_one_hot(52, 54, WN))
    status = result[HISTORY_ROWS + LEGAL_ROWS]

    assert int(numpy.argmax(status)) == STATUS_CHANNELS["ILLEGAL_MOVE"]
