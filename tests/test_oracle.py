import numpy

from vm_compiler.context import build_context_0
from vm_compiler.oracle import transition_oracle
from vm_compiler.protocol import HISTORY_ROWS, LEGAL_ROWS, STATUS_CHANNELS, encode_move_one_hot


def channels(rows):
    return numpy.argmax(rows, axis=1).tolist()


def test_e2e4_appends_history_and_emits_exact_sorted_black_legal_set():
    result = transition_oracle(build_context_0(), encode_move_one_hot(52, 54, 0))
    expected_black = [
        (17, 15, 0), (17, 16, 0),
        (27, 25, 0), (27, 26, 0),
        (28, 16, 0), (28, 36, 0),
        (37, 35, 0), (37, 36, 0),
        (47, 45, 0), (47, 46, 0),
        (57, 55, 0), (57, 56, 0),
        (67, 65, 0), (67, 66, 0),
        (77, 75, 0), (77, 76, 0),
        (78, 66, 0), (78, 86, 0),
        (87, 85, 0), (87, 86, 0),
    ]

    assert channels(result[:6]) == [52, 54, 0, 0, 0, 0]
    assert channels(result[HISTORY_ROWS : HISTORY_ROWS + 60]) == [
        channel for move in expected_black for channel in move
    ]
    assert channels(result[HISTORY_ROWS + LEGAL_ROWS : HISTORY_ROWS + LEGAL_ROWS + 1]) == [
        STATUS_CHANNELS["OK"]
    ]


def test_illegal_move_preserves_history_and_legal_set_and_sets_status():
    before = build_context_0()
    result = transition_oracle(before, encode_move_one_hot(52, 55, 0))
    status_row = HISTORY_ROWS + LEGAL_ROWS

    assert numpy.array_equal(result[:status_row], before[:status_row])
    assert channels(result[status_row : status_row + 1]) == [STATUS_CHANNELS["ILLEGAL_MOVE"]]


def test_fools_mate_emits_black_win_and_empty_legal_set():
    context = build_context_0()
    for move in [(62, 63, 0), (57, 55, 0), (72, 74, 0), (48, 84, 0)]:
        context = transition_oracle(context, encode_move_one_hot(*move))

    legal = context[HISTORY_ROWS : HISTORY_ROWS + LEGAL_ROWS]
    status = context[HISTORY_ROWS + LEGAL_ROWS]
    assert channels(legal) == [0] * LEGAL_ROWS
    assert int(numpy.argmax(status)) == STATUS_CHANNELS["BLACK_WIN"]
