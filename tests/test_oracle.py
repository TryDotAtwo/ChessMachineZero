import numpy
import pytest

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


@pytest.mark.parametrize("fractional_input", ["context", "request"])
def test_fractional_unit_sum_rows_are_not_hard_one_hot(fractional_input):
    context = build_context_0()
    request = encode_move_one_hot(52, 54, WP)
    if fractional_input == "context":
        context[-1, 0] = .75
        context[-1, 1] = .25
    else:
        request[0, 52] = .75
        request[0, 53] = .25
    with pytest.raises(ValueError, match="hard one-hot"):
        transition_oracle(context, request)


_TERMINAL_HISTORIES = [
    ([(62, 63, WP), (57, 55, BP), (72, 74, WP), (48, 84, BQ)], "BLACK_WIN"),
    ([(52, 54, WP), (57, 55, BP), (61, 34, PIECE_CHANNELS["WHITE_BISHOP"]),
      (28, 36, BN), (41, 85, WQ), (78, 66, BN), (85, 67, WQ)], "WHITE_WIN"),
    ([(71, 63, WN), (78, 66, BN), (63, 71, WN), (66, 78, BN),
      (71, 63, WN), (78, 66, BN), (63, 71, WN)], "DRAW"),
]


@pytest.mark.parametrize("moves,status", _TERMINAL_HISTORIES)
def test_completed_draws_and_wins_are_absorbing_with_canonical_status(moves, status):
    context = build_context_0()
    for move in moves:
        context = transition_oracle(context, encode_move_one_hot(*move))
    status_row = HISTORY_ROWS + LEGAL_ROWS
    assert context[status_row, STATUS_CHANNELS[status]] == 1.
    before = context.copy()
    assert channels(before[HISTORY_ROWS:status_row]) == [0] * LEGAL_ROWS
    # f6g8 is legal after the claimable draw fixture, but cannot resume it.
    for move in [(66, 78, BN), (52, 54, WP)]:
        context = transition_oracle(context, encode_move_one_hot(*move))
        assert numpy.array_equal(context[:HISTORY_ROWS], before[:HISTORY_ROWS])
        assert context[status_row, STATUS_CHANNELS[status]] == 1.
        assert channels(context[HISTORY_ROWS:status_row]) == [0] * LEGAL_ROWS
        assert numpy.array_equal(context, before)


@pytest.mark.parametrize("moves,status", _TERMINAL_HISTORIES)
def test_terminal_history_overrides_stale_incoming_status_and_legal_set(moves, status):
    # Construct history independently of transition_oracle, leaving context_0's
    # legal/status rows deliberately stale: history, not those rows, owns status.
    context = build_context_0()
    context[:3 * len(moves)] = numpy.concatenate([encode_move_one_hot(*move) for move in moves])
    result = transition_oracle(context, encode_move_one_hot(66, 78, BN))
    status_row = HISTORY_ROWS + LEGAL_ROWS
    assert numpy.array_equal(result[:HISTORY_ROWS], context[:HISTORY_ROWS])
    assert result[status_row, STATUS_CHANNELS[status]] == 1.
    assert channels(result[HISTORY_ROWS:status_row]) == [0] * LEGAL_ROWS
