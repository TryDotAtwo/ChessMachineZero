import numpy
import pytest

from vm_compiler.protocol import (
    CONTEXT_ROWS,
    HISTORY_ROWS,
    INPUT_ROWS,
    LEGAL_ROWS,
    MOVE_ROWS,
    PIECE_CHANNELS,
    SERVICE_ROWS,
    STATUS_CHANNELS,
    STATUS_ROWS,
    VOCAB_SIZE,
    encode_move_one_hot,
)


def test_e2_e4_is_native_from_to_result_piece_one_hot():
    move = encode_move_one_hot(52, 54, PIECE_CHANNELS["WHITE_PAWN"])

    assert move.shape == (3, 128)
    assert move.dtype == numpy.float32
    assert numpy.flatnonzero(move[0]).tolist() == [52]
    assert numpy.flatnonzero(move[1]).tolist() == [54]
    assert numpy.flatnonzero(move[2]).tolist() == [PIECE_CHANNELS["WHITE_PAWN"]]
    assert move.sum(axis=1).tolist() == [1.0, 1.0, 1.0]


def test_recurrent_tensor_partition_is_exact():
    assert MOVE_ROWS == 3
    assert HISTORY_ROWS == 1200
    assert LEGAL_ROWS == 768
    assert STATUS_ROWS == 1
    assert SERVICE_ROWS == 76
    assert CONTEXT_ROWS == 2045
    assert INPUT_ROWS == 2048
    assert VOCAB_SIZE == 128
    assert MOVE_ROWS + CONTEXT_ROWS == INPUT_ROWS


def test_statuses_are_vocabulary_channels_not_host_enums():
    assert STATUS_CHANNELS == {
        "OK": 89,
        "ILLEGAL_MOVE": 90,
        "WHITE_WIN": 91,
        "BLACK_WIN": 92,
        "DRAW": 93,
        "HISTORY_OVERFLOW": 94,
    }


@pytest.mark.parametrize("square", [10, 19, 20, 29, 90, 99, -1, 128])
def test_non_square_decimal_channels_are_rejected(square):
    with pytest.raises(ValueError, match="source square"):
        encode_move_one_hot(square, 54, PIECE_CHANNELS["WHITE_PAWN"])


@pytest.mark.parametrize("piece", [-1, 0, 5, 95, 108, 127, 128])
def test_only_twelve_result_piece_channels_are_accepted(piece):
    with pytest.raises(ValueError, match="result piece"):
        encode_move_one_hot(52, 54, piece)
