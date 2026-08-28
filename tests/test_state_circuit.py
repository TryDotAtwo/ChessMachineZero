import numpy

from vm_compiler.protocol import encode_move_one_hot
from vm_compiler.context import build_context_0
from vm_compiler.state_circuit import (
    EMPTY,
    PIECE_CHANNELS,
    build_initial_piece_state,
    build_initial_workspace_bias,
    build_square_decoder,
    normal_move_transition,
    piece_at,
    WORKSPACE_INPUT_START,
)


def test_initial_piece_state_is_exact_hard_one_hot_standard_position():
    board = build_initial_piece_state()

    assert board.shape == (64, 128)
    assert numpy.array_equal(board.sum(axis=1), numpy.ones(64, dtype=numpy.float32))
    assert piece_at(board, 11) == PIECE_CHANNELS["WHITE_ROOK"]
    assert piece_at(board, 21) == PIECE_CHANNELS["WHITE_KNIGHT"]
    assert piece_at(board, 51) == PIECE_CHANNELS["WHITE_KING"]
    assert piece_at(board, 52) == PIECE_CHANNELS["WHITE_PAWN"]
    assert piece_at(board, 55) == EMPTY
    assert piece_at(board, 57) == PIECE_CHANNELS["BLACK_PAWN"]
    assert piece_at(board, 58) == PIECE_CHANNELS["BLACK_KING"]


def test_square_decoder_maps_native_tokens_to_exact_compact_square_masks():
    decoder = build_square_decoder()
    source = encode_move_one_hot(52, 54, 0)[0]
    mask = source @ decoder

    assert decoder.shape == (128, 64)
    assert mask.sum() == 1.0
    assert numpy.flatnonzero(mask).tolist() == [33]


def test_normal_tensor_transition_moves_and_captures_without_piece_branches():
    board = build_initial_piece_state()
    board = normal_move_transition(board, encode_move_one_hot(52, 54, 0))
    board = normal_move_transition(board, encode_move_one_hot(47, 45, 0))
    board = normal_move_transition(board, encode_move_one_hot(54, 45, 0))

    assert piece_at(board, 52) == EMPTY
    assert piece_at(board, 54) == EMPTY
    assert piece_at(board, 47) == EMPTY
    assert piece_at(board, 45) == PIECE_CHANNELS["WHITE_PAWN"]
    assert numpy.array_equal(board.sum(axis=1), numpy.ones(64, dtype=numpy.float32))
    assert set(numpy.unique(board)).issubset({0.0, 1.0})


def test_frozen_position_bias_injects_board_only_into_hidden_workspace_rows():
    requested = encode_move_one_hot(52, 54, 0)
    vm_input = numpy.concatenate((requested, build_context_0()), axis=0)
    hidden = vm_input + build_initial_workspace_bias()
    workspace = hidden[WORKSPACE_INPUT_START : WORKSPACE_INPUT_START + 64]

    assert numpy.array_equal(workspace, build_initial_piece_state())
    assert numpy.array_equal(hidden[:WORKSPACE_INPUT_START], vm_input[:WORKSPACE_INPUT_START])
    assert numpy.array_equal(hidden[WORKSPACE_INPUT_START + 64 :], vm_input[WORKSPACE_INPUT_START + 64 :])
