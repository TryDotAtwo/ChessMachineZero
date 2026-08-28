import numpy

from vm_compiler.protocol import encode_move_one_hot
from vm_compiler.state_circuit import (
    EMPTY,
    PIECE_CHANNELS,
    build_initial_piece_state,
    build_square_decoder,
    normal_move_transition,
    piece_at,
    reconstruct_position_from_history,
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
    source = encode_move_one_hot(52, 54, PIECE_CHANNELS["WHITE_PAWN"])[0]
    mask = source @ decoder

    assert decoder.shape == (128, 64)
    assert mask.sum() == 1.0
    assert numpy.flatnonzero(mask).tolist() == [33]


def test_normal_tensor_transition_moves_and_captures_without_piece_branches():
    board = build_initial_piece_state()
    board = normal_move_transition(board, encode_move_one_hot(52, 54, PIECE_CHANNELS["WHITE_PAWN"]))
    board = normal_move_transition(board, encode_move_one_hot(47, 45, PIECE_CHANNELS["BLACK_PAWN"]))
    board = normal_move_transition(board, encode_move_one_hot(54, 45, PIECE_CHANNELS["WHITE_PAWN"]))

    assert piece_at(board, 52) == EMPTY
    assert piece_at(board, 54) == EMPTY
    assert piece_at(board, 47) == EMPTY
    assert piece_at(board, 45) == PIECE_CHANNELS["WHITE_PAWN"]
    assert numpy.array_equal(board.sum(axis=1), numpy.ones(64, dtype=numpy.float32))
    assert set(numpy.unique(board)).issubset({0.0, 1.0})


def test_one_latest_event_attention_reconstructs_position_from_history():
    moves = (
        encode_move_one_hot(52, 54, PIECE_CHANNELS["WHITE_PAWN"]),
        encode_move_one_hot(47, 45, PIECE_CHANNELS["BLACK_PAWN"]),
        encode_move_one_hot(54, 45, PIECE_CHANNELS["WHITE_PAWN"]),
    )
    history = numpy.zeros((1200, 128), dtype=numpy.float32)
    history[:, 0] = 1.0
    history[:9] = numpy.concatenate(moves)

    board = reconstruct_position_from_history(history)

    assert piece_at(board, 52) == EMPTY
    assert piece_at(board, 54) == EMPTY
    assert piece_at(board, 47) == EMPTY
    assert piece_at(board, 45) == PIECE_CHANNELS["WHITE_PAWN"]
    assert numpy.array_equal(board.sum(axis=1), numpy.ones(64, dtype=numpy.float32))


def test_castling_move_derives_rook_events_for_latest_event_attention():
    history = numpy.zeros((1200, 128), dtype=numpy.float32)
    history[:, 0] = 1.0
    history[:3] = encode_move_one_hot(51, 71, PIECE_CHANNELS["WHITE_KING"])

    board = reconstruct_position_from_history(history)

    assert piece_at(board, 51) == EMPTY
    assert piece_at(board, 71) == PIECE_CHANNELS["WHITE_KING"]
    assert piece_at(board, 81) == EMPTY
    assert piece_at(board, 61) == PIECE_CHANNELS["WHITE_ROOK"]


def test_en_passant_derives_captured_pawn_clear_event():
    moves = (
        encode_move_one_hot(52, 54, PIECE_CHANNELS["WHITE_PAWN"]),
        encode_move_one_hot(17, 16, PIECE_CHANNELS["BLACK_PAWN"]),
        encode_move_one_hot(54, 55, PIECE_CHANNELS["WHITE_PAWN"]),
        encode_move_one_hot(47, 45, PIECE_CHANNELS["BLACK_PAWN"]),
        encode_move_one_hot(55, 46, PIECE_CHANNELS["WHITE_PAWN"]),
    )
    history = numpy.zeros((1200, 128), dtype=numpy.float32)
    history[:, 0] = 1.0
    history[:15] = numpy.concatenate(moves)

    board = reconstruct_position_from_history(history)

    assert piece_at(board, 55) == EMPTY
    assert piece_at(board, 45) == EMPTY
    assert piece_at(board, 46) == PIECE_CHANNELS["WHITE_PAWN"]
