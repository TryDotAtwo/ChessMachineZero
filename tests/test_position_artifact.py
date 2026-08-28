import numpy
import torch

from vm_compiler.compiler import build_position_reconstruction_artifact
from vm_compiler.protocol import INPUT_ROWS, PIECE_CHANNELS, VOCAB_SIZE, encode_move_one_hot
from vm_compiler.reference_executor import execute_artifact_reference
from vm_compiler.relations import square_index
from vm_compiler.state_circuit import (
    CASTLING_EVENTS,
    build_en_passant_event_patterns,
    reconstruct_position_from_history,
)


def test_executable_latest_event_artifact_matches_exact_position_and_backpropagates():
    moves = (
        encode_move_one_hot(52, 54, PIECE_CHANNELS["WHITE_PAWN"]),
        encode_move_one_hot(47, 45, PIECE_CHANNELS["BLACK_PAWN"]),
        encode_move_one_hot(54, 45, PIECE_CHANNELS["WHITE_PAWN"]),
    )
    vm_input = numpy.zeros((INPUT_ROWS, VOCAB_SIZE), dtype=numpy.float32)
    vm_input[:, 0] = 1.0
    history = vm_input[3:1203]
    history[:9] = numpy.concatenate(moves)
    source = torch.tensor(vm_input[None], requires_grad=True)

    board = execute_artifact_reference(build_position_reconstruction_artifact(), source)
    expected = torch.tensor(reconstruct_position_from_history(history)[None])

    assert board.shape == (1, 64, 128)
    assert torch.equal(board, expected)
    board[0, square_index(45), PIECE_CHANNELS["WHITE_PAWN"]].backward()
    assert source.grad is not None
    assert source.grad[0, 3 + 8, PIECE_CHANNELS["WHITE_PAWN"]] != 0


def _execute_moves(*moves):
    vm_input = numpy.zeros((INPUT_ROWS, VOCAB_SIZE), dtype=numpy.float32)
    vm_input[:, 0] = 1.0
    history = vm_input[3:1203]
    history[: len(moves) * 3] = numpy.concatenate(moves)
    board = execute_artifact_reference(
        build_position_reconstruction_artifact(), torch.tensor(vm_input[None])
    )
    return board, torch.tensor(reconstruct_position_from_history(history)[None])


def test_executable_artifact_derives_castling_rook_events():
    board, expected = _execute_moves(
        encode_move_one_hot(51, 71, PIECE_CHANNELS["WHITE_KING"])
    )

    assert torch.equal(board, expected)


def test_executable_artifact_derives_en_passant_clear_event():
    board, expected = _execute_moves(
        encode_move_one_hot(52, 54, PIECE_CHANNELS["WHITE_PAWN"]),
        encode_move_one_hot(17, 16, PIECE_CHANNELS["BLACK_PAWN"]),
        encode_move_one_hot(54, 55, PIECE_CHANNELS["WHITE_PAWN"]),
        encode_move_one_hot(47, 45, PIECE_CHANNELS["BLACK_PAWN"]),
        encode_move_one_hot(55, 46, PIECE_CHANNELS["WHITE_PAWN"]),
    )

    assert torch.equal(board, expected)


def test_executable_artifact_covers_every_compiled_special_event_pattern():
    histories = []
    for pattern in CASTLING_EVENTS:
        history = numpy.zeros((1200, 128), dtype=numpy.float32)
        history[:, 0] = 1.0
        history[:3] = encode_move_one_hot(*pattern[:3])
        histories.append(history)
    for pattern in build_en_passant_event_patterns():
        history = numpy.zeros((1200, 128), dtype=numpy.float32)
        history[:, 0] = 1.0
        history[:3] = encode_move_one_hot(*pattern[3:6])
        history[3:6] = encode_move_one_hot(*pattern[:3])
        histories.append(history)

    vm_input = numpy.zeros((len(histories), INPUT_ROWS, VOCAB_SIZE), dtype=numpy.float32)
    vm_input[:, :, 0] = 1.0
    vm_input[:, 3:1203] = numpy.stack(histories)
    actual = execute_artifact_reference(
        build_position_reconstruction_artifact(), torch.tensor(vm_input)
    )
    expected = torch.tensor(
        numpy.stack([reconstruct_position_from_history(history) for history in histories])
    )

    assert len(histories) == 32
    assert torch.equal(actual, expected)
