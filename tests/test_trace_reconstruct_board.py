from __future__ import annotations

from chess_machine_zero.chess.board_io import STARTING_FEN, parse_fen, piece_token
from chess_machine_zero.trace.reconstruct import reconstruct_board_squares
from chess_machine_zero.vm.interpreter import ChessMachineVM
from chess_machine_zero.vm.trace_packet import TraceOp, TracePacket, TraceTag


def test_reconstruct_initial_board_from_vm_trace() -> None:
    board = parse_fen(STARTING_FEN)
    vm = ChessMachineVM(seed=1234)
    trace = vm.legal_move_trace(STARTING_FEN)
    assert reconstruct_board_squares(trace) == board.squares


def test_latest_write_wins_for_square_reconstruction() -> None:
    trace = [
        TracePacket(TraceOp.WRITE_SQ, 0, piece_token("R"), 0, 0, TraceTag.BOARD, 0),
        TracePacket(TraceOp.WRITE_SQ, 0, piece_token(None), 1, 0, TraceTag.BOARD, 0),
        TracePacket(TraceOp.WRITE_SQ, 63, piece_token("k"), 1, 0, TraceTag.BOARD, 0),
    ]
    squares = reconstruct_board_squares(trace)
    assert squares[0] is None
    assert squares[63] == "k"
