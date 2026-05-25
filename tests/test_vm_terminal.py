from __future__ import annotations

from chess_machine_zero.chess.outcome import ResultCode, TerminalReason
from chess_machine_zero.chess.rules_oracle import terminal_status as oracle_terminal_status
from chess_machine_zero.vm.interpreter import ChessMachineVM, terminal_status
from chess_machine_zero.chess.board_io import parse_fen


def test_checkmate_terminal_matches_oracle() -> None:
    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    status = terminal_status(parse_fen(fen))
    assert status == oracle_terminal_status(fen)
    assert status.result is ResultCode.BLACK_WIN
    assert status.reason is TerminalReason.CHECKMATE


def test_stalemate_terminal_matches_oracle() -> None:
    fen = "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"
    status = terminal_status(parse_fen(fen))
    assert status == oracle_terminal_status(fen)
    assert status.result is ResultCode.DRAW
    assert status.reason is TerminalReason.STALEMATE


def test_fifty_move_terminal() -> None:
    fen = "4k3/8/8/8/8/8/8/R3K3 w - - 100 1"
    status = terminal_status(parse_fen(fen))
    assert status.result is ResultCode.DRAW
    assert status.reason is TerminalReason.FIFTY_MOVE


def test_insufficient_material_terminal_matches_oracle() -> None:
    fen = "8/8/8/8/8/8/5k2/4K3 w - - 0 1"
    status = terminal_status(parse_fen(fen))
    assert status == oracle_terminal_status(fen)
    assert status.reason is TerminalReason.INSUFFICIENT_MATERIAL


def test_terminal_trace_emits_terminal_set() -> None:
    vm = ChessMachineVM(seed=1234)
    board = parse_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")
    trace = vm.terminal_trace(board)
    assert len(trace) == 1
    assert trace[0].a0 == int(ResultCode.DRAW)
    assert trace[0].a1 == int(TerminalReason.STALEMATE)
    assert trace[0].commit == 1
