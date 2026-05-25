"""Development verifier helpers for trace-vs-oracle checks."""

from __future__ import annotations

from chess_machine_zero.chess.rules_oracle import board_after_uci, legal_uci_set
from chess_machine_zero.trace.reconstruct import reconstruct_board_squares
from chess_machine_zero.chess.board_io import parse_fen
from chess_machine_zero.vm.interpreter import legal_uci_set_from_trace
from chess_machine_zero.vm.trace_packet import TracePacket


def legal_trace_matches_oracle(fen: str, trace: list[TracePacket]) -> bool:
    return legal_uci_set_from_trace(trace) == legal_uci_set(fen)


def make_move_trace_reconstructs_oracle_board(fen: str, uci: str, trace: list[TracePacket]) -> bool:
    oracle_board = parse_fen(board_after_uci(fen, uci))
    return reconstruct_board_squares(trace) == oracle_board.squares
