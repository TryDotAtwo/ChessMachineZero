from __future__ import annotations

import pytest

from chess_machine_zero.chess.board_io import STARTING_FEN
from chess_machine_zero.chess.rules_oracle import legal_uci_set
from chess_machine_zero.vm.interpreter import ChessMachineVM, legal_uci_set_from_trace


@pytest.mark.parametrize(
    "fen",
    [
        STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/RN2KB1R w KQkq - 0 1",
        "7k/P7/8/8/8/8/6K1/8 w - - 0 1",
        "4r3/8/8/8/8/8/4R3/4K2k w - - 0 1",
        "4r2k/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "8/8/8/8/8/8/5r2/4K2k w - - 0 1",
    ],
)
def test_vm_legal_move_trace_matches_python_chess_oracle(fen: str) -> None:
    vm = ChessMachineVM(seed=1234)
    trace = vm.legal_move_trace(fen)
    assert legal_uci_set_from_trace(trace) == legal_uci_set(fen)


def test_starting_position_has_20_legal_moves() -> None:
    vm = ChessMachineVM(seed=1234)
    assert len(vm.legal_moves(STARTING_FEN)) == 20


def test_en_passant_discovered_check_is_not_legal() -> None:
    fen = "4r2k/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
    vm = ChessMachineVM(seed=1234)
    legal = legal_uci_set_from_trace(vm.legal_move_trace(fen))
    assert "e5d6" not in legal
    assert legal == legal_uci_set(fen)
