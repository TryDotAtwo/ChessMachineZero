from __future__ import annotations

import random

import pytest

from chess_machine_zero.chess.board_io import STARTING_FEN, parse_fen
from chess_machine_zero.chess.rules_oracle import board_after_uci, legal_uci_set
from chess_machine_zero.trace.reconstruct import reconstruct_board_squares
from chess_machine_zero.vm.interpreter import ChessMachineVM, legal_uci_set_from_trace


@pytest.mark.parametrize(
    ("fen", "uci"),
    [
        (STARTING_FEN, "e2e4"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
        ("7k/P7/8/8/8/8/6K1/8 w - - 0 1", "a7a8n"),
        ("7k/8/8/3pP3/8/8/6K1/8 w - d6 0 1", "e5d6"),
    ],
)
def test_vm_make_move_matches_python_chess_oracle(fen: str, uci: str) -> None:
    vm = ChessMachineVM(seed=1234)
    assert vm.make_move(fen, uci) == board_after_uci(fen, uci)
    trace = vm.make_move_trace(fen, uci)
    assert reconstruct_board_squares(trace) == parse_fen(board_after_uci(fen, uci)).squares


def test_vm_rejects_illegal_make_move() -> None:
    vm = ChessMachineVM(seed=1234)
    with pytest.raises(ValueError):
        vm.make_move(STARTING_FEN, "e2e5")


def test_1000_deterministic_random_positions_match_oracle() -> None:
    vm = ChessMachineVM(seed=20260524)
    rng = random.Random(20260524)
    positions_checked = 0
    game_index = 0
    while positions_checked < 1000:
        fen = STARTING_FEN
        for _ply in range(256):
            trace = vm.legal_move_trace(fen)
            vm_legal = legal_uci_set_from_trace(trace)
            oracle_legal = legal_uci_set(fen)
            assert vm_legal == oracle_legal
            positions_checked += 1
            if positions_checked >= 1000 or not vm_legal:
                break
            selected = sorted(vm_legal)[rng.randrange(len(vm_legal))]
            assert vm.make_move(fen, selected) == board_after_uci(fen, selected)
            fen = vm.make_move(fen, selected)
        game_index += 1
        assert game_index < 64
