from __future__ import annotations

import pytest

from chess_machine_zero.chess.board_io import STARTING_FEN, parse_fen
from chess_machine_zero.chess.outcome import ResultCode, TerminalReason
from chess_machine_zero.chess.rules_oracle import board_after_uci, legal_uci_set, terminal_status as oracle_terminal_status
from chess_machine_zero.model.analytic_rules import AnalyticRuleCompiler
from chess_machine_zero.trace.reconstruct import reconstruct_board_squares
from chess_machine_zero.vm.interpreter import legal_uci_set_from_trace


CRITICAL_FENS = (
    STARTING_FEN,
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/RN2KB1R w KQkq - 0 1",
    "7k/P7/8/8/8/8/6K1/8 w - - 0 1",
    "4r3/8/8/8/8/8/4R3/4K2k w - - 0 1",
    "4r2k/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "8/8/8/8/8/8/5r2/4K2k w - - 0 1",
)


def _prompt(fen: str):
    rules = AnalyticRuleCompiler().compile_legal_generator()
    return rules.prompt_trace_from_board(parse_fen(fen))


@pytest.mark.parametrize("fen", CRITICAL_FENS)
def test_analytic_legal_generator_matches_oracle_for_critical_rules(fen: str) -> None:
    rules = AnalyticRuleCompiler().compile_legal_generator()
    trace = rules.legal_move_trace_from_prompt(_prompt(fen), include_halt=True)

    assert legal_uci_set_from_trace(trace) == legal_uci_set(fen)


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
def test_analytic_make_move_trace_matches_oracle_board(fen: str, uci: str) -> None:
    rules = AnalyticRuleCompiler().compile_legal_generator()
    trace = rules.make_move_trace_from_prompt(_prompt(fen), uci, ply=0)

    assert reconstruct_board_squares(trace) == parse_fen(board_after_uci(fen, uci)).squares


@pytest.mark.parametrize(
    "fen",
    [
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1",
        "4k3/8/8/8/8/8/8/R3K3 w - - 100 1",
        "8/8/8/8/8/8/5k2/4K3 w - - 0 1",
    ],
)
def test_analytic_terminal_trace_matches_oracle(fen: str) -> None:
    rules = AnalyticRuleCompiler().compile_legal_generator()
    packet = rules.terminal_trace_from_prompt(_prompt(fen), ply=3)[0]
    status = oracle_terminal_status(fen, ply=3)

    assert packet.a0 == int(status.result)
    assert packet.a1 == int(status.reason)
    assert packet.commit == int(status.is_terminal)


def test_analytic_threefold_terminal_trace_is_hard_rule() -> None:
    rules = AnalyticRuleCompiler().compile_legal_generator()
    packet = rules.terminal_trace_from_prompt(_prompt(STARTING_FEN), ply=12, repetition_count=3)[0]

    assert packet.a0 == int(ResultCode.DRAW)
    assert packet.a1 == int(TerminalReason.THREEFOLD)
    assert packet.commit == 1
