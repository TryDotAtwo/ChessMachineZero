from __future__ import annotations

from pathlib import Path

import pytest

from chess_machine_zero.chess.board_io import STARTING_FEN, parse_fen
from chess_machine_zero.chess.outcome import ResultCode, TerminalReason
from chess_machine_zero.chess.rules_oracle import board_after_uci, legal_uci_set, terminal_status as oracle_terminal_status
from chess_machine_zero.model.weight_compiled_rules import WeightCompiledRuleCompiler, WeightCompiledRulesTransformer
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


def _rules() -> WeightCompiledRulesTransformer:
    return WeightCompiledRuleCompiler().compile_legal_generator()


def _prompt(rules: WeightCompiledRulesTransformer, fen: str):
    return rules.prompt_trace_from_board(parse_fen(fen))


def test_weight_compiled_rules_are_frozen_model_weights_not_analytic_proxy() -> None:
    rules = _rules()
    state_keys = set(rules.state_dict())
    source = Path("src/chess_machine_zero/model/weight_compiled_rules.py").read_text(encoding="utf-8")

    assert rules.rule_execution_mode == "weight_compiled"
    assert rules.trainable_rule_parameter_count() == 0
    assert rules.compiled_rule_parameter_count() > 4096
    assert all(not parameter.requires_grad for parameter in rules.parameters())
    assert "rule_weights.ray_squares" in state_keys
    assert "AnalyticRulesTransformer" not in source
    assert "AnalyticRuleCompiler" not in source
    assert "ChessMachineVM" not in source
    assert "rules_oracle" not in source


def test_dashboard_rule_path_no_longer_uses_python_executors() -> None:
    source = Path("docker/native/start_dashboard.ps1").read_text(encoding="utf-8")

    assert "cargo run -p cmz-dashboard" in source
    assert "WeightCompiledRuleCompiler" not in source
    assert "AnalyticRuleCompiler" not in source
    assert "chess_machine_zero.dashboard.server" not in source


@pytest.mark.parametrize("fen", CRITICAL_FENS)
def test_weight_compiled_legal_generator_matches_oracle_for_critical_rules(fen: str) -> None:
    rules = _rules()
    trace = rules.legal_move_trace_from_prompt(_prompt(rules, fen), include_halt=True)

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
def test_weight_compiled_make_move_trace_matches_oracle_board(fen: str, uci: str) -> None:
    rules = _rules()
    trace = rules.make_move_trace_from_prompt(_prompt(rules, fen), uci, ply=0)

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
def test_weight_compiled_terminal_trace_matches_oracle(fen: str) -> None:
    rules = _rules()
    packet = rules.terminal_trace_from_prompt(_prompt(rules, fen), ply=3)[0]
    status = oracle_terminal_status(fen, ply=3)

    assert packet.a0 == int(status.result)
    assert packet.a1 == int(status.reason)
    assert packet.commit == int(status.is_terminal)


def test_weight_compiled_threefold_terminal_trace_is_hard_weight_rule() -> None:
    rules = _rules()
    packet = rules.terminal_trace_from_prompt(_prompt(rules, STARTING_FEN), ply=12, repetition_count=3)[0]

    assert packet.a0 == int(ResultCode.DRAW)
    assert packet.a1 == int(TerminalReason.THREEFOLD)
    assert packet.commit == 1
