from __future__ import annotations

import random
from pathlib import Path

import pytest

from chess_machine_zero.chess.board_io import STARTING_FEN, parse_fen
from chess_machine_zero.chess.rules_oracle import board_after_uci, legal_uci_set
from chess_machine_zero.model.percepta_parametric_rules import (
    PerceptaParametricRuleCompiler,
    PerceptaParametricRulesTransformer,
)
from chess_machine_zero.trace.reconstruct import reconstruct_board_squares
from chess_machine_zero.vm.interpreter import legal_uci_set_from_trace
from chess_machine_zero.vm.trace_packet import TraceOp


CRITICAL_FENS = (
    STARTING_FEN,
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/RN2KB1R w KQkq - 0 1",
    "7k/P7/8/8/8/8/6K1/8 w - - 0 1",
    "4r3/8/8/8/8/8/4R3/4K2k w - - 0 1",
    "4r2k/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "8/8/8/8/8/8/5r2/4K2k w - - 0 1",
)


def _rules() -> PerceptaParametricRulesTransformer:
    return PerceptaParametricRuleCompiler().compile_rule_circuit()


def _oracle_walk_positions(count: int) -> tuple[str, ...]:
    rng = random.Random(20260524)
    fen = STARTING_FEN
    positions: list[str] = []
    while len(positions) < count:
        positions.append(fen)
        legal = sorted(legal_uci_set(fen))
        if not legal:
            fen = STARTING_FEN
            continue
        fen = board_after_uci(fen, legal[rng.randrange(len(legal))])
    return tuple(positions)


def test_parametric_rule_transformer_uses_weights_not_finite_position_lookup() -> None:
    rules = _rules()
    source = Path("src/chess_machine_zero/model/percepta_parametric_rules.py").read_text(encoding="utf-8")

    assert rules.rule_execution_mode == "percepta_parametric_rule_weights"
    assert rules.attention_backend == "dense_hardmax_2d"
    assert rules.parametric_rule_weights is True
    assert rules.position_lookup is False
    assert rules.compiled_prompt_count == 0
    assert rules.trainable_rule_parameter_count() == 0
    assert rules.compiled_rule_parameter_count() > 4096
    assert all(not parameter.requires_grad for parameter in rules.parameters())
    assert "PerceptaE2ETraceDecoder" not in source
    assert "prompt_fingerprints" not in source
    assert "continuation_tokens" not in source
    assert "rules_oracle" not in source
    assert "ChessMachineVM" not in source
    assert "AnalyticRuleCompiler" not in source


@pytest.mark.parametrize("fen", CRITICAL_FENS)
def test_parametric_rule_transformer_matches_oracle_for_critical_rules(fen: str) -> None:
    rules = _rules()
    prompt = rules.prompt_trace_from_board(parse_fen(fen))
    trace = rules.legal_move_trace_from_prompt(prompt, include_halt=True)

    assert trace[-1].op is TraceOp.PROGRAM_HALT
    assert legal_uci_set_from_trace(trace) == legal_uci_set(fen)


def test_parametric_rule_transformer_matches_oracle_for_arbitrary_uncompiled_positions() -> None:
    rules = _rules()

    for fen in _oracle_walk_positions(128):
        prompt = rules.prompt_trace_from_board(parse_fen(fen))
        trace = rules.legal_move_trace_from_prompt(prompt, include_halt=True)

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
def test_parametric_rule_transformer_make_move_trace_matches_oracle_board(fen: str, uci: str) -> None:
    rules = _rules()
    prompt = rules.prompt_trace_from_board(parse_fen(fen))
    trace = rules.make_move_trace_from_prompt(prompt, uci, ply=0)

    assert reconstruct_board_squares(trace) == parse_fen(board_after_uci(fen, uci)).squares
