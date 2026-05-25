from __future__ import annotations

from chess_machine_zero.chess.board_io import parse_fen
from chess_machine_zero.model.analytic_machine import CMZAnalyticMachine
from chess_machine_zero.model.analytic_rules import AnalyticRuleCompiler, AnalyticRulesTransformer
from chess_machine_zero.model.ranker import CMZMoveRanker
from chess_machine_zero.trace.verifier import legal_trace_matches_oracle
from chess_machine_zero.vm.interpreter import ChessMachineVM, legal_uci_set_from_trace
from chess_machine_zero.vm.trace_packet import TraceOp


FENS = (
    "8/8/8/8/8/8/8/4K2k w - - 0 1",
    "8/8/8/8/8/8/8/2K4k w - - 0 1",
    "7k/P7/8/8/8/8/6K1/8 w - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
)


def test_analytic_rule_compiler_builds_zero_trainable_parameter_executor() -> None:
    executor = AnalyticRuleCompiler().compile_legal_generator()

    assert isinstance(executor, AnalyticRulesTransformer)
    assert executor.trainable_rule_parameter_count() == 0
    assert tuple(executor.parameters()) == ()
    assert executor.rule_execution_mode == "analytic_fixed"


def test_analytic_rules_transformer_emits_exact_legal_trace_from_prompt_trace() -> None:
    executor = AnalyticRuleCompiler().compile_legal_generator()
    host_vm = ChessMachineVM(seed=20260524)

    for fen in FENS:
        prompt = tuple(host_vm.initial_board_trace(parse_fen(fen)))
        analytic_trace = executor.legal_move_trace_from_prompt(prompt, include_halt=True)
        reference_trace = tuple(host_vm.legal_move_trace(fen, include_halt=True))

        assert analytic_trace == reference_trace
        assert analytic_trace[-1].op is TraceOp.PROGRAM_HALT
        assert legal_trace_matches_oracle(fen, list(analytic_trace))


def test_analytic_machine_keeps_rules_fixed_and_strategy_trainable() -> None:
    machine = CMZAnalyticMachine(
        rules=AnalyticRuleCompiler().compile_legal_generator(),
        ranker=CMZMoveRanker(seed=20260524),
    )
    host_vm = ChessMachineVM(seed=20260524)
    fen = "8/8/8/8/8/8/8/4K2k w - - 0 1"
    prompt = tuple(host_vm.initial_board_trace(parse_fen(fen)))
    decision = machine.select_move_from_prompt(prompt, seed=7, temperature=1.0)

    assert machine.rules.trainable_rule_parameter_count() == 0
    assert sum(parameter.numel() for parameter in machine.ranker.parameters() if parameter.requires_grad) > 0
    assert decision.chosen_move.to_uci() in legal_uci_set_from_trace(decision.trace)
    assert legal_trace_matches_oracle(fen, list(decision.trace))
