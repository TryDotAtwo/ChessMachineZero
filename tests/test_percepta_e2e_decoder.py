from __future__ import annotations

from pathlib import Path

from chess_machine_zero.chess.board_io import STARTING_FEN, parse_fen
from chess_machine_zero.model.percepta_e2e_decoder import PerceptaE2ETraceDecoder
from chess_machine_zero.model.percepta_selfplay import PerceptaSelfPlaySession
from chess_machine_zero.model.weight_compiled_rules import WeightCompiledRuleCompiler
from chess_machine_zero.trace.verifier import legal_trace_matches_oracle
from chess_machine_zero.vm.interpreter import legal_uci_set_from_trace
from chess_machine_zero.vm.trace_packet import TraceOp


FENS = (
    STARTING_FEN,
    "8/8/8/8/8/8/8/4K2k w - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "7k/P7/8/8/8/8/6K1/8 w - - 0 1",
)


def test_percepta_e2e_decoder_compiles_rule_traces_into_frozen_attention_weights() -> None:
    compiler = WeightCompiledRuleCompiler().compile_legal_generator()
    decoder = PerceptaE2ETraceDecoder.compile_from_rule_traces(FENS, compiler=compiler, include_halt=True)

    assert decoder.execution_mode == "percepta_e2e_trace_decoder"
    assert decoder.attention_backend == "dense_hardmax_2d"
    assert decoder.compiled_prompt_count == len(FENS)
    assert decoder.trainable_parameter_count() == 0
    assert decoder.compiled_parameter_count() > 4096
    assert all(not parameter.requires_grad for parameter in decoder.parameters())
    assert not hasattr(decoder, "compiler")
    assert not hasattr(decoder, "rules")
    assert not hasattr(decoder, "vm")


def test_percepta_e2e_decoder_source_has_no_runtime_rule_executor_imports() -> None:
    source = Path("src/chess_machine_zero/model/percepta_e2e_decoder.py").read_text(encoding="utf-8")

    assert "ChessMachineVM" not in source
    assert "AnalyticRuleCompiler" not in source
    assert "AnalyticRulesTransformer" not in source
    assert "rules_oracle" not in source


def test_percepta_e2e_decoder_model_only_decodes_starting_position_legal_trace() -> None:
    compiler = WeightCompiledRuleCompiler().compile_legal_generator()
    decoder = PerceptaE2ETraceDecoder.compile_from_rule_traces((STARTING_FEN,), compiler=compiler, include_halt=True)
    prompt = compiler.prompt_trace_from_board(parse_fen(STARTING_FEN))
    decoded = decoder.decode_until_halt(prompt, max_packets=128)
    full_trace = tuple(prompt) + decoded

    assert decoded[-1].op is TraceOp.PROGRAM_HALT
    assert len(legal_uci_set_from_trace(full_trace)) == 20
    assert "e2e4" in legal_uci_set_from_trace(full_trace)
    assert legal_trace_matches_oracle(STARTING_FEN, full_trace)


def test_percepta_e2e_decoder_exact_decodes_multiple_known_prompts() -> None:
    compiler = WeightCompiledRuleCompiler().compile_legal_generator()
    decoder = PerceptaE2ETraceDecoder.compile_from_rule_traces(FENS, compiler=compiler, include_halt=True)

    for fen in FENS:
        prompt = compiler.prompt_trace_from_board(parse_fen(fen))
        decoded = decoder.decode_until_halt(prompt, max_packets=256)
        full_trace = tuple(prompt) + decoded

        assert decoded[-1].op is TraceOp.PROGRAM_HALT
        assert legal_trace_matches_oracle(fen, full_trace)


def test_percepta_e2e_decoder_rejects_uncompiled_prompt_without_substitution() -> None:
    compiler = WeightCompiledRuleCompiler().compile_legal_generator()
    decoder = PerceptaE2ETraceDecoder.compile_from_rule_traces((STARTING_FEN,), compiler=compiler, include_halt=True)
    unknown_prompt = compiler.prompt_trace_from_board(parse_fen("8/8/8/8/8/8/8/2K4k w - - 0 1"))

    try:
        decoder.decode_until_halt(unknown_prompt, max_packets=128)
    except ValueError as error:
        assert "uncompiled prompt" in str(error)
    else:
        raise AssertionError("uncompiled prompt must not be silently substituted")


def test_percepta_selfplay_session_steps_by_decoded_commit_trace_only() -> None:
    session = PerceptaSelfPlaySession.compile_deterministic(
        start_fen=STARTING_FEN,
        seed=20260524,
        max_plies=8,
    )

    event = session.step()

    assert session.runtime_rule_executor is False
    assert event.actor == "transformer"
    assert event.trace[-1].op is TraceOp.PROGRAM_HALT
    assert event.trace_op_counts["CANDIDATE"] == 20
    assert event.trace_op_counts["LEGAL_SET"] == 20
    assert event.trace_op_counts["COMMIT_MOVE"] == 1
    assert event.move_uci in legal_uci_set_from_trace(event.trace)
    assert session.snapshot()["ply"] == 1
    assert session.snapshot()["illegal_commit_count"] == 0
