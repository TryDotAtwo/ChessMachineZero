from __future__ import annotations

import torch

from chess_machine_zero.chess.board_io import STARTING_FEN, parse_fen
from chess_machine_zero.chess.rules_oracle import board_after_uci, legal_uci_set
from chess_machine_zero.model.percepta_attention_block_stack import (
    FrozenTransformerAttentionBlockStack,
    _CompiledAttentionProgramExecutor,
)
from chess_machine_zero.model.percepta_frozen_attention_vm import PerceptaFrozenAttentionRuleCompiler
from chess_machine_zero.model.percepta_matrix_attention_runtime import FrozenMatrixAttentionInterpreter
from chess_machine_zero.model.percepta_rule_compiler import (
    ChessRuleISA,
    ChessRuleMicroprogramCompiler,
    ProgramEntrypoint,
)
from chess_machine_zero.trace.reconstruct import reconstruct_board_squares
from chess_machine_zero.vm.interpreter import legal_uci_set_from_trace


def test_rule_microprogram_compiles_to_frozen_attention_weight_program() -> None:
    compiler = ChessRuleMicroprogramCompiler()
    microprogram = compiler.build_microprogram()
    program = compiler.compile(microprogram)

    assert microprogram.source_language == "chess_rule_isa"
    assert program.compiler_backend == "rule_microprogram_to_frozen_attention_weights"
    assert program.execution_backend == "matrix_attention_interpreter"
    assert program.executor_substrate == "QK^T_mask_hardmax_select_V_residual_write"
    assert program.source_is_microprogram is True
    assert program.handwritten_stack_primitive_runtime is False
    assert program.source_microprogram_hash == microprogram.program_hash
    assert set(program.entrypoint_names) >= {
        ProgramEntrypoint.LEGAL_TRACE.name,
        ProgramEntrypoint.MAKE_MOVE_TRACE.name,
        ProgramEntrypoint.TERMINAL_TRACE.name,
    }
    assert program.instruction_count == len(microprogram.instructions)
    assert program.instruction_count >= 18
    assert program.attention_matrix_count >= program.instruction_count
    assert program.residual_write_count >= 3
    assert program.has_instruction(ChessRuleISA.PIECE_DISPATCH)
    assert program.has_instruction(ChessRuleISA.RAY_SCAN)
    assert program.has_instruction(ChessRuleISA.ATTACK_TEST)
    assert program.has_instruction(ChessRuleISA.LEGAL_FILTER)
    assert program.has_instruction(ChessRuleISA.MAKE_MOVE)
    assert program.has_instruction(ChessRuleISA.TERMINAL_PREDICATES)
    assert isinstance(program.instruction_opcodes, torch.nn.Parameter)
    assert isinstance(program.attention_query_weights, torch.nn.Parameter)
    assert program.attention_query_weights.shape == program.attention_key_weights.shape
    assert program.attention_query_weights.shape == program.attention_value_weights.shape
    assert all(not parameter.requires_grad for parameter in program.parameters())


def test_vm_uses_compiled_rule_program_and_unified_executor() -> None:
    rules = PerceptaFrozenAttentionRuleCompiler().compile_trace_computer()

    assert rules.percepta_compiler_pipeline == "chess_isa_microprogram_to_frozen_attention_weights"
    assert rules.rule_compiler_backend == "rule_microprogram_to_frozen_attention_weights"
    assert rules.rule_microprogram_source == "chess_rule_isa"
    assert rules.compiled_rule_program.source_is_microprogram is True
    assert rules.compiled_rule_program.handwritten_stack_primitive_runtime is False
    assert rules.unified_rule_executor_runtime is True
    assert rules.handwritten_stack_primitive_runtime is False
    assert rules.matrix_attention_interpreter_runtime is True
    assert rules.executor_substrate == "matrix_attention_interpreter"
    assert rules.attention_step_operator == "QK^T_mask_hardmax_select_V_residual_write"
    assert rules.pytorch_domain_shortcut_runtime is False
    assert rules.rule_microprogram_instruction_count >= 18
    assert rules.compiled_rule_program_weight_count >= rules.rule_microprogram_instruction_count
    assert rules.attention_block_stack.compiled_program is rules.compiled_rule_program
    assert isinstance(rules.attention_block_stack.matrix_interpreter, FrozenMatrixAttentionInterpreter)
    assert rules.attention_block_stack.handwritten_stack_primitive_runtime is False
    assert rules.attention_block_stack.unified_executor_runtime is True
    assert rules.attention_block_stack.matrix_attention_interpreter_runtime is True
    assert not hasattr(rules.attention_block_stack, "compiled_executor")


def test_compiled_program_runtime_does_not_call_stack_primitive_shortcuts(monkeypatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("stack primitive shortcut was called")

    for name in (
        "piece_dispatch",
        "ray_scan",
        "attack_test",
        "legal_filter",
        "make_move",
        "terminal_predicates",
        "legal_candidate_tensors",
    ):
        monkeypatch.setattr(FrozenTransformerAttentionBlockStack, name, forbidden, raising=False)

    rules = PerceptaFrozenAttentionRuleCompiler().compile_trace_computer()
    prompt_tensor = rules.prompt_tensor_from_board(parse_fen(STARTING_FEN))
    legal_tensor = rules.decode_legal_tensor_trace_host_append_only(prompt_tensor, max_packets=128)
    selected = rules.resolve_legal_move_tensor(prompt_tensor, "e2e4")
    move_tensor = rules.decode_make_move_tensor_trace_host_append_only(prompt_tensor, selected, ply=0, max_packets=64)
    next_board = rules.board_after_move_from_prompt(rules.tensor_trace_to_packets(prompt_tensor), "e2e4")

    assert legal_uci_set_from_trace(rules.tensor_trace_to_packets(legal_tensor)) == legal_uci_set(STARTING_FEN)
    assert reconstruct_board_squares(rules.tensor_trace_to_packets(move_tensor)) == parse_fen(board_after_uci(STARTING_FEN, "e2e4")).squares
    assert next_board.to_fen() == board_after_uci(STARTING_FEN, "e2e4")


def test_matrix_attention_runtime_does_not_call_legacy_compiled_executor(monkeypatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("legacy compiled executor shortcut was called")

    for name in (
        "board_from_trace_tensor",
        "legal_trace_with_halt_tensor",
        "make_move_trace_with_terminal_halt_tensor",
        "terminal_trace_tensor",
        "board_transition_tensor",
        "resolve_legal_move_tensor",
        "legal_move_tensors_from_prompt_tensor",
        "piece_dispatch",
        "ray_scan",
        "attack_test",
        "legal_filter",
        "make_move",
        "terminal_predicates",
        "legal_candidate_tensors",
    ):
        monkeypatch.setattr(_CompiledAttentionProgramExecutor, name, forbidden, raising=False)

    rules = PerceptaFrozenAttentionRuleCompiler().compile_trace_computer()
    prompt_tensor = rules.prompt_tensor_from_board(parse_fen(STARTING_FEN))
    legal_tensor = rules.decode_legal_tensor_trace_host_append_only(prompt_tensor, max_packets=128)
    selected = rules.resolve_legal_move_tensor(prompt_tensor, "e2e4")
    move_tensor = rules.decode_make_move_tensor_trace_host_append_only(prompt_tensor, selected, ply=0, max_packets=64)

    assert legal_uci_set_from_trace(rules.tensor_trace_to_packets(legal_tensor)) == legal_uci_set(STARTING_FEN)
    assert reconstruct_board_squares(rules.tensor_trace_to_packets(move_tensor)) == parse_fen(board_after_uci(STARTING_FEN, "e2e4")).squares
    assert rules.matrix_attention_step_count > 0
    assert rules.matrix_residual_write_count > 0
