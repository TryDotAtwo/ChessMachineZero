from __future__ import annotations

import ast
import inspect
import math
import random
import textwrap
from pathlib import Path

import pytest
import torch

from chess_machine_zero.chess.board_io import STARTING_FEN, parse_fen
from chess_machine_zero.chess.rules_oracle import board_after_uci, legal_uci_set
from chess_machine_zero.model.percepta_frozen_attention_vm import (
    PerceptaFrozenAttentionRuleCompiler,
    PerceptaFrozenAttentionRuleComputer,
)
from chess_machine_zero.model.percepta_attention_rule_kernels import FrozenAttentionTensorRuleKernels
from chess_machine_zero.model.percepta_attention_block_stack import (
    FrozenTransformerAttentionBlockStack,
    FrozenTransformerAttentionBlock,
    FrozenAttentionHead,
    ResidualTraceWrite,
)
from chess_machine_zero.model.percepta_tensor_trace_runtime import FrozenAttentionTensorTraceRuntime
from chess_machine_zero.model.percepta_rule_layer_graph import FrozenAttentionRuleLayerGraph
from chess_machine_zero.model.weight_compiled_rules import WeightCompiledRulesTransformer
from chess_machine_zero.trace.reconstruct import reconstruct_board_squares
from chess_machine_zero.vm.interpreter import legal_uci_set_from_trace
from chess_machine_zero.vm.trace_packet import TraceOp


def _rules() -> PerceptaFrozenAttentionRuleComputer:
    return PerceptaFrozenAttentionRuleCompiler().compile_trace_computer()


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


def _control_flow_nodes(method: object) -> list[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    control_nodes = (ast.For, ast.AsyncFor, ast.While, ast.If, ast.IfExp, ast.Match, ast.Try)
    return [type(node).__name__ for node in ast.walk(tree) if isinstance(node, control_nodes)]


def test_frozen_attention_rule_computer_has_no_mlp_and_no_position_lookup() -> None:
    rules = _rules()
    source = Path("src/chess_machine_zero/model/percepta_frozen_attention_vm.py").read_text(encoding="utf-8")

    assert rules.rule_execution_mode == "percepta_frozen_attention_trace_vm"
    assert rules.attention_backend == "logarithmic_2d_attention"
    assert rules.lookup_complexity == "O(log n)"
    assert rules.rule_core_execution_mode == "executable_frozen_attention_layer_graph"
    assert rules.primitive_kernel_execution_mode == "pure_frozen_attention_tensor_layers"
    assert rules.core_trace_runtime == "tensor_trace_in_frozen_attention_blocks_tensor_trace_out"
    assert rules.core_rule_compute_backend == "frozen_transformer_attention_block_stack"
    assert rules.tensor_kernel_shortcut_runtime is False
    assert rules.compiled_attention_block_stack is True
    assert rules.compiled_attention_block_count >= 6
    assert rules.compiled_attention_head_count >= rules.compiled_attention_block_count
    assert rules.residual_trace_write_count >= 2
    assert rules.python_host_boundary_role == "display_only"
    assert rules.tensor_trace_core_runtime is True
    assert rules.tracepacket_core_runtime is False
    assert rules.python_rule_primitive_runtime is False
    assert rules.python_control_flow_rule_primitives is False
    assert rules.uses_dense_scan is False
    assert rules.compiled_layer_graph_serialized is True
    assert rules.host_append_only is True
    assert rules.token_streaming is True
    assert rules.uses_mlp is False
    assert not hasattr(rules, "_hardmax")
    assert rules.position_lookup is False
    assert rules.finite_prompt_lookup is False
    assert rules.compiled_prompt_count == 0
    assert rules.compiled_isa_instruction_count >= 8
    assert rules.compiled_microprogram_step_count >= 16
    assert rules.compiled_attention_layer_count >= rules.compiled_microprogram_step_count
    assert rules.compiled_rule_primitive_count >= 6
    assert rules.tensor_kernel_count >= 6
    assert set(rules.compiled_rule_primitives) >= {
        "PIECE_DISPATCH",
        "RAY_SCAN",
        "ATTACK_TEST",
        "LEGAL_FILTER",
        "MAKE_MOVE",
        "TERMINAL_PREDICATES",
    }
    assert rules.trainable_rule_parameter_count() == 0
    assert all(not parameter.requires_grad for parameter in rules.parameters())
    assert "nn.Linear" not in source
    assert "PerceptaE2ETraceDecoder" not in source
    assert "prompt_fingerprints" not in source
    assert "continuation_tokens" not in source
    assert "rules_oracle" not in source
    assert "ChessMachineVM" not in source
    assert "AnalyticRuleCompiler" not in source
    assert "DenseHardmax2D" not in source


def test_frozen_attention_rule_primitives_are_lowered_to_tensor_kernels() -> None:
    rules = _rules()
    assert isinstance(rules.rule_layer_graph.rule_kernels, FrozenAttentionTensorRuleKernels)
    assert isinstance(rules.attention_block_stack, FrozenTransformerAttentionBlockStack)
    assert all(isinstance(block, FrozenTransformerAttentionBlock) for block in rules.attention_block_stack.blocks)
    assert all(isinstance(head, FrozenAttentionHead) for block in rules.attention_block_stack.blocks for head in block.heads)
    assert all(isinstance(write, ResidualTraceWrite) for write in rules.attention_block_stack.residual_writes)
    assert set(rules.attention_block_stack.block_names) >= set(rules.compiled_rule_primitives)
    assert isinstance(rules.tensor_trace_runtime, FrozenAttentionTensorTraceRuntime)
    for method_name in (
        "piece_dispatch",
        "ray_scan",
        "attack_test",
        "legal_filter",
        "make_move",
        "terminal_predicates",
        "legal_candidate_tensors",
    ):
        assert _control_flow_nodes(getattr(FrozenAttentionTensorRuleKernels, method_name)) == []

    for method_name in (
        "board_from_trace_tensor",
        "legal_trace_with_halt_tensor",
        "make_move_trace_with_terminal_halt_tensor",
        "terminal_trace_tensor",
        "board_transition_tensor",
    ):
        assert _control_flow_nodes(getattr(FrozenAttentionTensorTraceRuntime, method_name)) == []


def test_attention_stack_runtime_does_not_call_tensor_kernel_shortcuts(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("tensor kernel shortcut was called")

    for name in (
        "piece_dispatch",
        "ray_scan",
        "attack_test",
        "legal_filter",
        "make_move",
        "terminal_predicates",
        "legal_candidate_tensors",
    ):
        monkeypatch.setattr(FrozenAttentionTensorRuleKernels, name, forbidden)

    rules = _rules()
    prompt_tensor = rules.prompt_tensor_from_board(parse_fen(STARTING_FEN))
    legal_tensor = rules.decode_legal_tensor_trace_host_append_only(prompt_tensor, max_packets=128)
    selected = rules.resolve_legal_move_tensor(prompt_tensor, "e2e4")
    move_tensor = rules.decode_make_move_tensor_trace_host_append_only(prompt_tensor, selected, ply=0, max_packets=64)

    assert legal_uci_set_from_trace(rules.tensor_trace_to_packets(legal_tensor)) == legal_uci_set(STARTING_FEN)
    assert reconstruct_board_squares(rules.tensor_trace_to_packets(move_tensor)) == parse_fen(board_after_uci(STARTING_FEN, "e2e4")).squares


def test_tensor_trace_core_runtime_does_not_call_packet_or_boardstate_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("packet/BoardState graph runtime was called")

    for name in (
        "legal_move_trace_from_prompt",
        "make_move_trace_from_prompt",
        "board_after_move_from_prompt",
        "terminal_trace_from_prompt",
        "terminal_status_from_prompt",
        "generate_pseudo_legal_moves",
        "is_legal_move",
        "make_move_state",
        "terminal_status",
    ):
        monkeypatch.setattr(FrozenAttentionRuleLayerGraph, name, forbidden)

    rules = _rules()
    prompt_tensor = rules.prompt_tensor_from_board(parse_fen(STARTING_FEN))
    legal_tensor = rules.decode_legal_tensor_trace_host_append_only(prompt_tensor, max_packets=128)
    selected = rules.resolve_legal_move_tensor(prompt_tensor, "e2e4")
    move_tensor = rules.decode_make_move_tensor_trace_host_append_only(prompt_tensor, selected, ply=0, max_packets=64)

    assert isinstance(legal_tensor, torch.Tensor)
    assert isinstance(move_tensor, torch.Tensor)
    assert legal_tensor.ndim == 2
    assert legal_tensor.shape[1] == 7
    assert legal_uci_set_from_trace(rules.tensor_trace_to_packets(legal_tensor)) == legal_uci_set(STARTING_FEN)
    assert reconstruct_board_squares(rules.tensor_trace_to_packets(move_tensor)) == parse_fen(board_after_uci(STARTING_FEN, "e2e4")).squares


def test_frozen_attention_runtime_does_not_call_inherited_python_rule_primitives(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("inherited Python rule primitive was called")

    for name in (
        "legal_move_trace_from_prompt",
        "make_move_trace_from_prompt",
        "board_after_move_from_prompt",
        "terminal_trace_from_prompt",
        "terminal_status_from_prompt",
        "generate_pseudo_legal_moves",
        "is_legal_move",
        "make_move_state",
        "terminal_status",
    ):
        monkeypatch.setattr(WeightCompiledRulesTransformer, name, forbidden)

    rules = _rules()
    prompt = rules.prompt_trace_from_board(parse_fen(STARTING_FEN))
    legal_trace = rules.decode_legal_trace_host_append_only(prompt, max_packets=128)
    move_trace = rules.decode_make_move_trace_host_append_only(prompt, "e2e4", ply=0, max_packets=64)

    assert legal_uci_set_from_trace(legal_trace) == legal_uci_set(STARTING_FEN)
    assert reconstruct_board_squares(move_trace) == parse_fen(board_after_uci(STARTING_FEN, "e2e4")).squares
    assert rules.graph_execution_counts["PIECE_DISPATCH"] > 0
    assert rules.graph_execution_counts["RAY_SCAN"] > 0
    assert rules.graph_execution_counts["ATTACK_TEST"] > 0
    assert rules.graph_execution_counts["LEGAL_FILTER"] > 0
    assert rules.graph_execution_counts["MAKE_MOVE"] > 0
    assert rules.graph_execution_counts["TERMINAL_PREDICATES"] > 0


def test_frozen_attention_cursor_lookup_is_logarithmic_and_exact() -> None:
    rules = _rules()
    bound = math.ceil(math.log2(rules.max_decode_packets)) + 4

    for index in (0, 1, 7, 41, 255, 1024, rules.max_decode_packets - 1):
        assert rules.attention_select_decode_step(index) == index
        assert 1 <= rules.last_lookup_steps <= bound
        assert rules.last_lookup_used_dense_scan is False


def test_host_append_loop_emits_legal_trace_one_packet_per_forward() -> None:
    rules = _rules()
    prompt = rules.prompt_trace_from_board(parse_fen(STARTING_FEN))
    current = list(prompt)
    emitted: list[TraceOp] = []

    for _ in range(128):
        packet = rules.decode_next_legal_packet(tuple(current), prompt_length=len(prompt))
        current.append(packet)
        emitted.append(packet.op)
        if packet.op is TraceOp.PROGRAM_HALT:
            break

    assert emitted[-1] is TraceOp.PROGRAM_HALT
    assert rules.decode_forward_count == len(emitted)
    assert rules.max_lookup_steps <= math.ceil(math.log2(rules.max_decode_packets)) + 4
    assert emitted.count(TraceOp.CANDIDATE) == 20
    assert emitted.count(TraceOp.LEGAL_SET) == 20
    assert legal_uci_set_from_trace(tuple(current)) == legal_uci_set(STARTING_FEN)


def test_frozen_attention_vm_matches_oracle_for_arbitrary_uncompiled_positions() -> None:
    rules = _rules()

    for fen in _oracle_walk_positions(128):
        prompt = rules.prompt_trace_from_board(parse_fen(fen))
        trace = rules.decode_legal_trace_host_append_only(prompt, max_packets=256)

        assert trace[-1].op is TraceOp.PROGRAM_HALT
        assert legal_uci_set_from_trace(trace) == legal_uci_set(fen)


def test_frozen_attention_make_move_stream_matches_oracle_board() -> None:
    rules = _rules()
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    prompt = rules.prompt_trace_from_board(parse_fen(fen))
    trace = rules.decode_make_move_trace_host_append_only(prompt, "e1g1", ply=0, max_packets=64)

    assert trace[-1].op is TraceOp.PROGRAM_HALT
    assert reconstruct_board_squares(trace) == parse_fen(board_after_uci(fen, "e1g1")).squares


def test_frozen_attention_decoder_rejects_corrupted_appended_prefix() -> None:
    rules = _rules()
    prompt = rules.prompt_trace_from_board(parse_fen(STARTING_FEN))
    current = list(prompt)
    first = rules.decode_next_legal_packet(tuple(current), prompt_length=len(prompt))
    current.append(first)
    current.append(first)

    try:
        rules.decode_next_legal_packet(tuple(current), prompt_length=len(prompt))
    except ValueError as error:
        assert "corrupted appended trace" in str(error)
    else:
        raise AssertionError("corrupted appended trace must be rejected")
