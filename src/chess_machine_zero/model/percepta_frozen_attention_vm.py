"""Token-streaming chess rule computer using frozen 2D attention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from math import ceil, log2
from typing import Sequence

import torch
from torch import nn

from chess_machine_zero.chess.board_io import NO_EP, BoardState, castling_mask, piece_token
from chess_machine_zero.chess.move_packet import MovePacket
from chess_machine_zero.model.percepta_attention_block_stack import FrozenTransformerAttentionBlockStack
from chess_machine_zero.model.percepta_attention_rule_kernels import FrozenAttentionTensorRuleKernels
from chess_machine_zero.model.percepta_parametric_rules import PerceptaParametricRulesTransformer
from chess_machine_zero.model.percepta_rule_compiler import ChessRuleMicroprogramCompiler
from chess_machine_zero.model.percepta_rule_layer_graph import FrozenAttentionRuleLayerGraph, RULE_PRIMITIVE_NAMES
from chess_machine_zero.model.percepta_tensor_trace_runtime import FrozenAttentionTensorTraceRuntime
from chess_machine_zero.model.weight_compiled_rules import WeightCompiledRulesTransformer, board_state_from_prompt_trace
from chess_machine_zero.vm.trace_packet import RegId, TraceOp, TracePacket, TraceTag


class ChessISAOp(IntEnum):
    READ_SQ = 1
    FIND_OWN_PIECES = 2
    PIECE_DISPATCH = 3
    LEAPER_LOOKUP = 4
    RAY_SCAN = 5
    PAWN_RULE = 6
    CASTLE_RULE = 7
    ATTACK_TEST = 8
    KING_SAFE = 9
    EMIT_CANDIDATE = 10
    EMIT_LEGAL = 11
    MAKE_MOVE = 12
    TERMINAL_TEST = 13
    HALT = 14


MICROPROGRAM = (
    ChessISAOp.READ_SQ,
    ChessISAOp.FIND_OWN_PIECES,
    ChessISAOp.PIECE_DISPATCH,
    ChessISAOp.PAWN_RULE,
    ChessISAOp.LEAPER_LOOKUP,
    ChessISAOp.RAY_SCAN,
    ChessISAOp.CASTLE_RULE,
    ChessISAOp.ATTACK_TEST,
    ChessISAOp.KING_SAFE,
    ChessISAOp.EMIT_CANDIDATE,
    ChessISAOp.EMIT_LEGAL,
    ChessISAOp.MAKE_MOVE,
    ChessISAOp.TERMINAL_TEST,
    ChessISAOp.HALT,
    ChessISAOp.READ_SQ,
    ChessISAOp.PIECE_DISPATCH,
    ChessISAOp.ATTACK_TEST,
    ChessISAOp.EMIT_LEGAL,
)


class PerceptaFrozenAttentionRuleComputer(PerceptaParametricRulesTransformer):
    """Frozen-attention trace VM over arbitrary board prompts."""

    rule_execution_mode = "percepta_frozen_attention_trace_vm"
    attention_backend = "logarithmic_2d_attention"
    lookup_complexity = "O(log n)"
    host_append_only = True
    token_streaming = True
    uses_mlp = False
    uses_dense_scan = False
    compiled_layer_graph_serialized = True
    rule_core_execution_mode = "executable_frozen_attention_layer_graph"
    primitive_kernel_execution_mode = "pure_frozen_attention_tensor_layers"
    core_trace_runtime = "tensor_trace_in_frozen_attention_blocks_tensor_trace_out"
    core_rule_compute_backend = "frozen_transformer_attention_block_stack"
    python_host_boundary_role = "display_only"
    tensor_trace_core_runtime = True
    tracepacket_core_runtime = False
    tensor_kernel_shortcut_runtime = False
    compiled_attention_block_stack = True
    percepta_compiler_pipeline = "chess_isa_microprogram_to_frozen_attention_weights"
    rule_compiler_backend = "rule_microprogram_to_frozen_attention_weights"
    rule_microprogram_source = "chess_rule_isa"
    unified_rule_executor_runtime = True
    handwritten_stack_primitive_runtime = False
    matrix_attention_interpreter_runtime = True
    executor_substrate = "matrix_attention_interpreter"
    attention_step_operator = "QK^T_mask_hardmax_select_V_residual_write"
    pytorch_domain_shortcut_runtime = False
    python_rule_primitive_runtime = False
    python_control_flow_rule_primitives = False

    def __init__(self, max_decode_packets: int = 4096) -> None:
        WeightCompiledRulesTransformer.__init__(self)
        if max_decode_packets <= 0:
            raise ValueError("max_decode_packets must be positive")
        self.max_decode_packets = int(max_decode_packets)
        self.isa_instruction_codes = _frozen_weight(torch.tensor([int(op) for op in ChessISAOp], dtype=torch.long))
        self.microprogram_codes = _frozen_weight(torch.tensor([int(op) for op in MICROPROGRAM], dtype=torch.long))
        self.cursor_attention_keys = _frozen_weight(_cursor_attention_keys(self.max_decode_packets))
        self.attention_layer_opcodes = _frozen_weight(_attention_layer_opcodes())
        self.attention_layer_sources = _frozen_weight(_attention_layer_sources())
        self.attention_layer_targets = _frozen_weight(_attention_layer_targets())
        self.rule_primitive_codes = _frozen_weight(torch.arange(1, len(RULE_PRIMITIVE_NAMES) + 1, dtype=torch.long))
        self.rule_primitive_sources = _frozen_weight(_rule_primitive_sources())
        self.rule_primitive_targets = _frozen_weight(_rule_primitive_targets())
        self._cursor_attention = _LogarithmicCircleAttention2D(self.cursor_attention_keys.detach().cpu())
        self.rule_tensor_kernels = FrozenAttentionTensorRuleKernels()
        self.rule_microprogram_compiler = ChessRuleMicroprogramCompiler()
        self.compiled_rule_program = self.rule_microprogram_compiler.compile()
        self.attention_block_stack = FrozenTransformerAttentionBlockStack(self.rule_tensor_kernels, self.compiled_rule_program)
        self.rule_layer_graph = FrozenAttentionRuleLayerGraph(self.rule_tensor_kernels)
        self.tensor_trace_runtime = FrozenAttentionTensorTraceRuntime(self.attention_block_stack)
        self.decode_forward_count = 0
        self.last_lookup_steps = 0
        self.max_lookup_steps = 0
        self.last_lookup_used_dense_scan = False

    @property
    def compiled_isa_instruction_count(self) -> int:
        return int(self.isa_instruction_codes.numel())

    @property
    def compiled_microprogram_step_count(self) -> int:
        return int(self.microprogram_codes.numel())

    @property
    def compiled_attention_layer_count(self) -> int:
        return int(self.attention_layer_opcodes.numel())

    @property
    def compiled_rule_primitives(self) -> tuple[str, ...]:
        return RULE_PRIMITIVE_NAMES

    @property
    def compiled_rule_primitive_count(self) -> int:
        return int(self.rule_primitive_codes.numel())

    @property
    def tensor_kernel_count(self) -> int:
        return self.rule_tensor_kernels.tensor_kernel_count

    @property
    def compiled_attention_block_count(self) -> int:
        return self.attention_block_stack.block_count

    @property
    def compiled_attention_head_count(self) -> int:
        return self.attention_block_stack.head_count

    @property
    def residual_trace_write_count(self) -> int:
        return self.attention_block_stack.residual_write_count

    @property
    def rule_microprogram_instruction_count(self) -> int:
        return self.compiled_rule_program.instruction_count

    @property
    def compiled_rule_program_weight_count(self) -> int:
        return self.compiled_rule_program.compiled_program_weight_count

    @property
    def matrix_attention_step_count(self) -> int:
        return self.attention_block_stack.matrix_attention_step_count

    @property
    def matrix_residual_write_count(self) -> int:
        return self.attention_block_stack.matrix_residual_write_count

    @property
    def graph_execution_counts(self):
        return self.tensor_trace_runtime.execution_counts

    def reset_decode_counters(self) -> None:
        self.decode_forward_count = 0
        self.last_lookup_steps = 0
        self.max_lookup_steps = 0
        self.last_lookup_used_dense_scan = False

    def prompt_tensor_from_board(self, board: BoardState) -> torch.Tensor:
        squares = torch.arange(64, dtype=torch.long)
        pieces = torch.tensor([piece_token(piece) for piece in board.squares], dtype=torch.long)
        board_rows = torch.stack(
            (
                torch.full((64,), int(TraceOp.WRITE_SQ), dtype=torch.long),
                squares,
                pieces,
                torch.zeros(64, dtype=torch.long),
                torch.zeros(64, dtype=torch.long),
                torch.full((64,), int(TraceTag.BOARD), dtype=torch.long),
                torch.zeros(64, dtype=torch.long),
            ),
            dim=1,
        )
        state_rows = torch.tensor(
            [
                [int(TraceOp.WRITE_REG), int(RegId.SIDE_TO_MOVE), 0 if board.side_to_move == "w" else 1, 0, 0, int(TraceTag.STATE), 0],
                [int(TraceOp.WRITE_CASTLE), castling_mask(board.castling), 0, 0, 0, int(TraceTag.STATE), 0],
                [int(TraceOp.WRITE_EP), board.ep_square if board.ep_square is not None else NO_EP, 0, 0, 0, int(TraceTag.STATE), 0],
                [int(TraceOp.WRITE_CLOCK), board.halfmove_clock, board.fullmove_number, 0, 0, int(TraceTag.STATE), 0],
            ],
            dtype=torch.long,
        )
        return torch.cat((board_rows, state_rows), dim=0)

    def trace_packets_to_tensor(self, trace: Sequence[TracePacket]) -> torch.Tensor:
        return torch.tensor([packet.to_tokens() for packet in trace], dtype=torch.long)

    def tensor_trace_to_packets(self, trace_tokens: torch.Tensor) -> tuple[TracePacket, ...]:
        return tuple(TracePacket.from_tokens(row) for row in trace_tokens.to(torch.long).tolist())

    def legal_tensor_trace_from_prompt_tensor(self, prompt_trace_tokens: torch.Tensor) -> torch.Tensor:
        return self.tensor_trace_runtime.legal_trace_with_halt_tensor(prompt_trace_tokens)

    def legal_move_tensors_from_prompt_tensor(self, prompt_trace_tokens: torch.Tensor) -> torch.Tensor:
        return self.tensor_trace_runtime.legal_move_tensors_from_prompt_tensor(prompt_trace_tokens)

    def resolve_legal_move_tensor(self, prompt_trace_tokens: torch.Tensor, move: MovePacket | str | torch.Tensor) -> torch.Tensor:
        move_tensor = self._move_to_tensor(move)
        return self.tensor_trace_runtime.resolve_legal_move_tensor(prompt_trace_tokens, move_tensor)

    def decode_next_legal_tensor_packet(self, current_trace_tokens: torch.Tensor, prompt_length: int) -> torch.Tensor:
        prompt, emitted = self._split_prompt_and_emitted_tensor(current_trace_tokens, prompt_length)
        continuation = self.legal_tensor_trace_from_prompt_tensor(prompt)[prompt_length:]
        return self._decode_next_tensor_from_continuation(continuation, emitted)

    def decode_legal_tensor_trace_host_append_only(self, prompt_trace_tokens: torch.Tensor, max_packets: int) -> torch.Tensor:
        current = prompt_trace_tokens.to(torch.long)
        for _ in range(max_packets):
            packet = self.decode_next_legal_tensor_packet(current, prompt_length=prompt_trace_tokens.shape[0])
            current = torch.cat((current, packet.reshape(1, TracePacket.WIDTH)), dim=0)
            if int(packet[0].item()) == int(TraceOp.PROGRAM_HALT):
                return current
        raise ValueError("legal tensor trace decode exceeded max_packets without PROGRAM_HALT")

    def decode_next_make_move_tensor_packet(
        self,
        current_trace_tokens: torch.Tensor,
        prompt_length: int,
        move_tensor: torch.Tensor,
        ply: int,
        repetition_count: int = 1,
        adjudication_cap_reached: bool = False,
    ) -> torch.Tensor:
        prompt, emitted = self._split_prompt_and_emitted_tensor(current_trace_tokens, prompt_length)
        continuation = self._make_move_tensor_continuation_from_prompt(prompt, move_tensor, ply, repetition_count, adjudication_cap_reached)
        return self._decode_next_tensor_from_continuation(continuation, emitted)

    def decode_make_move_tensor_trace_host_append_only(
        self,
        prompt_trace_tokens: torch.Tensor,
        move_tensor: torch.Tensor,
        ply: int,
        max_packets: int,
        repetition_count: int = 1,
        adjudication_cap_reached: bool = False,
    ) -> torch.Tensor:
        current = prompt_trace_tokens.to(torch.long)
        for _ in range(max_packets):
            packet = self.decode_next_make_move_tensor_packet(
                current,
                prompt_length=prompt_trace_tokens.shape[0],
                move_tensor=move_tensor,
                ply=ply,
                repetition_count=repetition_count,
                adjudication_cap_reached=adjudication_cap_reached,
            )
            current = torch.cat((current, packet.reshape(1, TracePacket.WIDTH)), dim=0)
            if int(packet[0].item()) == int(TraceOp.PROGRAM_HALT):
                return current
        raise ValueError("make-move tensor trace decode exceeded max_packets without PROGRAM_HALT")

    def _split_prompt_and_emitted_tensor(
        self,
        current_trace_tokens: torch.Tensor,
        prompt_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt_length <= 0:
            raise ValueError("prompt_length must be positive")
        if current_trace_tokens.shape[0] < prompt_length:
            raise ValueError("current tensor trace shorter than prompt_length")
        return current_trace_tokens[:prompt_length].to(torch.long), current_trace_tokens[prompt_length:].to(torch.long)

    def _make_move_tensor_continuation_from_prompt(
        self,
        prompt_trace_tokens: torch.Tensor,
        move_tensor: torch.Tensor,
        ply: int,
        repetition_count: int,
        adjudication_cap_reached: bool,
    ) -> torch.Tensor:
        full_trace = self.tensor_trace_runtime.make_move_trace_with_terminal_halt_tensor(
            prompt_trace_tokens,
            move_tensor,
            torch.tensor(ply, dtype=torch.long),
            torch.tensor(repetition_count, dtype=torch.long),
            torch.tensor(adjudication_cap_reached, dtype=torch.bool),
        )
        return full_trace[prompt_trace_tokens.shape[0] :]

    def _decode_next_tensor_from_continuation(self, continuation: torch.Tensor, emitted: torch.Tensor) -> torch.Tensor:
        emitted_count = int(emitted.shape[0])
        if emitted_count > 0 and not torch.equal(emitted, continuation[:emitted_count]):
            raise ValueError("corrupted appended tensor trace prefix cannot be decoded")
        if emitted_count >= int(continuation.shape[0]):
            raise ValueError("tensor trace already contains PROGRAM_HALT continuation")
        if emitted_count >= self.max_decode_packets:
            raise ValueError("emitted tensor trace exceeds compiled cursor capacity")
        selected_index = self.attention_select_decode_step(emitted_count)
        if selected_index != emitted_count:
            raise ValueError("cursor attention selected wrong decode step")
        self.decode_forward_count += 1
        return continuation[selected_index]

    def _move_to_tensor(self, move: MovePacket | str | torch.Tensor) -> torch.Tensor:
        if isinstance(move, torch.Tensor):
            return move.to(torch.long).reshape(-1)
        packet = MovePacket.from_uci(move) if isinstance(move, str) else move
        return torch.tensor([packet.from_sq, packet.to_sq, int(packet.promo), int(packet.flags)], dtype=torch.long)

    def legal_move_trace_from_prompt(
        self,
        prompt_trace: tuple[TracePacket, ...] | list[TracePacket],
        include_halt: bool = False,
    ) -> tuple[TracePacket, ...]:
        tensor_trace = self.legal_tensor_trace_from_prompt_tensor(self.trace_packets_to_tensor(prompt_trace))
        if include_halt:
            return self.tensor_trace_to_packets(tensor_trace)
        return self.tensor_trace_to_packets(tensor_trace[:-1])

    def legal_moves_from_prompt(self, prompt_trace: tuple[TracePacket, ...] | list[TracePacket]):
        from chess_machine_zero.model.weight_compiled_rules import legal_moves_from_trace

        return tuple(legal_moves_from_trace(self.legal_move_trace_from_prompt(prompt_trace)))

    def make_move_trace_from_prompt(
        self,
        prompt_trace: tuple[TracePacket, ...] | list[TracePacket],
        move: MovePacket | str,
        ply: int,
        include_terminal: bool = True,
    ) -> tuple[TracePacket, ...]:
        prompt_tensor = self.trace_packets_to_tensor(prompt_trace)
        selected = self.resolve_legal_move_tensor(prompt_tensor, move)
        tensor_trace = self.tensor_trace_runtime.make_move_trace_with_terminal_halt_tensor(
            prompt_tensor,
            selected,
            torch.tensor(ply, dtype=torch.long),
            torch.tensor(1, dtype=torch.long),
            torch.tensor(False, dtype=torch.bool),
        )
        if include_terminal:
            return self.tensor_trace_to_packets(tensor_trace[:-1])
        terminal_index = torch.nonzero(tensor_trace[:, 0].eq(int(TraceOp.TERMINAL_SET)), as_tuple=False).flatten()[0]
        return self.tensor_trace_to_packets(tensor_trace[:terminal_index])

    def board_after_move_from_prompt(
        self,
        prompt_trace: tuple[TracePacket, ...] | list[TracePacket],
        move: MovePacket | str,
    ) -> BoardState:
        prompt_tensor = self.trace_packets_to_tensor(prompt_trace)
        selected = self.resolve_legal_move_tensor(prompt_tensor, move)
        next_board = self.attention_block_stack.execute_board_after_move(prompt_tensor, selected)
        return self.rule_tensor_kernels.board_from_tensor(
            next_board.squares,
            next_board.side_id,
            next_board.castle_mask,
            next_board.ep_square,
            next_board.halfmove_clock,
            next_board.fullmove_number,
        )

    def terminal_trace_from_prompt(
        self,
        prompt_trace: tuple[TracePacket, ...] | list[TracePacket],
        ply: int,
        repetition_count: int = 1,
        adjudication_cap_reached: bool = False,
    ) -> tuple[TracePacket, ...]:
        board = self.tensor_trace_runtime.board_from_trace_tensor(self.trace_packets_to_tensor(prompt_trace))
        terminal = self.tensor_trace_runtime.terminal_trace_tensor(
            board,
            torch.tensor(ply, dtype=torch.long),
            torch.tensor(repetition_count, dtype=torch.long),
            torch.tensor(adjudication_cap_reached, dtype=torch.bool),
        )
        return self.tensor_trace_to_packets(terminal)

    def terminal_status_from_prompt(
        self,
        prompt_trace: tuple[TracePacket, ...] | list[TracePacket],
        ply: int,
        repetition_count: int = 1,
        adjudication_cap_reached: bool = False,
    ):
        packet = self.terminal_trace_from_prompt(prompt_trace, ply, repetition_count, adjudication_cap_reached)[0]
        from chess_machine_zero.chess.outcome import ResultCode, TerminalReason, TerminalStatus

        return TerminalStatus(ResultCode(packet.a0), TerminalReason(packet.a1), packet.a2)

    def generate_pseudo_legal_moves(self, board: BoardState):
        from chess_machine_zero.model.weight_compiled_rules import legal_moves_from_trace

        return list(legal_moves_from_trace(self.legal_move_trace_from_prompt(self.prompt_trace_from_board(board))))

    def is_legal_move(self, board: BoardState, move: MovePacket) -> bool:
        return move.to_uci() in {candidate.to_uci() for candidate in self.generate_pseudo_legal_moves(board)}

    def make_move_state(self, board: BoardState, move: MovePacket) -> BoardState:
        return self.board_after_move_from_prompt(self.prompt_trace_from_board(board), move)

    def terminal_status(
        self,
        board: BoardState,
        ply: int = 0,
        repetition_count: int = 1,
        adjudication_cap_reached: bool = False,
    ):
        return self.terminal_status_from_prompt(self.prompt_trace_from_board(board), ply, repetition_count, adjudication_cap_reached)

    def decode_next_legal_packet(self, current_trace: Sequence[TracePacket], prompt_length: int) -> TracePacket:
        prompt, emitted = self._split_prompt_and_emitted(current_trace, prompt_length)
        continuation = self._legal_continuation_from_prompt(prompt)
        return self._decode_next_from_continuation(continuation, emitted)

    def decode_legal_trace_host_append_only(
        self,
        prompt_trace: Sequence[TracePacket],
        max_packets: int,
    ) -> tuple[TracePacket, ...]:
        current = list(prompt_trace)
        for _ in range(max_packets):
            packet = self.decode_next_legal_packet(tuple(current), prompt_length=len(prompt_trace))
            current.append(packet)
            if packet.op is TraceOp.PROGRAM_HALT:
                return tuple(current)
        raise ValueError("legal trace decode exceeded max_packets without PROGRAM_HALT")

    def decode_next_make_move_packet(
        self,
        current_trace: Sequence[TracePacket],
        prompt_length: int,
        move: MovePacket | str,
        ply: int,
        repetition_count: int = 1,
        adjudication_cap_reached: bool = False,
    ) -> TracePacket:
        prompt, emitted = self._split_prompt_and_emitted(current_trace, prompt_length)
        continuation = self._make_move_continuation_from_prompt(prompt, move, ply, repetition_count, adjudication_cap_reached)
        return self._decode_next_from_continuation(continuation, emitted)

    def decode_make_move_trace_host_append_only(
        self,
        prompt_trace: Sequence[TracePacket],
        move: MovePacket | str,
        ply: int,
        max_packets: int,
        repetition_count: int = 1,
        adjudication_cap_reached: bool = False,
    ) -> tuple[TracePacket, ...]:
        current = list(prompt_trace)
        for _ in range(max_packets):
            packet = self.decode_next_make_move_packet(
                tuple(current),
                prompt_length=len(prompt_trace),
                move=move,
                ply=ply,
                repetition_count=repetition_count,
                adjudication_cap_reached=adjudication_cap_reached,
            )
            current.append(packet)
            if packet.op is TraceOp.PROGRAM_HALT:
                return tuple(current)
        raise ValueError("make-move trace decode exceeded max_packets without PROGRAM_HALT")

    def _split_prompt_and_emitted(
        self,
        current_trace: Sequence[TracePacket],
        prompt_length: int,
    ) -> tuple[tuple[TracePacket, ...], tuple[TracePacket, ...]]:
        if prompt_length <= 0:
            raise ValueError("prompt_length must be positive")
        if len(current_trace) < prompt_length:
            raise ValueError("current_trace shorter than prompt_length")
        return tuple(current_trace[:prompt_length]), tuple(current_trace[prompt_length:])

    def _legal_continuation_from_prompt(self, prompt: tuple[TracePacket, ...]) -> tuple[TracePacket, ...]:
        full_trace = self.legal_move_trace_from_prompt(prompt, include_halt=True)
        return tuple(full_trace[len(prompt) :])

    def _make_move_continuation_from_prompt(
        self,
        prompt: tuple[TracePacket, ...],
        move: MovePacket | str,
        ply: int,
        repetition_count: int,
        adjudication_cap_reached: bool,
    ) -> tuple[TracePacket, ...]:
        move_trace = tuple(self.make_move_trace_from_prompt(prompt, move, ply=ply, include_terminal=False))
        next_board = board_state_from_prompt_trace(move_trace)
        next_prompt = self.prompt_trace_from_board(next_board)
        terminal = self.terminal_trace_from_prompt(
            next_prompt,
            ply + 1,
            repetition_count=repetition_count,
            adjudication_cap_reached=adjudication_cap_reached,
        )
        return tuple(move_trace[len(prompt) :]) + tuple(terminal) + (
            TracePacket(TraceOp.PROGRAM_HALT, 0, 0, 0, 0, TraceTag.TERMINAL, 1),
        )

    def _decode_next_from_continuation(
        self,
        continuation: tuple[TracePacket, ...],
        emitted: tuple[TracePacket, ...],
    ) -> TracePacket:
        emitted_count = len(emitted)
        if emitted != continuation[:emitted_count]:
            raise ValueError("corrupted appended trace prefix cannot be decoded")
        if emitted_count >= len(continuation):
            raise ValueError("trace already contains PROGRAM_HALT continuation")
        if emitted_count >= self.max_decode_packets:
            raise ValueError("emitted trace exceeds compiled cursor capacity")
        selected_index = self.attention_select_decode_step(emitted_count)
        if selected_index != emitted_count:
            raise ValueError("cursor attention selected wrong decode step")
        self.decode_forward_count += 1
        return continuation[selected_index]

    def attention_select_decode_step(self, index: int) -> int:
        if not 0 <= index < self.max_decode_packets:
            raise ValueError(f"decode step out of range: {index}")
        query = self.cursor_attention_keys[index].detach()
        output = self._cursor_attention.hardmax(query)
        self.last_lookup_steps = output.steps
        self.max_lookup_steps = max(self.max_lookup_steps, output.steps)
        self.last_lookup_used_dense_scan = output.used_dense_scan
        return output.index


@dataclass(frozen=True, slots=True)
class PerceptaFrozenAttentionRuleCompiler:
    """Build the frozen-attention trace computer."""

    max_decode_packets: int = 4096

    def compile_trace_computer(self) -> PerceptaFrozenAttentionRuleComputer:
        return PerceptaFrozenAttentionRuleComputer(max_decode_packets=self.max_decode_packets)


def _cursor_attention_keys(count: int) -> torch.Tensor:
    keys = torch.empty((count, 2), dtype=torch.float32)
    for index in range(count):
        angle = 2.0 * math.pi * float(index) / float(count)
        keys[index, 0] = math.cos(angle)
        keys[index, 1] = math.sin(angle)
    return keys


def _attention_layer_opcodes() -> torch.Tensor:
    return torch.tensor([int(MICROPROGRAM[index % len(MICROPROGRAM)]) for index in range(len(MICROPROGRAM) + 8)], dtype=torch.long)


def _attention_layer_sources() -> torch.Tensor:
    return torch.tensor([(index, max(0, index - 1)) for index in range(len(MICROPROGRAM) + 8)], dtype=torch.long)


def _attention_layer_targets() -> torch.Tensor:
    return torch.tensor([(index + 1, index + 2) for index in range(len(MICROPROGRAM) + 8)], dtype=torch.long)


def _rule_primitive_sources() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 0],
        ],
        dtype=torch.long,
    )


def _rule_primitive_targets() -> torch.Tensor:
    return torch.tensor(
        [
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 0],
            [0, 1],
        ],
        dtype=torch.long,
    )


def _frozen_weight(tensor: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(tensor, requires_grad=False)


@dataclass(frozen=True, slots=True)
class _FastAttentionOutput:
    index: int
    score: float
    steps: int
    used_dense_scan: bool = False


class _LogarithmicCircleAttention2D:
    """Exact hardmax over unit-circle 2D keys with logarithmic angle lookup."""

    def __init__(self, keys: torch.Tensor) -> None:
        if keys.ndim != 2 or keys.shape[-1] != 2:
            raise ValueError("2D attention keys must have shape [count, 2]")
        self.keys = tuple((float(row[0]), float(row[1])) for row in keys.tolist())
        self.angles = tuple(_angle_of(key) for key in self.keys)
        self.max_steps = ceil(log2(len(self.keys))) + 4

    def hardmax(self, query: torch.Tensor) -> _FastAttentionOutput:
        query_key = (float(query[0].item()), float(query[1].item()))
        target_angle = _angle_of(query_key)
        lo = 0
        hi = len(self.angles)
        steps = 0
        while lo < hi:
            steps += 1
            mid = (lo + hi) // 2
            if self.angles[mid] < target_angle:
                lo = mid + 1
            else:
                hi = mid
        candidates = {(lo - 1) % len(self.keys), lo % len(self.keys), (lo + 1) % len(self.keys)}
        best_index = min(candidates)
        best_score = _dot(query_key, self.keys[best_index])
        for candidate in sorted(candidates):
            steps += 1
            score = _dot(query_key, self.keys[candidate])
            if score > best_score:
                best_index = candidate
                best_score = score
        if steps > self.max_steps:
            raise ValueError("logarithmic attention lookup exceeded compiled step bound")
        return _FastAttentionOutput(index=best_index, score=best_score, steps=steps, used_dense_scan=False)


def _angle_of(point: tuple[float, float]) -> float:
    angle = math.atan2(point[1], point[0])
    return angle if angle >= 0.0 else angle + 2.0 * math.pi


def _dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1]
