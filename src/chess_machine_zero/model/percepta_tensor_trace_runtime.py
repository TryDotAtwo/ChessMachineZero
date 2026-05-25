"""Tensor-trace runtime for frozen Percepta chess rule blocks."""

from __future__ import annotations

from collections import Counter

import torch
from torch import nn

from chess_machine_zero.model.percepta_attention_block_stack import FrozenTransformerAttentionBlockStack
from chess_machine_zero.model.percepta_attention_rule_kernels import TensorBoardState


PACKET_WIDTH = 7
OP = 0
A0 = 1
A1 = 2
A2 = 3
A3 = 4
TAG = 5
COMMIT = 6
RULE_PRIMITIVE_NAMES = (
    "PIECE_DISPATCH",
    "RAY_SCAN",
    "ATTACK_TEST",
    "LEGAL_FILTER",
    "MAKE_MOVE",
    "TERMINAL_PREDICATES",
)


class FrozenAttentionTensorTraceRuntime(nn.Module):
    """Frozen block runtime with tensor trace input and tensor trace output."""

    core_trace_runtime = "tensor_trace_in_frozen_attention_blocks_tensor_trace_out"
    python_host_boundary_role = "display_only"
    tensor_trace_core_runtime = True
    tracepacket_core_runtime = False

    def __init__(self, attention_block_stack: FrozenTransformerAttentionBlockStack) -> None:
        super().__init__()
        self.attention_block_stack = attention_block_stack
        self.execution_counts: Counter[str] = Counter({name: 0 for name in RULE_PRIMITIVE_NAMES})

    def board_from_trace_tensor(self, trace_tokens: torch.Tensor) -> TensorBoardState:
        return self.attention_block_stack.board_from_trace_tensor(trace_tokens)

    def legal_trace_with_halt_tensor(self, prompt_trace_tokens: torch.Tensor) -> torch.Tensor:
        self.execution_counts["PIECE_DISPATCH"] += 1
        self.execution_counts["RAY_SCAN"] += 1
        self.execution_counts["ATTACK_TEST"] += 1
        self.execution_counts["LEGAL_FILTER"] += 1
        return self.attention_block_stack.legal_trace_with_halt_tensor(prompt_trace_tokens)

    def make_move_trace_with_terminal_halt_tensor(
        self,
        prompt_trace_tokens: torch.Tensor,
        move_tensor: torch.Tensor,
        ply: torch.Tensor,
        repetition_count: torch.Tensor,
        adjudication_cap_reached: torch.Tensor,
    ) -> torch.Tensor:
        self.execution_counts["MAKE_MOVE"] += 1
        self.execution_counts["TERMINAL_PREDICATES"] += 1
        return self.attention_block_stack.make_move_trace_with_terminal_halt_tensor(
            prompt_trace_tokens,
            move_tensor,
            ply,
            repetition_count,
            adjudication_cap_reached,
        )

    def terminal_trace_tensor(
        self,
        board: TensorBoardState,
        ply: torch.Tensor,
        repetition_count: torch.Tensor,
        adjudication_cap_reached: torch.Tensor,
    ) -> torch.Tensor:
        self.execution_counts["TERMINAL_PREDICATES"] += 1
        return self.attention_block_stack.terminal_trace_tensor(board, ply, repetition_count, adjudication_cap_reached)

    def board_transition_tensor(self, before: TensorBoardState, after: TensorBoardState, ply: torch.Tensor) -> torch.Tensor:
        return self.attention_block_stack.board_transition_tensor(before, after, ply)

    def resolve_legal_move_tensor(self, prompt_trace_tokens: torch.Tensor, move_tensor: torch.Tensor) -> torch.Tensor:
        return self.attention_block_stack.resolve_legal_move_tensor(prompt_trace_tokens, move_tensor)

    def legal_move_tensors_from_prompt_tensor(self, prompt_trace_tokens: torch.Tensor) -> torch.Tensor:
        return self.attention_block_stack.legal_move_tensors_from_prompt_tensor(prompt_trace_tokens)
