"""Trace compilation for transformer-hosted VM training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from chess_machine_zero.chess.board_io import parse_fen
from chess_machine_zero.vm.interpreter import ChessMachineVM
from chess_machine_zero.vm.trace_packet import TracePacket


@dataclass(frozen=True, slots=True)
class LegalTraceExample:
    fen: str
    prompt_trace: tuple[TracePacket, ...]
    full_trace: tuple[TracePacket, ...]

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_trace)

    @property
    def continuation_trace(self) -> tuple[TracePacket, ...]:
        return self.full_trace[self.prompt_length :]


@dataclass(frozen=True, slots=True)
class NextPacketBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    loss_mask: torch.Tensor
    lengths: tuple[int, ...]
    field_vocab_sizes: tuple[int, ...]


def compile_legal_trace_examples(
    fens: Sequence[str],
    vm: ChessMachineVM,
    include_halt: bool,
) -> tuple[LegalTraceExample, ...]:
    if not fens:
        raise ValueError("at least one FEN is required")
    examples: list[LegalTraceExample] = []
    for fen in fens:
        board = parse_fen(fen)
        prompt_trace = tuple(vm.initial_board_trace(board))
        full_trace = tuple(vm.legal_move_trace(fen, include_halt=include_halt))
        if full_trace[: len(prompt_trace)] != prompt_trace:
            raise ValueError("compiled full trace does not begin with prompt trace")
        if len(full_trace) <= len(prompt_trace):
            raise ValueError("compiled trace has no continuation packets")
        examples.append(LegalTraceExample(fen=fen, prompt_trace=prompt_trace, full_trace=full_trace))
    return tuple(examples)


def compile_next_packet_batch(examples: Sequence[LegalTraceExample], minimum_vocab_size: int = 2) -> NextPacketBatch:
    if not examples:
        raise ValueError("at least one trace example is required")
    max_training_len = max(len(example.full_trace) - 1 for example in examples)
    packet_width = TracePacket.WIDTH
    inputs = torch.zeros((len(examples), max_training_len, packet_width), dtype=torch.long)
    targets = torch.zeros((len(examples), max_training_len, packet_width), dtype=torch.long)
    loss_mask = torch.zeros((len(examples), max_training_len), dtype=torch.bool)
    lengths: list[int] = []
    for row, example in enumerate(examples):
        tokens = torch.tensor([packet.to_tokens() for packet in example.full_trace], dtype=torch.long)
        training_len = tokens.shape[0] - 1
        lengths.append(training_len)
        inputs[row, :training_len] = tokens[:-1]
        targets[row, :training_len] = tokens[1:]
        loss_start = example.prompt_length - 1
        loss_mask[row, loss_start:training_len] = True
    max_by_field = torch.cat([inputs.reshape(-1, packet_width), targets.reshape(-1, packet_width)], dim=0).max(dim=0).values
    field_vocab_sizes = tuple(max(int(value) + 1, minimum_vocab_size) for value in max_by_field)
    return NextPacketBatch(
        inputs=inputs,
        targets=targets,
        loss_mask=loss_mask,
        lengths=tuple(lengths),
        field_vocab_sizes=field_vocab_sizes,
    )
