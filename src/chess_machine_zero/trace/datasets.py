"""Deterministic trace datasets for next-packet prediction."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from chess_machine_zero.vm.trace_packet import TracePacket


def packets_to_tensor(packets: Sequence[TracePacket]) -> torch.Tensor:
    if not packets:
        raise ValueError("at least one packet is required")
    return torch.tensor([packet.to_tokens() for packet in packets], dtype=torch.long)


def next_packet_training_tensors(packets: Sequence[TracePacket]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(packets) < 2:
        raise ValueError("at least two packets are required for next-packet training")
    tokens = packets_to_tensor(packets)
    return tokens[:-1].unsqueeze(0), tokens[1:].unsqueeze(0)


def trace_field_vocab_sizes(packets: Sequence[TracePacket], minimum_size: int = 2) -> tuple[int, ...]:
    tokens = packets_to_tensor(packets)
    max_by_field = tokens.max(dim=0).values
    return tuple(max(int(value) + 1, minimum_size) for value in max_by_field)
