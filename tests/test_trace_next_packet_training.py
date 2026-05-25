from __future__ import annotations

import torch

from chess_machine_zero.model.machine_transformer import CMZMachineTransformer
from chess_machine_zero.train.trainer import train_next_packet_overfit
from chess_machine_zero.trace.datasets import next_packet_training_tensors, trace_field_vocab_sizes
from chess_machine_zero.vm.trace_packet import TraceOp, TracePacket, TraceTag


def test_machine_transformer_overfits_tiny_trace_dataset() -> None:
    torch.manual_seed(20260524)
    packets = [
        TracePacket(TraceOp.WRITE_SQ, 0, 1, 0, 0, TraceTag.BOARD, 0),
        TracePacket(TraceOp.WRITE_SQ, 1, 2, 0, 0, TraceTag.BOARD, 0),
        TracePacket(TraceOp.CANDIDATE, 15, 0, 1, 0, TraceTag.MOVE, 0),
        TracePacket(TraceOp.LEGAL_SET, 15, 1, 0, 0, TraceTag.LEGAL, 0),
        TracePacket(TraceOp.TERMINAL_SET, 0, 0, 0, 0, TraceTag.TERMINAL, 0),
    ]
    inputs, targets = next_packet_training_tensors(packets)
    model = CMZMachineTransformer(
        field_vocab_sizes=trace_field_vocab_sizes(packets, minimum_size=8),
        d_model=16,
        n_heads=8,
        n_layers=1,
        d_ff=64,
        max_seq_len=8,
        dropout=0.0,
    )

    history = train_next_packet_overfit(model, inputs, targets, steps=120, lr=0.05, seed=20260524)

    assert history.initial_loss > 0.5
    assert history.final_loss < 0.08
    assert history.final_loss < history.initial_loss
