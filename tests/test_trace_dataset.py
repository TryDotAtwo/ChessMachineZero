from __future__ import annotations

import torch

from chess_machine_zero.trace.datasets import next_packet_training_tensors, trace_field_vocab_sizes
from chess_machine_zero.vm.trace_packet import TraceOp, TracePacket, TraceTag


def test_next_packet_training_tensors_shift_packets() -> None:
    packets = [
        TracePacket(TraceOp.WRITE_SQ, 0, 1, 0, 0, TraceTag.BOARD, 0),
        TracePacket(TraceOp.WRITE_SQ, 1, 2, 0, 0, TraceTag.BOARD, 0),
        TracePacket(TraceOp.CANDIDATE, 15, 0, 1, 0, TraceTag.MOVE, 0),
    ]
    inputs, targets = next_packet_training_tensors(packets)

    assert inputs.shape == (1, 2, 7)
    assert targets.shape == (1, 2, 7)
    assert torch.equal(inputs[0, 0], torch.tensor(packets[0].to_tokens()))
    assert torch.equal(targets[0, 1], torch.tensor(packets[2].to_tokens()))


def test_trace_field_vocab_sizes_cover_max_token_plus_one() -> None:
    packets = [
        TracePacket(TraceOp.CANDIDATE, 15, 3, 7, 4, TraceTag.MOVE, 8),
        TracePacket(TraceOp.LEGAL_SET, 15, 1, 0, 0, TraceTag.LEGAL, 0),
    ]
    sizes = trace_field_vocab_sizes(packets, minimum_size=2)

    assert sizes[0] > int(TraceOp.LEGAL_SET)
    assert sizes[1] == 16
    assert sizes[4] == 5
    assert sizes[5] > int(TraceTag.LEGAL)
    assert sizes[6] == 9
