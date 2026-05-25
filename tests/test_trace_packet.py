from __future__ import annotations

import pytest

from chess_machine_zero.vm.trace_packet import TraceOp, TracePacket, TraceTag


def test_trace_packet_fixed_width_round_trip() -> None:
    packet = TracePacket(TraceOp.CANDIDATE, 100, 12, 28, 4, TraceTag.MOVE, 8)
    assert packet.to_tuple() == (int(TraceOp.CANDIDATE), 100, 12, 28, 4, int(TraceTag.MOVE), 8)
    assert packet.to_tokens() == packet.to_tuple()
    assert TracePacket.from_tuple(packet.to_tuple()) == packet
    assert TracePacket.from_tokens(packet.to_tokens()) == packet


def test_trace_packet_rejects_bad_width_and_negative_field() -> None:
    with pytest.raises(ValueError):
        TracePacket.from_tuple([1, 2, 3])
    with pytest.raises(ValueError):
        TracePacket(TraceOp.WRITE_SQ, -1)
