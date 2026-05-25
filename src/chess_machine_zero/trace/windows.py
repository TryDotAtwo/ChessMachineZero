"""Trace-window memory controls."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from chess_machine_zero.vm.trace_packet import TracePacket


@dataclass(slots=True)
class TraceWindow:
    max_packets: int
    _packets: deque[TracePacket] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_packets <= 0:
            raise ValueError("max_packets must be positive")
        self._packets = deque(maxlen=self.max_packets)

    def append(self, packet: TracePacket) -> None:
        self._packets.append(packet)

    def extend(self, packets: Iterable[TracePacket]) -> None:
        for packet in packets:
            self.append(packet)

    def to_tuple(self) -> tuple[TracePacket, ...]:
        return tuple(self._packets)
