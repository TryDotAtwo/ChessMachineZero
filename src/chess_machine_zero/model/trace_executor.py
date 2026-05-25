"""Trace executor facade for short deterministic VM programs."""

from __future__ import annotations

from dataclasses import dataclass

from chess_machine_zero.vm.interpreter import ChessMachineVM
from chess_machine_zero.vm.trace_packet import TracePacket


@dataclass(frozen=True, slots=True)
class CMZTraceExecutor:
    """Milestone 3 executor facade with trace output as the only interface."""

    vm: ChessMachineVM

    def execute_legal_generator(self, fen: str) -> list[TracePacket]:
        return self.vm.legal_move_trace(fen)

    def execute_make_move(self, fen: str, uci: str, ply: int = 0) -> list[TracePacket]:
        return self.vm.make_move_trace(fen, uci, ply)
