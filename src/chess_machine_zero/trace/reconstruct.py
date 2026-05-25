"""Trace reconstruction for board square writes."""

from __future__ import annotations

from collections.abc import Iterable

from chess_machine_zero.chess.board_io import piece_from_token
from chess_machine_zero.vm.trace_packet import RegId, TraceOp, TracePacket


def reconstruct_board_squares(trace: Iterable[TracePacket]) -> tuple[str | None, ...]:
    squares: list[str | None] = [None] * 64
    for packet in trace:
        if packet.op is TraceOp.WRITE_SQ:
            if not 0 <= packet.a0 < 64:
                raise ValueError(f"WRITE_SQ square out of range: {packet.a0}")
            squares[packet.a0] = piece_from_token(packet.a1)
    return tuple(squares)


def reconstruct_side_to_move(trace: Iterable[TracePacket]) -> str | None:
    side: str | None = None
    for packet in trace:
        if packet.op is TraceOp.WRITE_REG and packet.a0 == int(RegId.SIDE_TO_MOVE):
            side = "w" if packet.a1 == 0 else "b"
    return side
