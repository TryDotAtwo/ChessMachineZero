"""Internal trace lookahead programs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from chess_machine_zero.chess.board_io import parse_fen
from chess_machine_zero.chess.move_packet import MovePacket
from chess_machine_zero.model.baseline import CMZOutcomeBaseline
from chess_machine_zero.selfplay.game_record import outcome_from_side
from chess_machine_zero.trace.windows import TraceWindow
from chess_machine_zero.vm.decision_program import trace_negamax_program
from chess_machine_zero.vm.interpreter import ChessMachineVM, legal_moves_from_trace, terminal_status
from chess_machine_zero.vm.trace_packet import TraceOp, TracePacket, TraceTag


@dataclass(frozen=True, slots=True)
class LookaheadChild:
    move: MovePacket
    value: float
    child_fen: str
    make_move_trace: tuple[TracePacket, ...]


@dataclass(frozen=True, slots=True)
class LookaheadResult:
    depth: int
    value: float
    trace: tuple[TracePacket, ...]
    legal_moves: tuple[MovePacket, ...]
    children: tuple[LookaheadChild, ...]


@dataclass(frozen=True, slots=True)
class InternalTraceLookahead:
    """Depth-limited internal program represented as trace packets."""

    vm: ChessMachineVM
    baseline: CMZOutcomeBaseline
    max_trace_packets: int = 4096

    def trace_negamax(self, fen: str, depth: int, ply: int = 0) -> LookaheadResult:
        if depth < 0:
            raise ValueError("depth must be nonnegative")
        trace_negamax_program()
        board = parse_fen(fen)
        terminal = terminal_status(board, ply)
        if terminal.is_terminal:
            trace = self.vm.terminal_trace(board, ply)
            value = outcome_from_side(terminal.result, board.side_to_move)
            return LookaheadResult(depth=depth, value=value, trace=tuple(trace), legal_moves=(), children=())
        if depth == 0:
            value = self._baseline_value(board.side_to_move, ply)
            score_packet = TracePacket(TraceOp.SCORE_SET, 0, _score_bucket(value), depth, ply, TraceTag.MOVE, 0)
            return LookaheadResult(depth=0, value=value, trace=(score_packet,), legal_moves=(), children=())

        legal_trace = self.vm.legal_move_trace(fen)
        legal_moves = tuple(legal_moves_from_trace(legal_trace))
        trace_window = TraceWindow(self.max_trace_packets)
        trace_window.extend(legal_trace)
        children: list[LookaheadChild] = []
        child_scores: list[float] = []
        for move in legal_moves:
            make_trace = tuple(self.vm.make_move_trace(fen, move, ply))
            child_fen = self.vm.make_move(fen, move)
            child = self.trace_negamax(child_fen, depth - 1, ply + 1)
            score = -child.value
            children.append(LookaheadChild(move=move, value=score, child_fen=child_fen, make_move_trace=make_trace))
            child_scores.append(score)
            trace_window.extend(make_trace)
            trace_window.append(TracePacket(TraceOp.SCORE_SET, move.move_id, _score_bucket(score), depth, ply, TraceTag.MOVE, 0))
        value = max(child_scores) if child_scores else self._baseline_value(board.side_to_move, ply)
        return LookaheadResult(
            depth=depth,
            value=value,
            trace=trace_window.to_tuple(),
            legal_moves=legal_moves,
            children=tuple(children),
        )

    def _baseline_value(self, side_to_move: str, ply: int) -> float:
        side_id = torch.tensor([0 if side_to_move == "w" else 1], dtype=torch.long)
        ply_tensor = torch.tensor([ply], dtype=torch.float32)
        with torch.no_grad():
            return float(self.baseline(side_id, ply_tensor).item())


def _score_bucket(score: float) -> int:
    return max(0, min(2**31 - 1, int(round((score + 32.0) * 1024.0))))
