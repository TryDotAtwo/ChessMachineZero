"""Self-play game and replay records."""

from __future__ import annotations

from dataclasses import dataclass

from chess_machine_zero.chess.move_packet import MovePacket
from chess_machine_zero.chess.outcome import ResultCode, TerminalStatus
from chess_machine_zero.vm.trace_packet import TracePacket


@dataclass(frozen=True, slots=True)
class MoveDecision:
    game_id: int
    ply: int
    side_to_move: str
    fen_before: str
    legal_moves: tuple[MovePacket, ...]
    chosen_move: MovePacket
    logprob_chosen: float
    entropy: float
    temperature: float
    trace: tuple[TracePacket, ...]
    trace_hash_before: str
    trace_hash_after: str
    stochastic_seed: int


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    game_id: int
    ply: int
    side_to_move: str
    fen_before: str
    legal_moves: tuple[MovePacket, ...]
    chosen_move: MovePacket
    logprob_chosen: float
    entropy: float
    temperature: float
    final_outcome_from_side_to_move: float
    trace_hash_before: str
    trace_hash_after: str
    rules_version: str
    model_checkpoint_id: str
    stochastic_seed: int

    @property
    def legal_uci(self) -> tuple[str, ...]:
        return tuple(move.to_uci() for move in self.legal_moves)


@dataclass(frozen=True, slots=True)
class GameRecord:
    game_id: int
    start_fen: str
    final_fen: str
    terminal_status: TerminalStatus
    decisions: tuple[MoveDecision, ...]
    replay_records: tuple[ReplayRecord, ...]
    trace: tuple[TracePacket, ...]


def outcome_from_side(result: ResultCode, side_to_move: str) -> float:
    if result is ResultCode.DRAW:
        return 0.0
    if result is ResultCode.WHITE_WIN:
        return 1.0 if side_to_move == "w" else -1.0
    if result is ResultCode.BLACK_WIN:
        return 1.0 if side_to_move == "b" else -1.0
    return 0.0
