"""Exact self-play game integrity checks."""

from __future__ import annotations

from dataclasses import dataclass

from chess_machine_zero.selfplay.game_record import GameRecord
from chess_machine_zero.vm.trace_hash import trace_hash_hex


class SelfPlayAuditError(AssertionError):
    """Raised when a generated game violates self-play trace invariants."""


@dataclass(frozen=True, slots=True)
class GameAudit:
    game_id: int
    decision_count: int
    replay_count: int
    illegal_commit_count: int
    missing_outcome_count: int
    terminal: bool
    trace_hash: str


def audit_game_record(game: GameRecord) -> GameAudit:
    errors: list[str] = []
    illegal_commit_count = 0
    missing_outcome_count = 0
    if not game.terminal_status.is_terminal:
        errors.append("terminal_status_is_not_terminal")
    if len(game.decisions) != len(game.replay_records):
        errors.append(f"decision_replay_count_mismatch:{len(game.decisions)}!={len(game.replay_records)}")
    for decision in game.decisions:
        legal_uci = {move.to_uci() for move in decision.legal_moves}
        if decision.chosen_move.to_uci() not in legal_uci:
            illegal_commit_count += 1
            errors.append(f"illegal_commit:ply={decision.ply}:move={decision.chosen_move.to_uci()}")
    for record in game.replay_records:
        if record.final_outcome_from_side_to_move not in (-1.0, 0.0, 1.0):
            missing_outcome_count += 1
            errors.append(f"invalid_outcome:ply={record.ply}:value={record.final_outcome_from_side_to_move}")
        if record.chosen_move.to_uci() not in record.legal_uci:
            illegal_commit_count += 1
            errors.append(f"illegal_replay_commit:ply={record.ply}:move={record.chosen_move.to_uci()}")
    if errors:
        raise SelfPlayAuditError(";".join(errors))
    return GameAudit(
        game_id=game.game_id,
        decision_count=len(game.decisions),
        replay_count=len(game.replay_records),
        illegal_commit_count=illegal_commit_count,
        missing_outcome_count=missing_outcome_count,
        terminal=game.terminal_status.is_terminal,
        trace_hash=game_record_trace_hash(game),
    )


def game_record_trace_hash(game: GameRecord) -> str:
    return trace_hash_hex(game.trace)
