"""Replay storage for self-play records."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from chess_machine_zero.selfplay.game_record import GameRecord, ReplayRecord


@dataclass(slots=True)
class ReplayStore:
    records: list[ReplayRecord] = field(default_factory=list)

    def add_game(self, game: GameRecord) -> None:
        self.records.extend(game.replay_records)

    def sample_deterministic(self, batch_size: int, seed: int) -> tuple[ReplayRecord, ...]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.records:
            raise ValueError("ReplayStore is empty")
        rng = random.Random(seed)
        if batch_size >= len(self.records):
            return tuple(self.records)
        indices = sorted(rng.sample(range(len(self.records)), batch_size))
        return tuple(self.records[index] for index in indices)
