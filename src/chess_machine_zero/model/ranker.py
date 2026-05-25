"""Trainable legal-move ranker interface."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from chess_machine_zero.chess.move_packet import MovePacket
from chess_machine_zero.rng import DEFAULT_SEED


MOVE_VOCAB_SIZE = 64 * 64 * 5


class CMZMoveRanker(nn.Module):
    """Scores VM-emitted legal move packets without enumerating illegal actions."""

    def __init__(self, seed: int = DEFAULT_SEED, d_model: int = 32, move_vocab_size: int = MOVE_VOCAB_SIZE) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.move_embedding = nn.Embedding(move_vocab_size, d_model)
            self.flag_embedding = nn.Embedding(64, d_model)
            self.side_embedding = nn.Embedding(2, d_model)
            self.net = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )

    def forward(
        self,
        move_ids: torch.Tensor,
        flags: torch.Tensor | None = None,
        side_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        move_ids = move_ids.long()
        if flags is None:
            flags = torch.zeros_like(move_ids)
        if side_ids is None:
            side_ids = torch.zeros_like(move_ids)
        hidden = self.move_embedding(move_ids) + self.flag_embedding(flags.long()) + self.side_embedding(side_ids.long())
        return self.net(hidden).squeeze(-1)

    def score_moves(self, moves: Sequence[MovePacket], side_to_move: str) -> torch.Tensor:
        if not moves:
            raise ValueError("CMZMoveRanker requires at least one legal move")
        side_id = 0 if side_to_move == "w" else 1
        move_ids = torch.tensor([move.move_id for move in moves], dtype=torch.long)
        flags = torch.tensor([int(move.flags) for move in moves], dtype=torch.long)
        sides = torch.full_like(move_ids, side_id)
        return self(move_ids, flags, sides)
