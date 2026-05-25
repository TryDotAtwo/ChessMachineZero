"""Outcome baseline trained from self-play outcomes."""

from __future__ import annotations

import torch
from torch import nn

from chess_machine_zero.rng import DEFAULT_SEED


class CMZOutcomeBaseline(nn.Module):
    """Predicts final self-play outcome from decision metadata."""

    def __init__(self, seed: int = DEFAULT_SEED, d_model: int = 16) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.side_embedding = nn.Embedding(2, d_model)
            self.net = nn.Sequential(
                nn.Linear(d_model + 1, d_model),
                nn.Tanh(),
                nn.Linear(d_model, 1),
                nn.Tanh(),
            )

    def forward(self, side_ids: torch.Tensor, plies: torch.Tensor) -> torch.Tensor:
        side_hidden = self.side_embedding(side_ids.long())
        ply_feature = plies.float().unsqueeze(-1) / 256.0
        return self.net(torch.cat([side_hidden, ply_feature], dim=-1)).squeeze(-1)
