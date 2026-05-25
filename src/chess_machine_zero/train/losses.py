"""Self-play losses for ranker and outcome baseline."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import nn

from chess_machine_zero.model.baseline import CMZOutcomeBaseline
from chess_machine_zero.model.ranker import CMZMoveRanker
from chess_machine_zero.selfplay.game_record import ReplayRecord


@dataclass(frozen=True, slots=True)
class SelfPlayTrainStats:
    total_loss: float
    policy_loss: float
    baseline_loss: float
    entropy: float


def train_ranker_baseline_step(
    ranker: CMZMoveRanker,
    baseline: CMZOutcomeBaseline,
    records: Sequence[ReplayRecord],
    optimizer: torch.optim.Optimizer,
    entropy_weight: float = 0.01,
) -> SelfPlayTrainStats:
    if not records:
        raise ValueError("records must be nonempty")
    ranker.train()
    baseline.train()
    optimizer.zero_grad(set_to_none=True)
    policy_losses = []
    baseline_losses = []
    entropies = []
    for record in records:
        side_id_value = 0 if record.side_to_move == "w" else 1
        move_ids = torch.tensor([move.move_id for move in record.legal_moves], dtype=torch.long)
        flags = torch.tensor([int(move.flags) for move in record.legal_moves], dtype=torch.long)
        sides = torch.full_like(move_ids, side_id_value)
        scores = ranker(move_ids, flags, sides)
        log_probs = torch.log_softmax(scores / max(record.temperature, 1e-6), dim=0)
        probabilities = torch.softmax(scores / max(record.temperature, 1e-6), dim=0)
        chosen_index = record.legal_uci.index(record.chosen_move.to_uci())
        side_tensor = torch.tensor([side_id_value], dtype=torch.long)
        ply_tensor = torch.tensor([record.ply], dtype=torch.float32)
        baseline_pred = baseline(side_tensor, ply_tensor).squeeze(0)
        outcome = torch.tensor(record.final_outcome_from_side_to_move, dtype=torch.float32)
        advantage = outcome - baseline_pred.detach()
        policy_losses.append(-advantage * log_probs[chosen_index])
        baseline_losses.append(nn.functional.mse_loss(baseline_pred, outcome))
        entropies.append(-(probabilities * log_probs).sum())
    policy_loss = torch.stack(policy_losses).mean()
    baseline_loss = torch.stack(baseline_losses).mean()
    entropy = torch.stack(entropies).mean()
    total_loss = policy_loss + baseline_loss - entropy_weight * entropy
    total_loss.backward()
    optimizer.step()
    return SelfPlayTrainStats(
        total_loss=float(total_loss.detach().item()),
        policy_loss=float(policy_loss.detach().item()),
        baseline_loss=float(baseline_loss.detach().item()),
        entropy=float(entropy.detach().item()),
    )
