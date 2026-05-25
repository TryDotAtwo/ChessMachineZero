"""Small deterministic next-packet training utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chess_machine_zero.model.machine_transformer import CMZMachineTransformer


@dataclass(frozen=True, slots=True)
class NextPacketTrainingHistory:
    initial_loss: float
    final_loss: float
    losses: tuple[float, ...]
    exact_match: bool


def trace_next_packet_loss(
    model: CMZMachineTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    output = model(inputs)
    losses = []
    for field_index, logits in enumerate(output.field_logits):
        field_target = targets[:, :, field_index]
        flat_loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            field_target.reshape(-1),
            reduction="none",
        )
        if loss_mask is not None:
            flat_mask = loss_mask.reshape(-1).to(flat_loss.device)
            if not bool(flat_mask.any()):
                raise ValueError("loss_mask must select at least one packet position")
            losses.append(flat_loss[flat_mask].mean())
        else:
            losses.append(flat_loss.mean())
    return torch.stack(losses).mean()


def train_next_packet_overfit(
    model: CMZMachineTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    lr: float,
    seed: int,
    loss_mask: torch.Tensor | None = None,
) -> NextPacketTrainingHistory:
    torch.manual_seed(seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses: list[float] = []
    with torch.no_grad():
        initial_loss = float(trace_next_packet_loss(model, inputs, targets, loss_mask).item())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = trace_next_packet_loss(model, inputs, targets, loss_mask)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    with torch.no_grad():
        final_loss = float(trace_next_packet_loss(model, inputs, targets, loss_mask).item())
        exact_match = next_packet_predictions_exact(model, inputs, targets, loss_mask)
    return NextPacketTrainingHistory(
        initial_loss=initial_loss,
        final_loss=final_loss,
        losses=tuple(losses),
        exact_match=exact_match,
    )


def next_packet_predictions_exact(
    model: CMZMachineTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> bool:
    output = model(inputs)
    predictions = torch.stack([logits.argmax(dim=-1) for logits in output.field_logits], dim=-1)
    matches = predictions.eq(targets)
    if loss_mask is not None:
        mask = loss_mask.to(matches.device).unsqueeze(-1).expand_as(matches)
        return bool(matches[mask].all().item())
    return bool(matches.all().item())
