"""Development reference for the VM's hard-forward floating-backward primitives."""

from __future__ import annotations

import torch

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def masked_hardmax_ste(
    logits: torch.Tensor, mask: torch.Tensor, temperature: float
) -> torch.Tensor:
    if logits.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("hardmax mask must be boolean and match logits")
    if temperature <= 0.0 or not bool(mask.any(dim=-1).all()):
        raise ValueError("hardmax temperature and mask rows must be valid")
    masked = logits.masked_fill(~mask, -torch.inf)
    probabilities = torch.softmax(masked / temperature, dim=-1)
    indices = masked.argmax(dim=-1, keepdim=True)
    hard = torch.zeros_like(logits).scatter(-1, indices, 1.0)
    return hard + probabilities - probabilities.detach()


def fp4_ste(values: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0.0:
        raise ValueError("FP4 scale must be positive")
    positive = torch.tensor(_E2M1, dtype=values.dtype, device=values.device)
    lattice = torch.cat((-positive[1:].flip(0), positive)) * scale
    indices = (values.unsqueeze(-1) - lattice).abs().argmin(dim=-1)
    quantized = lattice[indices]
    return quantized + values - values.detach()
