"""Dense exact hardmax attention for 2D executor lookups."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DenseHardmax2DOutput:
    scores: torch.Tensor
    indices: torch.Tensor
    values: torch.Tensor | None


class DenseHardmax2D:
    """Exact brute-force 2D hardmax over a key prefix."""

    def __call__(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor | None = None,
    ) -> DenseHardmax2DOutput:
        if query.shape[-1] != 2 or keys.shape[-1] != 2:
            raise ValueError("DenseHardmax2D requires query and key tensors with final dimension 2D")
        if keys.ndim != 2:
            raise ValueError("DenseHardmax2D requires keys shaped [prefix, 2D]")
        if query.ndim == 1:
            query = query.unsqueeze(0)
        scores = query @ keys.transpose(-1, -2)
        indices = scores.argmax(dim=-1)
        selected_values = None
        if values is not None:
            if values.shape[0] != keys.shape[0]:
                raise ValueError("values first dimension must match key prefix length")
            selected_values = values[indices]
        return DenseHardmax2DOutput(scores=scores, indices=indices, values=selected_values)
