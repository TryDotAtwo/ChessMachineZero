"""Top-k retrieval plus local softmax for 2D ranking heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from chess_machine_zero.hullkv.nested_hulls import NestedHullTopK2D


@dataclass(frozen=True, slots=True)
class SparseTopKAttentionOutput:
    indices: tuple[int, ...]
    weights: torch.Tensor
    value: torch.Tensor


class NestedHullTopKAttention2D:
    """Uses NestedHullTopK2D retrieval before applying local softmax."""

    def __call__(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, k: int) -> SparseTopKAttentionOutput:
        if query.shape != (2,) or keys.ndim != 2 or keys.shape[-1] != 2:
            raise ValueError("NestedHullTopKAttention2D expects query [2] and keys [n, 2]")
        if values.shape[0] != keys.shape[0]:
            raise ValueError("values first dimension must match keys")
        retriever = NestedHullTopK2D(tuple((float(x), float(y)) for x, y in keys.tolist()))
        indices = retriever.topk((float(query[0].item()), float(query[1].item())), k)
        index_tensor = torch.tensor(indices, dtype=torch.long, device=keys.device)
        local_scores = keys.index_select(0, index_tensor) @ query
        weights = torch.softmax(local_scores, dim=0)
        selected_values = values.index_select(0, index_tensor)
        return SparseTopKAttentionOutput(indices=indices, weights=weights, value=weights @ selected_values)
