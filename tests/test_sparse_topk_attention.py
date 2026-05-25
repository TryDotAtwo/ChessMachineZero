from __future__ import annotations

import torch

from chess_machine_zero.model.sparse_topk_attention import NestedHullTopKAttention2D


def test_nested_hull_topk_attention_matches_dense_local_softmax_on_retrieved_set() -> None:
    query = torch.tensor([1.0, 2.0])
    keys = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
            [-2.0, 3.0],
        ]
    )
    values = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
            [-2.0, 3.0],
        ]
    )
    output = NestedHullTopKAttention2D()(query, keys, values, k=3)
    index_tensor = torch.tensor(output.indices)
    expected_scores = keys.index_select(0, index_tensor) @ query
    expected_weights = torch.softmax(expected_scores, dim=0)
    expected_value = expected_weights @ values.index_select(0, index_tensor)

    assert torch.allclose(output.weights, expected_weights)
    assert torch.allclose(output.value, expected_value)
