from __future__ import annotations

import torch

from chess_machine_zero.model.hardmax_attention import DenseHardmax2D


def test_dense_hardmax_2d_returns_manual_argmax_and_values() -> None:
    query = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
    keys = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
        ]
    )
    values = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    output = DenseHardmax2D()(query, keys, values)

    manual_scores = query @ keys.t()
    assert torch.equal(output.indices, manual_scores.argmax(dim=-1))
    assert torch.equal(output.values, values[output.indices])


def test_dense_hardmax_2d_rejects_non_2d_keys() -> None:
    query = torch.zeros(1, 3)
    keys = torch.zeros(2, 3)
    try:
        DenseHardmax2D()(query, keys)
    except ValueError as exc:
        assert "2D" in str(exc)
    else:
        raise AssertionError("DenseHardmax2D accepted non-2D query/key tensors")
