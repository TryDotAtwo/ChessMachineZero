from __future__ import annotations

import pytest
import torch

from chess_machine_zero.model.machine_transformer import CMZMachineTransformer


def test_machine_transformer_requires_head_dim_two() -> None:
    with pytest.raises(AssertionError):
        CMZMachineTransformer(field_vocab_sizes=(8, 8, 8, 8, 8, 8, 8), d_model=12, n_heads=3)


def test_machine_transformer_field_logits_shapes() -> None:
    torch.manual_seed(1234)
    model = CMZMachineTransformer(
        field_vocab_sizes=(17, 32, 32, 32, 32, 8, 8),
        d_model=16,
        n_heads=8,
        n_layers=1,
        d_ff=32,
        max_seq_len=16,
        dropout=0.0,
    )
    tokens = torch.randint(0, 8, (2, 5, 7))
    output = model(tokens)

    assert len(output.field_logits) == 7
    assert output.packet_width == 7
    assert output.hidden.shape == (2, 5, 16)
    assert output.field_logits[0].shape == (2, 5, 17)
    assert output.field_logits[6].shape == (2, 5, 8)
