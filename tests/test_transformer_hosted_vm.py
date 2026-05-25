from __future__ import annotations

import torch
import pytest

from chess_machine_zero.chess.board_io import parse_fen
from chess_machine_zero.model.checkpoint import load_transformer_checkpoint, save_transformer_checkpoint
from chess_machine_zero.model.hosted_vm import TransformerHostedVM
from chess_machine_zero.model.machine_transformer import CMZMachineTransformer
from chess_machine_zero.train.trainer import train_next_packet_overfit
from chess_machine_zero.trace.datasets import next_packet_training_tensors, trace_field_vocab_sizes
from chess_machine_zero.trace.verifier import legal_trace_matches_oracle
from chess_machine_zero.vm.interpreter import ChessMachineVM


@pytest.fixture(scope="module")
def trained_hosted_vm_case():
    torch.manual_seed(20260524)
    fen = "8/8/8/8/8/8/8/4K2k w - - 0 1"
    host_vm = ChessMachineVM(seed=20260524)
    board = parse_fen(fen)
    full_trace = host_vm.legal_move_trace(fen, include_halt=True)
    prompt_trace = tuple(host_vm.initial_board_trace(board))
    expected_continuation = tuple(full_trace[len(prompt_trace) :])
    inputs, targets = next_packet_training_tensors(full_trace)
    loss_mask = torch.zeros(inputs.shape[:2], dtype=torch.bool)
    loss_mask[:, len(prompt_trace) - 1 :] = True
    model = CMZMachineTransformer(
        field_vocab_sizes=trace_field_vocab_sizes(full_trace, minimum_size=16),
        d_model=32,
        n_heads=16,
        n_layers=2,
        d_ff=96,
        max_seq_len=128,
        dropout=0.0,
    )

    history = train_next_packet_overfit(model, inputs, targets, steps=600, lr=0.02, seed=20260524, loss_mask=loss_mask)
    return model, prompt_trace, expected_continuation, fen, history


def test_transformer_hosted_vm_decodes_legal_trace_without_host_vm_dependency(trained_hosted_vm_case) -> None:
    model, prompt_trace, expected_continuation, fen, history = trained_hosted_vm_case
    hosted_vm = TransformerHostedVM(model=model)
    decoded = hosted_vm.decode_until_halt(prompt_trace, max_packets=len(expected_continuation) + 2)

    assert not hasattr(hosted_vm, "vm")
    assert not hasattr(hosted_vm, "host_vm")
    assert history.exact_match is True
    assert decoded == expected_continuation
    assert legal_trace_matches_oracle(fen, list(prompt_trace + decoded))


def test_transformer_hosted_vm_checkpoint_roundtrip_preserves_decode(tmp_path, trained_hosted_vm_case) -> None:
    model, prompt_trace, expected_continuation, _fen, history = trained_hosted_vm_case
    assert history.exact_match is True

    checkpoint_path = tmp_path / "hosted_vm.pt"
    save_transformer_checkpoint(model, checkpoint_path)
    loaded = load_transformer_checkpoint(checkpoint_path)
    decoded = TransformerHostedVM(model=loaded).decode_until_halt(prompt_trace, max_packets=len(expected_continuation) + 2)

    assert decoded == expected_continuation
