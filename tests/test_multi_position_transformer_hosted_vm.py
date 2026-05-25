from __future__ import annotations

import torch

from chess_machine_zero.model.hosted_vm import TransformerHostedVM
from chess_machine_zero.model.machine_transformer import CMZMachineTransformer
from chess_machine_zero.train.trainer import train_next_packet_overfit
from chess_machine_zero.trace.compiler import compile_legal_trace_examples, compile_next_packet_batch
from chess_machine_zero.trace.verifier import legal_trace_matches_oracle
from chess_machine_zero.vm.interpreter import ChessMachineVM


FENS = (
    "8/8/8/8/8/8/8/4K2k w - - 0 1",
    "8/8/8/8/8/8/8/2K4k w - - 0 1",
)


def test_one_transformer_hosted_vm_decodes_multiple_position_traces() -> None:
    torch.manual_seed(20260524)
    examples = compile_legal_trace_examples(FENS, ChessMachineVM(seed=20260524), include_halt=True)
    batch = compile_next_packet_batch(examples, minimum_vocab_size=16)
    model = CMZMachineTransformer(
        field_vocab_sizes=batch.field_vocab_sizes,
        d_model=48,
        n_heads=24,
        n_layers=2,
        d_ff=144,
        max_seq_len=batch.inputs.shape[1] + 2,
        dropout=0.0,
    )

    history = train_next_packet_overfit(
        model,
        batch.inputs,
        batch.targets,
        steps=900,
        lr=0.015,
        seed=20260524,
        loss_mask=batch.loss_mask,
    )
    hosted_vm = TransformerHostedVM(model=model)

    assert history.exact_match is True
    assert not hasattr(hosted_vm, "vm")
    for example in examples:
        decoded = hosted_vm.decode_until_halt(example.prompt_trace, max_packets=len(example.continuation_trace) + 2)
        assert decoded == example.continuation_trace
        assert legal_trace_matches_oracle(example.fen, list(example.prompt_trace + decoded))
