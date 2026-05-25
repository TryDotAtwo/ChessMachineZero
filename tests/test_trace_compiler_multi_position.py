from __future__ import annotations

from chess_machine_zero.trace.compiler import compile_legal_trace_examples, compile_next_packet_batch
from chess_machine_zero.vm.interpreter import ChessMachineVM
from chess_machine_zero.vm.trace_packet import TraceOp


FENS = (
    "8/8/8/8/8/8/8/4K2k w - - 0 1",
    "8/8/8/8/8/8/8/2K4k w - - 0 1",
    "8/8/8/8/8/8/8/K6k w - - 0 1",
)


def test_compile_legal_trace_examples_have_prompt_continuation_and_halt() -> None:
    examples = compile_legal_trace_examples(FENS, ChessMachineVM(seed=20260524), include_halt=True)

    assert tuple(example.fen for example in examples) == FENS
    for example in examples:
        assert len(example.prompt_trace) == 68
        assert example.full_trace[: example.prompt_length] == example.prompt_trace
        assert example.continuation_trace == example.full_trace[example.prompt_length :]
        assert example.continuation_trace[-1].op is TraceOp.PROGRAM_HALT
        assert len(example.continuation_trace) > 0


def test_compile_next_packet_batch_masks_only_continuation_targets() -> None:
    examples = compile_legal_trace_examples(FENS, ChessMachineVM(seed=20260524), include_halt=True)
    batch = compile_next_packet_batch(examples, minimum_vocab_size=16)

    assert batch.inputs.shape == batch.targets.shape
    assert batch.inputs.shape[0] == len(FENS)
    assert batch.loss_mask.shape == batch.inputs.shape[:2]
    assert batch.field_vocab_sizes[0] > int(TraceOp.PROGRAM_HALT)
    for row, example in enumerate(examples):
        selected_positions = [idx for idx, selected in enumerate(batch.loss_mask[row].tolist()) if selected]
        assert selected_positions == list(range(example.prompt_length - 1, len(example.full_trace) - 1))
        assert batch.lengths[row] == len(example.full_trace) - 1
