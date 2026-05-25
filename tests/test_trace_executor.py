from __future__ import annotations

from chess_machine_zero.chess.board_io import STARTING_FEN
from chess_machine_zero.model.trace_executor import CMZTraceExecutor
from chess_machine_zero.trace.verifier import legal_trace_matches_oracle
from chess_machine_zero.vm.interpreter import ChessMachineVM


def test_trace_executor_generated_short_legal_trace_passes_verifier() -> None:
    executor = CMZTraceExecutor(vm=ChessMachineVM(seed=1234))
    trace = executor.execute_legal_generator(STARTING_FEN)

    assert legal_trace_matches_oracle(STARTING_FEN, trace)
