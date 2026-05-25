"""Decision-program skeletons for trace-based move selection."""

from __future__ import annotations

from chess_machine_zero.vm.bytecode import Instruction, InstructionOp


def select_move_program() -> tuple[Instruction, ...]:
    return (
        Instruction(InstructionOp.BEGIN_CANDIDATE_SCORING),
        Instruction(InstructionOp.CALL_MOVE_RANKER),
        Instruction(InstructionOp.WRITE_SCORE),
        Instruction(InstructionOp.SAMPLE_BY_SCORE),
        Instruction(InstructionOp.COMMIT_SELECTED_MOVE),
        Instruction(InstructionOp.HALT),
    )


def trace_negamax_program() -> tuple[Instruction, ...]:
    return (
        Instruction(InstructionOp.CHECK_TERMINAL),
        Instruction(InstructionOp.BRANCH_IF, comment="terminal returns game result"),
        Instruction(InstructionOp.BRANCH_IF, comment="depth zero calls outcome baseline"),
        Instruction(InstructionOp.LOOP_BEGIN, comment="iterate legal VM candidates"),
        Instruction(InstructionOp.MAKE_MOVE),
        Instruction(InstructionOp.CALL_MOVE_RANKER, comment="depth-zero baseline leaf for child trace"),
        Instruction(InstructionOp.WRITE_SCORE),
        Instruction(InstructionOp.LOOP_NEXT),
        Instruction(InstructionOp.HALT),
    )
