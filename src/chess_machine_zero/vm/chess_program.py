"""Deterministic chess VM program skeletons.

Milestone 2 executes these programs through host Python functions while keeping
program structure explicit for the later transformer-hosted executor.
"""

from __future__ import annotations

from chess_machine_zero.vm.bytecode import Instruction, InstructionOp


def legal_generator_program() -> tuple[Instruction, ...]:
    return (
        Instruction(InstructionOp.LOOP_BEGIN, comment="iterate board squares"),
        Instruction(InstructionOp.BOARD_READ, comment="read piece at square"),
        Instruction(InstructionOp.GEN_PSEUDO_PAWN),
        Instruction(InstructionOp.GEN_PSEUDO_KNIGHT),
        Instruction(InstructionOp.GEN_PSEUDO_BISHOP),
        Instruction(InstructionOp.GEN_PSEUDO_ROOK),
        Instruction(InstructionOp.GEN_PSEUDO_QUEEN),
        Instruction(InstructionOp.GEN_PSEUDO_KING),
        Instruction(InstructionOp.GEN_CASTLING),
        Instruction(InstructionOp.GEN_EP),
        Instruction(InstructionOp.FILTER_LEGAL),
        Instruction(InstructionOp.EMIT, comment="emit CANDIDATE and LEGAL_SET"),
        Instruction(InstructionOp.LOOP_NEXT),
        Instruction(InstructionOp.HALT),
    )


def make_move_program() -> tuple[Instruction, ...]:
    return (
        Instruction(InstructionOp.BOARD_READ, comment="read source and target"),
        Instruction(InstructionOp.MAKE_MOVE),
        Instruction(InstructionOp.UPDATE_CASTLING),
        Instruction(InstructionOp.UPDATE_EP),
        Instruction(InstructionOp.UPDATE_CLOCKS),
        Instruction(InstructionOp.BOARD_WRITE, comment="emit changed square writes"),
        Instruction(InstructionOp.EMIT, comment="emit COMMIT_MOVE"),
        Instruction(InstructionOp.HALT),
    )


def terminal_check_program() -> tuple[Instruction, ...]:
    return (
        Instruction(InstructionOp.CHECK_TERMINAL),
        Instruction(InstructionOp.EMIT, comment="emit TERMINAL_SET"),
        Instruction(InstructionOp.HALT),
    )
