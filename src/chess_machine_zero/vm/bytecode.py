"""Minimal chess bytecode declarations for deterministic VM programs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class InstructionOp(IntEnum):
    BOARD_READ = 1
    BOARD_WRITE = 2
    REG_READ = 3
    REG_WRITE = 4
    LOOP_BEGIN = 5
    LOOP_NEXT = 6
    BRANCH_IF = 7
    EMIT = 8
    HALT = 9
    GEN_PSEUDO_PAWN = 100
    GEN_PSEUDO_KNIGHT = 101
    GEN_PSEUDO_BISHOP = 102
    GEN_PSEUDO_ROOK = 103
    GEN_PSEUDO_QUEEN = 104
    GEN_PSEUDO_KING = 105
    GEN_CASTLING = 106
    GEN_EP = 107
    FILTER_LEGAL = 108
    MAKE_MOVE = 109
    UPDATE_CASTLING = 110
    UPDATE_EP = 111
    UPDATE_CLOCKS = 112
    CHECK_TERMINAL = 113
    BEGIN_CANDIDATE_SCORING = 200
    CALL_MOVE_RANKER = 201
    WRITE_SCORE = 202
    SAMPLE_BY_SCORE = 203
    COMMIT_SELECTED_MOVE = 204


@dataclass(frozen=True, slots=True)
class Instruction:
    op: InstructionOp
    a0: int = 0
    a1: int = 0
    a2: int = 0
    comment: str = ""
