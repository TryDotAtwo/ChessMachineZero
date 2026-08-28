"""Explicit generic operator IR stored in every frozen VM artifact."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class OpCode(IntEnum):
    TOKEN_PROJECT = 1
    POSITION_ADD = 2
    HULL_ATTN_2D = 3
    RESIDUAL_ADD = 4
    GATED_FFN = 5
    OUTPUT_PROJECT = 6
    ROW_ROUTE = 7
    HARDMAX_STE = 8


@dataclass(frozen=True)
class Operation:
    opcode: OpCode
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    attributes: tuple[int, ...] = ()
