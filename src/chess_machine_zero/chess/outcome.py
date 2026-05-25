"""Terminal result encodings for trace records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ResultCode(IntEnum):
    ONGOING = 0
    WHITE_WIN = 1
    BLACK_WIN = 2
    DRAW = 3


class TerminalReason(IntEnum):
    NONE = 0
    CHECKMATE = 1
    STALEMATE = 2
    FIFTY_MOVE = 3
    THREEFOLD = 4
    INSUFFICIENT_MATERIAL = 5
    ADJUDICATION_CAP = 6


@dataclass(frozen=True, slots=True)
class TerminalStatus:
    result: ResultCode
    reason: TerminalReason
    ply: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.result is not ResultCode.ONGOING
