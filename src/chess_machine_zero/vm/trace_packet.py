"""Fixed-width append-only trace packet codec."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class TraceOp(IntEnum):
    NOP = 0
    WRITE_SQ = 1
    READ_SQ = 2
    WRITE_REG = 3
    PUSH_STACK = 4
    POP_STACK = 5
    WRITE_CASTLE = 6
    WRITE_EP = 7
    WRITE_CLOCK = 8
    WRITE_HASH = 9
    CANDIDATE = 10
    LEGAL_SET = 11
    SCORE_SET = 12
    SAMPLE_SET = 13
    COMMIT_MOVE = 14
    TERMINAL_SET = 15
    HALT_GAME = 16
    PROGRAM_HALT = 17


class TraceTag(IntEnum):
    NONE = 0
    BOARD = 1
    STATE = 2
    MOVE = 3
    LEGAL = 4
    TERMINAL = 5


class RegId(IntEnum):
    SIDE_TO_MOVE = 1


@dataclass(frozen=True, slots=True)
class TracePacket:
    """Fixed-width trace packet: `[OP, A0, A1, A2, A3, TAG, COMMIT]`."""

    op: TraceOp
    a0: int = 0
    a1: int = 0
    a2: int = 0
    a3: int = 0
    tag: TraceTag = TraceTag.NONE
    commit: int = 0

    WIDTH = 7
    MAX_FIELD = 2**31 - 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "op", TraceOp(int(self.op)))
        object.__setattr__(self, "tag", TraceTag(int(self.tag)))
        for name in ("a0", "a1", "a2", "a3", "commit"):
            value = int(getattr(self, name))
            if value < 0 or value > self.MAX_FIELD:
                raise ValueError(f"TracePacket field {name} out of range: {value}")
            object.__setattr__(self, name, value)

    def to_tuple(self) -> tuple[int, int, int, int, int, int, int]:
        return (int(self.op), self.a0, self.a1, self.a2, self.a3, int(self.tag), self.commit)

    def to_tokens(self) -> tuple[int, int, int, int, int, int, int]:
        return self.to_tuple()

    @classmethod
    def from_tuple(cls, values: tuple[int, ...] | list[int]) -> "TracePacket":
        if len(values) != cls.WIDTH:
            raise ValueError(f"TracePacket requires {cls.WIDTH} fields, got {len(values)}")
        return cls(
            TraceOp(int(values[0])),
            int(values[1]),
            int(values[2]),
            int(values[3]),
            int(values[4]),
            TraceTag(int(values[5])),
            int(values[6]),
        )

    @classmethod
    def from_tokens(cls, values: tuple[int, ...] | list[int]) -> "TracePacket":
        return cls.from_tuple(values)
