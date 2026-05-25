"""Compact move packet codec.

Square indexing follows the common chess-programming convention:
`a1 == 0`, `b1 == 1`, ..., `h8 == 63`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag


FILES = "abcdefgh"
RANKS = "12345678"
SQUARE_COUNT = 64
PROMO_BASE = 5


class Promo(IntEnum):
    NONE = 0
    KNIGHT = 1
    BISHOP = 2
    ROOK = 3
    QUEEN = 4


class MoveFlag(IntFlag):
    QUIET = 0
    CAPTURE = 1
    EP = 2
    CASTLE = 4
    PROMOTION = 8
    CHECK = 16
    MATE_CANDIDATE = 32


PROMO_TO_UCI = {
    Promo.NONE: "",
    Promo.KNIGHT: "n",
    Promo.BISHOP: "b",
    Promo.ROOK: "r",
    Promo.QUEEN: "q",
}
UCI_TO_PROMO = {value: key for key, value in PROMO_TO_UCI.items() if value}


def square_name(square: int) -> str:
    if not 0 <= square < SQUARE_COUNT:
        raise ValueError(f"square out of range: {square}")
    return FILES[square % 8] + RANKS[square // 8]


def square_index(name: str) -> int:
    if len(name) != 2 or name[0] not in FILES or name[1] not in RANKS:
        raise ValueError(f"invalid square name: {name!r}")
    return FILES.index(name[0]) + 8 * RANKS.index(name[1])


@dataclass(frozen=True, slots=True)
class MovePacket:
    """Fixed-width move packet: `[FROM_SQ, TO_SQ, PROMO, FLAGS]`."""

    from_sq: int
    to_sq: int
    promo: Promo = Promo.NONE
    flags: MoveFlag = MoveFlag.QUIET

    WIDTH = 4

    def __post_init__(self) -> None:
        if not 0 <= int(self.from_sq) < SQUARE_COUNT:
            raise ValueError(f"from_sq out of range: {self.from_sq}")
        if not 0 <= int(self.to_sq) < SQUARE_COUNT:
            raise ValueError(f"to_sq out of range: {self.to_sq}")
        object.__setattr__(self, "promo", Promo(int(self.promo)))
        object.__setattr__(self, "flags", MoveFlag(int(self.flags)))
        if self.promo is not Promo.NONE and not (self.flags & MoveFlag.PROMOTION):
            object.__setattr__(self, "flags", self.flags | MoveFlag.PROMOTION)

    @property
    def move_id(self) -> int:
        return self.from_sq * SQUARE_COUNT * PROMO_BASE + self.to_sq * PROMO_BASE + int(self.promo)

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.from_sq, self.to_sq, int(self.promo), int(self.flags))

    @classmethod
    def from_tuple(cls, values: tuple[int, int, int, int] | list[int]) -> "MovePacket":
        if len(values) != cls.WIDTH:
            raise ValueError(f"MovePacket requires {cls.WIDTH} fields, got {len(values)}")
        return cls(int(values[0]), int(values[1]), Promo(int(values[2])), MoveFlag(int(values[3])))

    @classmethod
    def from_move_id(cls, move_id: int, flags: MoveFlag = MoveFlag.QUIET) -> "MovePacket":
        if move_id < 0 or move_id >= SQUARE_COUNT * SQUARE_COUNT * PROMO_BASE:
            raise ValueError(f"move_id out of range: {move_id}")
        from_sq, rem = divmod(move_id, SQUARE_COUNT * PROMO_BASE)
        to_sq, promo = divmod(rem, PROMO_BASE)
        return cls(from_sq, to_sq, Promo(promo), flags)

    def to_uci(self) -> str:
        return square_name(self.from_sq) + square_name(self.to_sq) + PROMO_TO_UCI[self.promo]

    @classmethod
    def from_uci(cls, uci: str, flags: MoveFlag = MoveFlag.QUIET) -> "MovePacket":
        if len(uci) not in (4, 5):
            raise ValueError(f"invalid UCI move length: {uci!r}")
        from_sq = square_index(uci[0:2])
        to_sq = square_index(uci[2:4])
        promo = Promo.NONE
        if len(uci) == 5:
            promo = UCI_TO_PROMO.get(uci[4])
            if promo is None:
                raise ValueError(f"invalid UCI promotion: {uci!r}")
            flags |= MoveFlag.PROMOTION
        return cls(from_sq, to_sq, promo, flags)

    def sort_key(self) -> tuple[int, int, int, int]:
        return (self.from_sq, self.to_sq, int(self.promo), int(self.flags))
