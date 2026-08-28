"""Compile reusable chess geometry as immutable binary tensors."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy


def square_index(channel: int) -> int:
    file_index, rank_index = divmod(channel, 10)
    if not 1 <= file_index <= 8 or not 1 <= rank_index <= 8:
        raise ValueError(f"invalid square channel: {channel}")
    return (file_index - 1) * 8 + rank_index - 1


def _channel(file_index: int, rank_index: int) -> int:
    return file_index * 10 + rank_index


def _inside(file_index: int, rank_index: int) -> bool:
    return 1 <= file_index <= 8 and 1 <= rank_index <= 8


@dataclass(frozen=True)
class RuleRelations:
    king: numpy.ndarray
    knight: numpy.ndarray
    rook_ray: numpy.ndarray
    bishop_ray: numpy.ndarray
    pawn_step: numpy.ndarray
    pawn_double: numpy.ndarray
    pawn_attack: numpy.ndarray
    between: numpy.ndarray

    def __iter__(self):
        return iter(getattr(self, field.name) for field in fields(self))


def build_rule_relations() -> RuleRelations:
    king = numpy.zeros((64, 64), dtype=numpy.float32)
    knight = numpy.zeros_like(king)
    rook_ray = numpy.zeros_like(king)
    bishop_ray = numpy.zeros_like(king)
    pawn_step = numpy.zeros((2, 64, 64), dtype=numpy.float32)
    pawn_double = numpy.zeros_like(pawn_step)
    pawn_attack = numpy.zeros_like(pawn_step)
    between = numpy.zeros((64, 64, 64), dtype=numpy.float32)

    for source_file in range(1, 9):
        for source_rank in range(1, 9):
            source = square_index(_channel(source_file, source_rank))
            for destination_file in range(1, 9):
                for destination_rank in range(1, 9):
                    destination = square_index(_channel(destination_file, destination_rank))
                    file_delta = destination_file - source_file
                    rank_delta = destination_rank - source_rank
                    absolute = (abs(file_delta), abs(rank_delta))
                    king[source, destination] = float(max(absolute) == 1)
                    knight[source, destination] = float(sorted(absolute) == [1, 2])
                    rook_ray[source, destination] = float(
                        destination != source and (file_delta == 0 or rank_delta == 0)
                    )
                    bishop_ray[source, destination] = float(
                        destination != source and abs(file_delta) == abs(rank_delta)
                    )

                    aligned = bool(rook_ray[source, destination] or bishop_ray[source, destination])
                    if aligned:
                        file_step = (file_delta > 0) - (file_delta < 0)
                        rank_step = (rank_delta > 0) - (rank_delta < 0)
                        file_cursor = source_file + file_step
                        rank_cursor = source_rank + rank_step
                        while (file_cursor, rank_cursor) != (destination_file, destination_rank):
                            between[source, destination, square_index(_channel(file_cursor, rank_cursor))] = 1.0
                            file_cursor += file_step
                            rank_cursor += rank_step

            for color, direction, starting_rank in ((0, 1, 2), (1, -1, 7)):
                forward_rank = source_rank + direction
                if _inside(source_file, forward_rank):
                    pawn_step[color, source, square_index(_channel(source_file, forward_rank))] = 1.0
                    for file_step in (-1, 1):
                        attack_file = source_file + file_step
                        if _inside(attack_file, forward_rank):
                            pawn_attack[color, source, square_index(_channel(attack_file, forward_rank))] = 1.0
                if source_rank == starting_rank:
                    pawn_double[
                        color,
                        source,
                        square_index(_channel(source_file, source_rank + 2 * direction)),
                    ] = 1.0

    return RuleRelations(
        king=king,
        knight=knight,
        rook_ray=rook_ray,
        bishop_ray=bishop_ray,
        pawn_step=pawn_step,
        pawn_double=pawn_double,
        pawn_attack=pawn_attack,
        between=between,
    )
