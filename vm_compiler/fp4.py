"""Deterministic IEEE-style E2M1 nibble packing for frozen weights."""

from __future__ import annotations

import numpy

_POSITIVE_LATTICE = numpy.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=numpy.float32
)


def encode_e2m1(values: numpy.ndarray) -> numpy.ndarray:
    flat = numpy.asarray(values, dtype=numpy.float32).reshape(-1)
    if flat.size % 2:
        raise ValueError("E2M1 packing requires an even value count")
    if not numpy.isfinite(flat).all():
        raise ValueError("E2M1 values must be finite")

    nibbles = numpy.empty(flat.size, dtype=numpy.uint8)
    for index, value in enumerate(flat):
        magnitude = abs(float(value))
        matches = numpy.flatnonzero(_POSITIVE_LATTICE == magnitude)
        if matches.size != 1:
            raise ValueError(f"value is outside the exact E2M1 lattice: {value}")
        sign = 0x8 if numpy.signbit(value) and magnitude != 0.0 else 0
        nibbles[index] = int(matches[0]) | sign
    return nibbles[0::2] | (nibbles[1::2] << 4)


def decode_e2m1(packed: numpy.ndarray, value_count: int) -> numpy.ndarray:
    raw = numpy.asarray(packed, dtype=numpy.uint8).reshape(-1)
    if value_count < 0 or value_count > raw.size * 2:
        raise ValueError("E2M1 value count exceeds packed capacity")
    nibbles = numpy.empty(raw.size * 2, dtype=numpy.uint8)
    nibbles[0::2] = raw & 0xF
    nibbles[1::2] = raw >> 4
    selected = nibbles[:value_count]
    magnitudes = _POSITIVE_LATTICE[selected & 0x7]
    signs = numpy.where(selected & 0x8, -1.0, 1.0).astype(numpy.float32)
    return magnitudes * signs
