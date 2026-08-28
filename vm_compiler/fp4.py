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

    magnitudes = numpy.abs(flat)
    indices = numpy.searchsorted(_POSITIVE_LATTICE, magnitudes)
    clipped = numpy.minimum(indices, len(_POSITIVE_LATTICE) - 1)
    valid = (indices < len(_POSITIVE_LATTICE)) & (
        _POSITIVE_LATTICE[clipped] == magnitudes
    )
    if not valid.all():
        value = flat[int(numpy.flatnonzero(~valid)[0])]
        raise ValueError(f"value is outside the exact E2M1 lattice: {value}")
    signs = (numpy.signbit(flat) & (magnitudes != 0.0)).astype(numpy.uint8) << 3
    nibbles = indices.astype(numpy.uint8) | signs
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
