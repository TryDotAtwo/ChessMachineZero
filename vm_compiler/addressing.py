"""Frozen two-dimensional convex addressing for hard attention."""

from __future__ import annotations

import math

import numpy

from .artifact import TensorRecord
from .fp4 import encode_e2m1


def build_ring_address_keys(count: int) -> numpy.ndarray:
    if count < 3:
        raise ValueError("2D ring addressing requires at least three keys")
    angles = numpy.arange(count, dtype=numpy.float64) * (2.0 * math.pi / count)
    return numpy.stack((numpy.cos(angles), numpy.sin(angles)), axis=1).astype(numpy.float32)


def build_ring_address_record(name: str, count: int) -> TensorRecord:
    keys = build_ring_address_keys(count)
    flat = keys.reshape(-1)
    lattice = numpy.where(flat < 0.0, -1.0, 1.0).astype(numpy.float32)
    lattice[flat == 0.0] = 0.0
    scales = numpy.abs(flat)
    scales[scales == 0.0] = 1.0
    return TensorRecord(
        name=name,
        shape=keys.shape,
        packed=encode_e2m1(lattice).tobytes(),
        scales=tuple(float(scale) for scale in scales),
        block_size=1,
    )
