"""Offline compiler entry points for immutable VM artifact records."""

from __future__ import annotations

import math

from .artifact import Artifact, TensorRecord
from .context import build_context_0
from .fp4 import encode_e2m1

_CONTEXT_BLOCK_SIZE = 4096


def build_context_0_record() -> TensorRecord:
    context = build_context_0()
    flat = context.reshape(-1)
    packed = encode_e2m1(flat).tobytes()
    scale_count = math.ceil(flat.size / _CONTEXT_BLOCK_SIZE)
    return TensorRecord(
        name="context_0",
        shape=context.shape,
        packed=packed,
        scales=(1.0,) * scale_count,
        block_size=_CONTEXT_BLOCK_SIZE,
    )


def build_bootstrap_artifact() -> Artifact:
    return Artifact(tensors=(build_context_0_record(),), operations=())
