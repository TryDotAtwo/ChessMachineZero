"""Offline compiler entry points for immutable VM artifact records."""

from __future__ import annotations

import math

from .artifact import Artifact, TensorRecord
from .addressing import build_ring_address_record
from .context import build_context_0
from .fp4 import encode_e2m1
from .relations import build_rule_relations

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
    return Artifact(
        tensors=(
            build_context_0_record(),
            build_ring_address_record("vocabulary_addresses", 128),
            build_ring_address_record("input_row_addresses", 2048),
        ),
        operations=(),
    )


def build_rule_relation_records() -> tuple[TensorRecord, ...]:
    names = (
        "king_relation",
        "knight_relation",
        "rook_ray_relation",
        "bishop_ray_relation",
        "pawn_step_relation",
        "pawn_double_relation",
        "pawn_attack_relation",
        "between_relation",
    )
    records = []
    for name, relation in zip(names, build_rule_relations()):
        flat = relation.reshape(-1)
        records.append(
            TensorRecord(
                name=name,
                shape=relation.shape,
                packed=encode_e2m1(flat).tobytes(),
                scales=(1.0,) * math.ceil(flat.size / _CONTEXT_BLOCK_SIZE),
                block_size=_CONTEXT_BLOCK_SIZE,
            )
        )
    return tuple(records)


def build_rule_image_artifact() -> Artifact:
    bootstrap = build_bootstrap_artifact()
    return Artifact(
        tensors=bootstrap.tensors + build_rule_relation_records(),
        operations=(),
    )
