"""Offline compiler entry points for immutable VM artifact records."""

from __future__ import annotations

import math

from .artifact import Artifact, TensorRecord
from .addressing import build_ring_address_record
from .context import build_context_0
from .fp4 import encode_e2m1
from .relations import build_rule_relations
from .state_circuit import (
    build_initial_piece_state,
    build_square_decoder,
)

_CONTEXT_BLOCK_SIZE = 4096


def _binary_record(name, values) -> TensorRecord:
    flat = values.reshape(-1)
    return TensorRecord(
        name=name,
        shape=values.shape,
        packed=encode_e2m1(flat).tobytes(),
        scales=(1.0,) * math.ceil(flat.size / _CONTEXT_BLOCK_SIZE),
        block_size=_CONTEXT_BLOCK_SIZE,
    )


def build_context_0_record() -> TensorRecord:
    return _binary_record("context_0", build_context_0())


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
        records.append(_binary_record(name, relation))
    return tuple(records)


def build_state_circuit_records() -> tuple[TensorRecord, TensorRecord]:
    return (
        _binary_record("initial_piece_state", build_initial_piece_state()),
        _binary_record("square_decoder", build_square_decoder()),
    )


def build_rule_image_artifact() -> Artifact:
    bootstrap = build_bootstrap_artifact()
    return Artifact(
        tensors=(
            bootstrap.tensors
            + build_rule_relation_records()
            + build_state_circuit_records()
        ),
        operations=(),
    )
