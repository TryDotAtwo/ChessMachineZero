"""Offline compiler entry points for immutable VM artifact records."""

from __future__ import annotations

import math

import numpy

from .artifact import Artifact, TensorRecord
from .addressing import build_ring_address_record
from .context import build_context_0
from .fp4 import encode_e2m1
from .graph import OpCode, Operation
from .protocol import PIECE_CHANNELS, VOCAB_SIZE
from .relations import build_rule_relations
from .state_circuit import (
    CASTLING_EVENTS,
    EMPTY,
    build_castling_derived_events,
    build_en_passant_event_patterns,
    build_initial_piece_state,
    build_square_decoder,
)

_CONTEXT_BLOCK_SIZE = 4096
_LATEST_EVENT_EPSILON = 1.0e-6


def _binary_record(name, values) -> TensorRecord:
    flat = values.reshape(-1)
    return TensorRecord(
        name=name,
        shape=values.shape,
        packed=encode_e2m1(flat).tobytes(),
        scales=(1.0,) * math.ceil(flat.size / _CONTEXT_BLOCK_SIZE),
        block_size=_CONTEXT_BLOCK_SIZE,
    )


def _scaled_record(name: str, values: numpy.ndarray) -> TensorRecord:
    flat = numpy.asarray(values, dtype=numpy.float32).reshape(-1)
    lattice = numpy.sign(flat).astype(numpy.float32)
    scales = numpy.abs(flat)
    scales[scales == 0.0] = 1.0
    return TensorRecord(
        name=name,
        shape=values.shape,
        packed=encode_e2m1(lattice).tobytes(),
        scales=tuple(float(scale) for scale in scales),
        block_size=1,
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


def build_state_circuit_records() -> tuple[TensorRecord, ...]:
    return (
        _binary_record("initial_piece_state", build_initial_piece_state()),
        _binary_record("square_decoder", build_square_decoder()),
        _binary_record("castling_derived_events", build_castling_derived_events()),
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


def build_position_reconstruction_artifact() -> Artifact:
    square_tokens = numpy.zeros((64, VOCAB_SIZE), dtype=numpy.float32)
    key_projection = numpy.zeros((VOCAB_SIZE, 2), dtype=numpy.float32)
    queries = numpy.zeros((64, 2), dtype=numpy.float32)
    for compact in range(64):
        file_index, rank_index = divmod(compact, 8)
        channel = (file_index + 1) * 10 + rank_index + 1
        x = numpy.float32(1.0 + compact / 64.0)
        square_tokens[compact, channel] = 1.0
        key_projection[channel] = (x, x * x)
        queries[compact] = (2.0 * x, -1.0)

    empty_events = numpy.zeros((400, VOCAB_SIZE), dtype=numpy.float32)
    empty_events[:, 95] = 1.0
    pad_row = numpy.zeros((1, VOCAB_SIZE), dtype=numpy.float32)
    pad_row[0, 0] = 1.0

    en_passant_patterns = build_en_passant_event_patterns()

    tensor_records: list[TensorRecord] = []

    def tensor(record: TensorRecord) -> int:
        tensor_records.append(record)
        return len(tensor_records) - 1

    def pattern_projection(name: str, patterns, slot: int) -> int:
        values = numpy.zeros((VOCAB_SIZE, len(patterns) + 1), dtype=numpy.float32)
        for index, pattern in enumerate(patterns):
            values[pattern[slot], index] = 1.0
        return tensor(_binary_record(name, values))

    def pattern_output(name: str, channels) -> int:
        values = numpy.zeros((len(channels) + 1, VOCAB_SIZE), dtype=numpy.float32)
        values[-1, 0] = 1.0
        for index, channel in enumerate(channels):
            values[index, channel] = 1.0
        return tensor(_binary_record(name, values))

    key_projection_index = tensor(_scaled_record("latest_square_key_projection", key_projection))
    query_index = tensor(_scaled_record("latest_square_queries", queries))
    initial_targets_index = tensor(_binary_record("initial_square_tokens", square_tokens))
    initial_board_index = tensor(_binary_record("initial_piece_state", build_initial_piece_state()))
    empty_events_index = tensor(_binary_record("empty_source_events", empty_events))
    pad_index = tensor(_binary_record("padding_row", pad_row))

    castle_patterns = tuple(tuple(pattern[:3]) for pattern in CASTLING_EVENTS)
    castle_projection_indices = tuple(
        pattern_projection(f"castle_match_slot_{slot}", castle_patterns, slot) for slot in range(3)
    )
    castle_bias = numpy.zeros((400, 5), dtype=numpy.float32)
    castle_bias[:, -1] = 2.5
    castle_bias_index = tensor(_scaled_record("castle_no_match_bias", castle_bias))
    castle_output_indices = (
        pattern_output("castle_rook_source_targets", [pattern[3] for pattern in CASTLING_EVENTS]),
        pattern_output("castle_rook_source_values", [pattern[4] for pattern in CASTLING_EVENTS]),
        pattern_output("castle_rook_destination_targets", [pattern[5] for pattern in CASTLING_EVENTS]),
        pattern_output("castle_rook_destination_values", [pattern[6] for pattern in CASTLING_EVENTS]),
    )

    ep_projection_indices = tuple(
        pattern_projection(f"en_passant_match_slot_{slot}", en_passant_patterns, slot)
        for slot in range(6)
    )
    ep_bias = numpy.zeros((400, len(en_passant_patterns) + 1), dtype=numpy.float32)
    ep_bias[:, -1] = 5.5
    ep_bias_index = tensor(_scaled_record("en_passant_no_match_bias", ep_bias))
    ep_target_index = pattern_output(
        "en_passant_clear_targets", [pattern[6] for pattern in en_passant_patterns]
    )
    ep_value_index = pattern_output(
        "en_passant_clear_values", [EMPTY for _ in en_passant_patterns]
    )

    operations: list[Operation] = []
    next_value = 1

    def emit(opcode: OpCode, inputs: tuple[int, ...], attributes: tuple[int, ...] = ()) -> int:
        nonlocal next_value
        output = next_value
        next_value += 1
        operations.append(Operation(opcode, inputs, (output,), attributes))
        return output

    source_rows = emit(OpCode.ROW_ROUTE, (0,), (3, 1203, 3))
    destination_rows = emit(OpCode.ROW_ROUTE, (0,), (4, 1204, 3))
    piece_rows = emit(OpCode.ROW_ROUTE, (0,), (5, 1205, 3))
    initial_targets = emit(OpCode.FROZEN_EXPAND, (0,), (initial_targets_index,))

    castle_logits = [
        emit(OpCode.TOKEN_PROJECT, (rows,), (projection,))
        for rows, projection in zip(
            (source_rows, destination_rows, piece_rows), castle_projection_indices
        )
    ]
    castle_sum = emit(OpCode.RESIDUAL_ADD, (castle_logits[0], castle_logits[1]))
    castle_sum = emit(OpCode.RESIDUAL_ADD, (castle_sum, castle_logits[2]))
    castle_sum = emit(OpCode.POSITION_ADD, (castle_sum,), (castle_bias_index,))
    castle_match = emit(OpCode.HARDMAX_STE, (castle_sum,), (500,))
    castle_source_targets = emit(
        OpCode.TOKEN_PROJECT, (castle_match,), (castle_output_indices[0],)
    )
    castle_source_values = emit(
        OpCode.TOKEN_PROJECT, (castle_match,), (castle_output_indices[1],)
    )
    castle_destination_targets = emit(
        OpCode.TOKEN_PROJECT, (castle_match,), (castle_output_indices[2],)
    )
    castle_destination_values = emit(
        OpCode.TOKEN_PROJECT, (castle_match,), (castle_output_indices[3],)
    )

    previous_source = emit(OpCode.ROW_ROUTE, (0,), (3, 1200, 3))
    previous_destination = emit(OpCode.ROW_ROUTE, (0,), (4, 1201, 3))
    previous_piece = emit(OpCode.ROW_ROUTE, (0,), (5, 1202, 3))
    pad = emit(OpCode.FROZEN_EXPAND, (0,), (pad_index,))
    previous_source = emit(OpCode.ROW_CONCAT, (pad, previous_source))
    previous_destination = emit(OpCode.ROW_CONCAT, (pad, previous_destination))
    previous_piece = emit(OpCode.ROW_CONCAT, (pad, previous_piece))
    ep_rows = (
        source_rows,
        destination_rows,
        piece_rows,
        previous_source,
        previous_destination,
        previous_piece,
    )
    ep_logits = [
        emit(OpCode.TOKEN_PROJECT, (rows,), (projection,))
        for rows, projection in zip(ep_rows, ep_projection_indices)
    ]
    ep_sum = ep_logits[0]
    for logits in ep_logits[1:]:
        ep_sum = emit(OpCode.RESIDUAL_ADD, (ep_sum, logits))
    ep_sum = emit(OpCode.POSITION_ADD, (ep_sum,), (ep_bias_index,))
    ep_match = emit(OpCode.HARDMAX_STE, (ep_sum,), (500,))
    ep_targets = emit(OpCode.TOKEN_PROJECT, (ep_match,), (ep_target_index,))
    ep_values = emit(OpCode.TOKEN_PROJECT, (ep_match,), (ep_value_index,))

    event_targets = emit(
        OpCode.ROW_CONCAT,
        (
            initial_targets,
            source_rows,
            destination_rows,
            castle_source_targets,
            castle_destination_targets,
            ep_targets,
        ),
    )
    event_times = numpy.concatenate(
        (numpy.zeros(64, dtype=numpy.float32),)
        + tuple(numpy.arange(1, 401, dtype=numpy.float32) for _ in range(5))
    )
    time_bias = numpy.zeros((2064, 2), dtype=numpy.float32)
    time_bias[:, 1] = -_LATEST_EVENT_EPSILON * event_times
    time_bias_index = tensor(_scaled_record("latest_event_time_bias", time_bias))
    keys = emit(OpCode.TOKEN_PROJECT, (event_targets,), (key_projection_index,))
    keys = emit(OpCode.POSITION_ADD, (keys,), (time_bias_index,))
    queries_value = emit(OpCode.FROZEN_EXPAND, (0,), (query_index,))
    initial_board = emit(OpCode.FROZEN_EXPAND, (0,), (initial_board_index,))
    empty_events_value = emit(OpCode.FROZEN_EXPAND, (0,), (empty_events_index,))
    event_values = emit(
        OpCode.ROW_CONCAT,
        (
            initial_board,
            empty_events_value,
            piece_rows,
            castle_source_values,
            castle_destination_values,
            ep_values,
        ),
    )
    emit(
        OpCode.HULL_ATTN_2D,
        (queries_value, keys, event_values),
        (8, 500, *range(2064)),
    )
    return Artifact(tensors=tuple(tensor_records), operations=tuple(operations))
