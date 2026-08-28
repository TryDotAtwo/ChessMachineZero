import numpy

from vm_compiler.artifact import Artifact
from vm_compiler.compiler import (
    build_bootstrap_artifact,
    build_context_0_record,
    build_rule_image_artifact,
    build_rule_relation_records,
    build_position_reconstruction_artifact,
    build_state_circuit_records,
)
from vm_compiler.context import build_context_0
from vm_compiler.fp4 import decode_e2m1
from vm_compiler.graph import OpCode


def test_context_0_record_decodes_exactly_without_board_side_channel():
    record = build_context_0_record()
    decoded = decode_e2m1(numpy.frombuffer(record.packed, dtype=numpy.uint8), 2045 * 128)
    scales = numpy.repeat(numpy.asarray(record.scales, dtype=numpy.float32), record.block_size)
    decoded = (decoded * scales[: decoded.size]).reshape(record.shape)

    assert record.name == "context_0"
    assert record.shape == (2045, 128)
    assert numpy.array_equal(decoded, build_context_0())


def test_bootstrap_artifact_round_trip_contains_context_and_2d_address_weights():
    artifact = build_bootstrap_artifact()
    restored = Artifact.from_bytes(artifact.to_bytes())

    assert restored == artifact
    assert [tensor.name for tensor in restored.tensors] == [
        "context_0",
        "vocabulary_addresses",
        "input_row_addresses",
    ]
    assert restored.tensors[1].shape == (128, 2)
    assert restored.tensors[2].shape == (2048, 2)
    assert restored.operations == ()


def test_rule_relations_pack_as_exact_immutable_fp4_records():
    records = build_rule_relation_records()

    assert [record.name for record in records] == [
        "king_relation",
        "knight_relation",
        "rook_ray_relation",
        "bishop_ray_relation",
        "pawn_step_relation",
        "pawn_double_relation",
        "pawn_attack_relation",
        "between_relation",
    ]
    assert records[-1].shape == (64, 64, 64)
    for record in records:
        decoded = decode_e2m1(
            numpy.frombuffer(record.packed, dtype=numpy.uint8), numpy.prod(record.shape)
        )
        assert set(numpy.unique(decoded)).issubset({0.0, 1.0})


def test_rule_image_serializes_bootstrap_and_reusable_relations_together():
    image = build_rule_image_artifact()
    restored = Artifact.from_bytes(image.to_bytes())

    assert restored == image
    assert len(restored.tensors) == 14
    assert restored.tensors[0].name == "context_0"
    assert restored.tensors[-1].name == "castling_derived_events"
    assert restored.operations == ()


def test_state_circuit_constants_pack_exactly_as_fp4_records():
    initial_state, decoder, castling = build_state_circuit_records()

    assert initial_state.name == "initial_piece_state"
    assert initial_state.shape == (64, 128)
    assert decoder.name == "square_decoder"
    assert decoder.shape == (128, 64)
    assert castling.name == "castling_derived_events"
    assert castling.shape == (4, 7, 128)
    for record in (initial_state, decoder, castling):
        decoded = decode_e2m1(
            numpy.frombuffer(record.packed, dtype=numpy.uint8), numpy.prod(record.shape)
        )
        assert set(numpy.unique(decoded)) == {0.0, 1.0}


def test_position_reconstruction_artifact_is_an_executable_latest_event_attention_graph():
    artifact = build_position_reconstruction_artifact()
    restored = Artifact.from_bytes(artifact.to_bytes())

    assert restored == artifact
    assert set(operation.opcode for operation in artifact.operations).issubset(
        {
            OpCode.ROW_ROUTE,
            OpCode.FROZEN_EXPAND,
            OpCode.ROW_CONCAT,
            OpCode.TOKEN_PROJECT,
            OpCode.POSITION_ADD,
            OpCode.RESIDUAL_ADD,
            OpCode.HARDMAX_STE,
            OpCode.HULL_ATTN_2D,
        }
    )
    assert sum(op.opcode == OpCode.HARDMAX_STE for op in artifact.operations) == 2
    assert artifact.operations[-1].outputs == (len(artifact.operations),)
    assert artifact.operations[-1].attributes[:2] == (8, 500)
    assert len(artifact.operations[-1].attributes[2:]) == 2064
