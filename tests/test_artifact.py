import hashlib

import numpy
import pytest

from vm_compiler.artifact import Artifact, ArtifactError, TensorRecord
from vm_compiler.fp4 import decode_e2m1, encode_e2m1
from vm_compiler.graph import OpCode, Operation


def test_e2m1_packs_two_exact_lattice_values_per_byte():
    values = numpy.array([0.0, 0.5, 1.0, 1.5, -2.0, -3.0, 4.0, -6.0], dtype=numpy.float32)

    packed = encode_e2m1(values)

    assert packed.tolist() == [0x10, 0x32, 0xDC, 0xF6]
    assert decode_e2m1(packed, len(values)).tolist() == values.tolist()


def test_artifact_round_trip_preserves_explicit_graph_and_tensor_metadata():
    artifact = Artifact(
        tensors=(
            TensorRecord(
                name="token_projection",
                shape=(2, 2),
                packed=bytes([0x10, 0x32]),
                scales=(1.0,),
                block_size=4,
            ),
        ),
        operations=(
            Operation(OpCode.TOKEN_PROJECT, inputs=(0,), outputs=(1,), attributes=(4,)),
            Operation(OpCode.ROW_ROUTE, inputs=(1,), outputs=(2,), attributes=(2045, 2048)),
        ),
    )

    restored = Artifact.from_bytes(artifact.to_bytes())

    assert restored == artifact


def test_tensor_payload_and_scale_counts_must_match_shape():
    tensor = TensorRecord(
        name="broken",
        shape=(4, 4),
        packed=b"\x00",
        scales=(1.0,),
        block_size=4,
    )

    with pytest.raises(ArtifactError, match="packed FP4 length"):
        Artifact(tensors=(tensor,), operations=()).to_bytes()


def test_unknown_opcode_is_rejected_without_substitution():
    artifact = Artifact(
        tensors=(),
        operations=(Operation(OpCode.HARDMAX_STE, inputs=(0,), outputs=(1,)),),
    )
    encoded = bytearray(artifact.to_bytes())
    encoded[56:58] = b"\xff\x00"
    encoded[24:56] = hashlib.sha256(encoded[56:]).digest()

    with pytest.raises(ArtifactError, match="unknown opcode 255"):
        Artifact.from_bytes(bytes(encoded))
