import numpy

from vm_compiler.artifact import Artifact
from vm_compiler.compiler import build_bootstrap_artifact, build_context_0_record
from vm_compiler.context import build_context_0
from vm_compiler.fp4 import decode_e2m1


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
