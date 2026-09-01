"""Generic matrix-layout/GEMM wire operations, not chess-rule tests."""

import pytest
import torch

from vm_compiler.artifact import Artifact, TensorRecord
from vm_compiler.graph import OpCode, Operation
from vm_compiler.reference_executor import execute_artifact_reference


def matrix_fixture():
    # Literal FP4 selector: input columns 0 and 1, no computed oracle weights.
    packed = bytearray(128)
    packed[0] = 2
    packed[1] = 32
    weights = TensorRecord("two_columns", (128, 2), bytes(packed), (1.0,), 256)
    return Artifact((weights,), (
        Operation(OpCode.ROW_ROUTE, (0,), (1,), (0, 2)),
        Operation(OpCode.TOKEN_PROJECT, (1,), (2,), (0,)),
        Operation(OpCode(11), (2,), (3,)),
        Operation(OpCode(12), (3,), (4,), (1, 4)),
        Operation(OpCode(12), (4,), (5,), (4, 1)),
        Operation(OpCode(11), (5,), (6,)),
        Operation(OpCode(13), (5, 6), (7,)),
    ))


def test_layout_matmul_preserve_coordinates_batches_and_exact_gradients():
    source = torch.zeros(2, 2048, 128)
    source[0, :2, :2] = torch.tensor([[1., 2.], [3., 4.]])
    source[1, :2, :2] = torch.tensor([[2., 4.], [6., 8.]])
    source.requires_grad_()
    artifact = Artifact.from_bytes(matrix_fixture().to_bytes())
    result = execute_artifact_reference(artifact, source)
    expected = torch.tensor([
        [[1., 3., 2., 4.], [3., 9., 6., 12.], [2., 6., 4., 8.], [4., 12., 8., 16.]],
        [[4., 12., 8., 16.], [12., 36., 24., 48.], [8., 24., 16., 32.], [16., 48., 32., 64.]],
    ])
    assert torch.equal(result, expected)
    result.sum().backward()
    expected_grad = torch.zeros_like(source)
    expected_grad[0, :2, :2] = 20.
    expected_grad[1, :2, :2] = 40.
    assert torch.equal(source.grad, expected_grad)


def test_grouped_matmul_uses_independent_static_row_groups_and_gradients():
    packed_left = bytearray(128)
    packed_left[0] = 2
    packed_left[1] = 32
    packed_right = bytearray(64)
    packed_right[0] = 2
    artifact = Artifact((
        TensorRecord("two_columns", (128, 2), bytes(packed_left), (1.0,), 256),
        TensorRecord("one_column", (128, 1), bytes(packed_right), (1.0,), 128),
    ), (
        Operation(OpCode.ROW_ROUTE, (0,), (1,), (0, 4)),
        Operation(OpCode.TOKEN_PROJECT, (1,), (2,), (0,)),
        Operation(OpCode.ROW_ROUTE, (0,), (3,), (4, 8)),
        Operation(OpCode.TOKEN_PROJECT, (3,), (4,), (1,)),
        Operation(OpCode.GROUPED_MATRIX_MATMUL, (2, 4), (5,), (2,)),
    ))
    source = torch.zeros(1, 2048, 128)
    source[0, :4, :2] = torch.tensor([[1., 2.], [3., 4.], [7., 8.], [9., 10.]])
    source[0, 4:8, 0] = torch.tensor([5., 6., 11., 12.])
    source.requires_grad_()

    result = execute_artifact_reference(artifact, source)

    assert torch.equal(result, torch.tensor([[[17.], [39.], [173.], [219.]]]))
    result.sum().backward()
    assert torch.equal(source.grad[0, :4, :2], torch.tensor([
        [5., 6.], [5., 6.], [11., 12.], [11., 12.],
    ]))
    assert torch.equal(source.grad[0, 4:8, 0], torch.tensor([4., 6., 16., 18.]))


@pytest.mark.parametrize("opcode,inputs,attributes,reason", [
    (11, (0,), (1,), "schema"),
    (12, (0,), (0, 128), "reshape"),
    (12, (0,), (2, 128), "reshape"),
    (12, (0,), (2048,), "schema"),
    (13, (0, 0), (), "matmul"),
    (13, (0,), (), "schema"),
    (14, (0, 0), (), "schema"),
    (14, (0,), (2,), "schema"),
    (14, (0, 0), (0,), "group"),
    (14, (0, 0), (3,), "group"),
])
def test_invalid_matrix_metadata_is_rejected(opcode, inputs, attributes, reason):
    artifact = Artifact((), (Operation(OpCode(opcode), inputs, (1,), attributes),))
    with pytest.raises(ValueError, match=reason):
        execute_artifact_reference(artifact, torch.zeros(1, 2048, 128))
