"""Artifact hardmax shares the standalone reference's validation and gradients."""

import pytest
import torch

from vm_compiler.artifact import Artifact
from vm_compiler.graph import OpCode, Operation
from vm_compiler.reference_executor import execute_artifact_reference


def _hardmax_graph(temperature_milli, masked=True):
    operations = [Operation(OpCode.ROW_ROUTE, (0,), (1,), (0, 1))]
    if masked:
        operations.append(Operation(OpCode.ROW_ROUTE, (0,), (2,), (1, 2)))
    operations.append(Operation(OpCode.HARDMAX_STE, (1, 2) if masked else (1,),
                                (3,), (temperature_milli,)))
    return Artifact(tensors=(), operations=tuple(operations))


def test_artifact_hardmax_rejects_an_empty_mask_row():
    source = torch.tensor([[[1., 2., 3.], [0., 0., 0.]]])
    with pytest.raises(ValueError):
        execute_artifact_reference(_hardmax_graph(1000), source)


@pytest.mark.parametrize("temperature", [0, -1000, float("nan"), float("inf"), -float("inf")])
def test_artifact_hardmax_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        execute_artifact_reference(_hardmax_graph(temperature, masked=False), torch.ones((1, 1, 3)))


def test_artifact_hardmax_exact_masked_ties_and_softmax_gradient():
    source = torch.tensor([[[0., 0., 7.], [1., 1., 0.]]], requires_grad=True)
    actual = execute_artifact_reference(_hardmax_graph(500), source)
    actual.backward(torch.tensor([[[2., -1., 99.]]]))
    assert torch.equal(actual, torch.tensor([[[1., 0., 0.]]]))
    assert torch.equal(source.grad, torch.tensor([[[1.5, -1.5, 0.], [0., 0., 0.]]]))
