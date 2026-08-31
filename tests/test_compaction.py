"""Stable compaction via generic matrices, with explicit capacity evidence."""

import itertools

import numpy as np
import torch

from vm_compiler.artifact import Artifact
from vm_compiler.circuit import Circuit
from vm_compiler.compaction import stable_compact
from vm_compiler.reference_executor import execute_artifact_reference_values


def make_fixture(count, capacity):
    circuit = Circuit()
    rows = circuit.route(circuit.input, 0, count)
    mask_weights = np.zeros((128, 1), dtype=np.float32)
    mask_weights[0, 0] = 1
    payload_weights = np.zeros((128, 2), dtype=np.float32)
    payload_weights[1, 0] = payload_weights[2, 1] = 1
    mask = circuit.project(rows, mask_weights)
    payload = circuit.project(rows, payload_weights)
    result = stable_compact(circuit, mask, payload, capacity=capacity)
    return circuit, result, Artifact.from_bytes(circuit.artifact(result.rows).to_bytes())


def check_masks(count, capacity, masks):
    circuit, result, artifact = make_fixture(count, capacity)
    source = torch.zeros(len(masks), 2048, 128)
    for batch, mask in enumerate(masks):
        source[batch, :count, 0] = torch.tensor(mask)
        source[batch, :count, 1] = torch.arange(1, count + 1)
        source[batch, :count, 2] = -torch.arange(1, count + 1)
    _, values, _ = execute_artifact_reference_values(artifact, source)
    for batch, mask in enumerate(masks):
        chosen = [index + 1 for index, bit in enumerate(mask) if bit]
        expected = torch.zeros(capacity, 2)
        for row, index in enumerate(chosen[:capacity]):
            expected[row] = torch.tensor([index, -index])
        assert torch.equal(values[result.rows][batch], expected)
        assert values[result.count][batch, 0, 0].item() == len(chosen)
        assert values[result.overflow][batch, 0, 0].item() == (len(chosen) > capacity)
        assert values[result.present][batch, :, 0].tolist() == [float(row < len(chosen)) for row in range(capacity)]
    return circuit, artifact


def test_all_six_bit_masks_exact_stable_order_padding_and_explicit_overflow():
    masks = list(itertools.product((0., 1.), repeat=6))
    for start in range(0, len(masks), 8):
        check_masks(6, 4, masks[start:start + 8])


def test_two_level_prefix_crosses_64_row_boundaries_without_quadratic_weights():
    count = 137
    mask = [float(index in (0, 63, 64, 127, 136)) for index in range(count)]
    circuit, artifact = check_masks(count, 256, [mask])
    assert (count, count) not in [tensor.shape for tensor in artifact.tensors]
    assert circuit.memory_estimate()["frozen_fp32_bytes"] < 64 * 1024


def test_gather_payload_derivative_reaches_only_selected_rows():
    _, result, artifact = make_fixture(6, 4)
    source = torch.zeros(1, 2048, 128)
    source[0, :6, 0] = torch.tensor([0., 1., 0., 1., 1., 0.])
    source.requires_grad_()
    _, values, _ = execute_artifact_reference_values(artifact, source)
    values[result.rows].sum().backward()
    expected = torch.tensor([0., 1., 0., 1., 1., 0.])
    assert torch.equal(source.grad[0, :6, 1], expected)
    assert torch.equal(source.grad[0, :6, 2], expected)
    assert torch.count_nonzero(source.grad[0, 6:]) == 0
