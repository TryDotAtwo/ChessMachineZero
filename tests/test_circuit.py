"""Exhaustive discrete and independent surrogate tests of frozen circuit lowering."""

import math

import numpy as np
import pytest
import torch

from vm_compiler.artifact import Artifact
from vm_compiler.circuit import Circuit
from vm_compiler.graph import OpCode
from vm_compiler.reference_executor import execute_artifact_reference


def columns(circuit, count):
    rows = circuit.route(circuit.input, 0, count)
    left = np.zeros((128, 1), dtype=np.float32)
    right = left.copy()
    left[0, 0] = right[1, 0] = 1
    return circuit.project(rows, left, "left"), circuit.project(rows, right, "right")


def test_boolean_circuits_exhaustive_forward_and_batch_isolation():
    circuit = Circuit()
    a, b = columns(circuit, 4)
    output = circuit.concat_columns((
        circuit.logical_and(a, b), circuit.logical_or(a, b),
        circuit.logical_not(a), circuit.select(a, b, circuit.logical_not(b)),
    ), "truth_table")
    artifact = Artifact.from_bytes(circuit.artifact(output).to_bytes())
    source = torch.zeros(2, 2048, 128)
    source[0, :4, :2] = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    source[1, :4, :2] = source[0, :4, :2].flip(0)
    actual = execute_artifact_reference(artifact, source)
    expected = torch.tensor([[0., 0., 1., 1.], [0., 1., 1., 0.],
                             [0., 1., 0., 0.], [1., 1., 0., 1.]])
    assert torch.equal(actual, torch.stack((expected, expected.flip(0))))
    assert set(op.opcode for op in artifact.operations) <= {
        OpCode.ROW_ROUTE, OpCode.TOKEN_PROJECT, OpCode.POSITION_ADD,
        OpCode.RESIDUAL_ADD, OpCode.MATRIX_TRANSPOSE, OpCode.MATRIX_RESHAPE,
        OpCode.ROW_CONCAT, OpCode.HARDMAX_STE,
    }
    assert circuit.semantics[output]["name"] == "truth_table"
    assert circuit.shapes[output] == (4, 4)


def test_threshold_gradient_matches_independent_logistic_derivative():
    circuit = Circuit()
    a, _ = columns(circuit, 3)
    output = circuit.threshold(a, .5, temperature=.5)
    source = torch.zeros(1, 2048, 128, dtype=torch.float64)
    source[0, :3, 0] = torch.tensor([0., .5, 1.])
    source.requires_grad_()
    actual = execute_artifact_reference(circuit.artifact(output), source)
    # A tie selects column zero, hence exactly zero, not 0.5.
    assert torch.equal(actual[0, :, 0], torch.tensor([0., 0., 1.], dtype=torch.float64))
    actual.sum().backward()
    expected = torch.zeros_like(source)
    for row, value in enumerate((0., .5, 1.)):
        p = 1 / (1 + math.exp(-(value - .5) / .5))
        expected[0, row, 0] = p * (1 - p) / .5
    torch.testing.assert_close(source.grad, expected, rtol=1e-12, atol=1e-12)


def test_constants_round_trip_exactly_including_odd_count_and_non_lattice():
    circuit = Circuit()
    output = circuit.constant("fractions", np.array([[.125], [-7.25], [0.]], dtype=np.float32))
    artifact = Artifact.from_bytes(circuit.artifact(output).to_bytes())
    actual = execute_artifact_reference(artifact, torch.zeros(2, 2048, 128))
    assert torch.equal(actual, torch.tensor([[[.125], [-7.25], [0.]]] * 2))
    estimate = circuit.memory_estimate(batch_size=2)
    assert estimate["frozen_fp32_bytes"] == 12
    assert estimate["retained_values_fp32_bytes"] == 2 * (2048 * 128 + 3) * 4
    assert estimate["excludes_autograd_and_workspace"] is True


def test_static_shape_validation_does_not_build_invalid_graphs():
    circuit = Circuit()
    a, _ = columns(circuit, 4)
    with pytest.raises(ValueError, match="reshape"):
        circuit.reshape(a, 2, 3)
    with pytest.raises(ValueError, match="matmul"):
        circuit.matmul(a, a)
    with pytest.raises(ValueError, match="projection"):
        circuit.project(a, np.ones((2, 3)), "bad")
    with pytest.raises(ValueError, match="broadcast"):
        circuit.add(a, circuit.constant("three_rows", np.ones((3, 2))))
    with pytest.raises(ValueError, match="finite"):
        circuit.constant("bad_nan", np.array([[np.nan]]))
    with pytest.raises(ValueError, match="temperature"):
        circuit.threshold(a, .5, temperature=.0001)


def test_include_graph_remaps_sources_and_preserves_position_values():
    from vm_compiler.compiler import build_position_reconstruction_artifact
    from vm_compiler.state_circuit import build_initial_piece_state

    circuit = Circuit()
    # A preceding op makes imported SSA IDs differ from the original IDs.
    source = circuit.bias(circuit.input, 0.)
    output = circuit.include_graph(build_position_reconstruction_artifact(), source)
    assert circuit.shapes[output] == (64, 128)
    request = torch.zeros(2, 2048, 128)
    request[:, :, 0] = 1
    actual = execute_artifact_reference(circuit.artifact(output), request)
    expected = torch.from_numpy(build_initial_piece_state()).unsqueeze(0).expand(2, -1, -1)
    assert torch.equal(actual, expected)


def test_include_graph_values_exposes_remapped_intermediates_without_reexecution():
    from vm_compiler.compiler import build_position_reconstruction_artifact

    imported = build_position_reconstruction_artifact()
    circuit = Circuit()
    source = circuit.bias(circuit.input, 0.)
    values = circuit.include_graph_values(imported, source)

    assert values[0] == source
    assert circuit.shapes[values[40]] == (2064, 2)   # dynamic event keys
    assert circuit.shapes[values[41]] == (64, 2)     # square queries
    assert circuit.shapes[values[44]] == (2064, 128) # dynamic event values
    assert circuit.shapes[values[45]] == (64, 128)   # final board
    assert len(circuit.operations) == 1 + len(imported.operations)
