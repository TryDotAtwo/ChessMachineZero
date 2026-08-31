"""Full legal-set equality; the tested subgraph is not a recurrent VM."""

import pytest
import torch

from chess_oracle import legal_history_snapshots, legal_set_for_history_input, legal_set_snapshots
from vm_compiler.artifact import Artifact
from vm_compiler.legal_circuit import build_legal_artifact
from vm_compiler.reference_executor import execute_artifact_reference, execute_artifact_reference_values


@pytest.fixture(scope="module")
def legal_artifact():
    circuit, output = build_legal_artifact()
    # Preflight logical payloads, not a claim about peak allocator/autograd use.
    estimate = circuit.memory_estimate()
    assert estimate["frozen_fp32_bytes"] < 64 * 1024**2
    assert estimate["retained_values_fp32_bytes"] < 1024**3
    return Artifact.from_bytes(circuit.artifact(output).to_bytes())


@pytest.mark.parametrize("moves", [
    "", "e2e4 d7d5 e4d5", "f2f3 e7e5 g2g4 d8h4",
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1",
    "d2d4 d7d5 b1c3 b8c6 c1f4 c8f5 d1d2 d8d7 e1c1 e8c8",
    "e2e4 a7a6 e4e5 d7d5 e5d6",
    "a2a4 h7h5 a4a5 h5h4 a5a6 h4h3 a6b7 h3g2 b7a8q g2h1q",
    "h2h4 a7a6 h1h3 a6a5 h3h1 a5a4 g2g3 d7d5 f1g2 d5d4 g1f3 b7b6",
    "g2g3 b7b6 f1g2 c8a6 g1f3 e7e6 e2e4 h7h6",
    "e2e4 e7e5 f1c4 b8c6 g1h3 d7d6 f2f4 d8h4",
])
def test_every_legal_triple_and_padding_for_all_prefixes(legal_artifact, moves):
    with torch.no_grad():
        for ply, (source, expected) in enumerate(legal_set_snapshots(moves.split())):
            actual = execute_artifact_reference(legal_artifact, torch.from_numpy(source).unsqueeze(0))[0]
            assert torch.equal(actual, torch.from_numpy(expected)), f"{moves}, prefix={ply}"


def test_long_history_legal_sets_through_capacity_and_promotions(legal_artifact):
    snapshots, special = legal_history_snapshots(seed=1)
    prefixes = sorted({28, 29, 244, 245, 246, 398, 399, 400, *special["promotion"]})
    with torch.no_grad():
        for ply in prefixes:
            source = snapshots[ply][0]
            expected = legal_set_for_history_input(source)
            actual = execute_artifact_reference(legal_artifact, torch.from_numpy(source).unsqueeze(0))[0]
            assert torch.equal(actual, torch.from_numpy(expected)), f"seed1, prefix={ply}"


def test_named_legal_diagnostics_and_request_is_not_applied():
    circuit, output = build_legal_artifact()
    source = torch.zeros(1, 2048, 128)
    source[:, :, 0] = 1
    source[:, :3, 0] = 0
    source[:, :3, 127] = 1
    _, values, _ = execute_artifact_reference_values(circuit.artifact(output), source)
    named = {name: values[value] for name, value in circuit.named_outputs.items()}
    assert named["legal_count"].item() == 20
    assert named["legal_capacity_overflow"].item() == 0
    assert named["black_to_move"].item() == 0
    assert named["legal_slots_present"][0, :, 0].tolist() == [1.] * 20 + [0.] * 236
    assert torch.equal(named["legal_set"][0], torch.from_numpy(legal_set_snapshots([])[0][1]))
