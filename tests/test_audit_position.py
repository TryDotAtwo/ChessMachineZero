"""Regressions for stale-square theft and exact legal-history reconstruction."""

import pytest
import torch

from chess_oracle import legal_history_snapshots
from vm_compiler.compiler import build_position_reconstruction_artifact
from vm_compiler.reference_executor import _materialize, execute_artifact_reference


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_seed_one_400_legal_nonterminal_plies_preserve_exact_full_boards():
    snapshots, special = legal_history_snapshots(seed=1)
    assert special["promotion"], "fixture must exercise real legal promotion"
    artifact = build_position_reconstruction_artifact()
    # The old epsilon loses square identity after 244 plies; include both sides,
    # the h1 event at 29, and the neighboring king event at 399 from the audit.
    prefixes = {0, 1, 28, 29, 244, 245, 246, 398, 399, 400}
    prefixes.update(special["promotion"])
    mismatches = []
    for ply in sorted(prefixes):
        source, expected = snapshots[ply]
        actual = execute_artifact_reference(artifact, torch.tensor(source[None]))
        if not torch.equal(actual[0], torch.tensor(expected)):
            mismatches.append(ply)
    assert not mismatches, f"full one-hot board mismatch at plies {mismatches}"


@pytest.mark.parametrize("moves,kind", [
    (("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1"), "castling"),
    (("e2e4", "a7a6", "e4e5", "d7d5", "e5d6"), "en_passant"),
])
def test_legal_special_move_prefixes_match_independent_full_board(moves, kind):
    snapshots, special = legal_history_snapshots(uci_moves=moves)
    assert special[kind] == [len(moves)]
    artifact = build_position_reconstruction_artifact()
    for source, expected in snapshots.values():
        actual = execute_artifact_reference(artifact, torch.tensor(source[None]))
        assert torch.equal(actual[0], torch.tensor(expected))


def test_materialized_fp32_scores_preserve_square_priority_and_every_timestamp():
    artifact = build_position_reconstruction_artifact()
    tensors = {record.name: _materialize(record, torch.empty((), dtype=torch.float32))
               for record in artifact.tensors}
    queries = tensors["latest_square_queries"]
    tokens = tensors["initial_square_tokens"]
    base_keys = tokens @ tensors["latest_square_key_projection"]
    # Initial time zero plus the first event family's actual compiled 1..400 biases.
    times = tensors["latest_event_time_bias"][[0, *range(64, 464)]]
    keys = base_keys[:, None, :] + times[None, :, :]
    scores = queries @ keys.reshape(-1, 2).T
    scores = scores.reshape(64, 64, 401)
    exact = scores[torch.arange(64), torch.arange(64)]
    other = ~torch.eye(64, dtype=torch.bool)
    assert bool((exact[:, :1] - scores[:, :, 400])[other].gt(0).all())
    assert bool(torch.diff(exact, dim=-1).gt(0).all())
    # Also cover every intermediate cross-square timestamp, not just extrema.
    assert bool((exact[:, :1, None] - scores)[other].gt(0).all())
