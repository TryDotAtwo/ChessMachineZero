"""Exact full-context acceptance for the recurrent frozen tensor artifact."""

import random

import chess
import numpy
import pytest
import torch

from chess_oracle import _native_rows
from vm_compiler.artifact import Artifact
from vm_compiler.context import build_context_0
from vm_compiler.oracle import transition_oracle
from vm_compiler.protocol import (
    CONTEXT_ROWS,
    INPUT_ROWS,
    PIECE_CHANNELS,
    STATUS_CHANNELS,
    VOCAB_SIZE,
    encode_move_one_hot,
)
from vm_compiler.recurrent_circuit import build_recurrent_artifact
from vm_compiler.reference_executor import (
    execute_artifact_reference,
    execute_artifact_reference_values,
)


WP = PIECE_CHANNELS["WHITE_PAWN"]
WN = PIECE_CHANNELS["WHITE_KNIGHT"]
BP = PIECE_CHANNELS["BLACK_PAWN"]
BQ = PIECE_CHANNELS["BLACK_QUEEN"]


def _source(context, request):
    source = numpy.concatenate((request, context)).astype(numpy.float32, copy=False)
    assert source.shape == (INPUT_ROWS, VOCAB_SIZE)
    return torch.from_numpy(source).unsqueeze(0)


@pytest.fixture(scope="module")
def recurrent_artifact():
    circuit, output = build_recurrent_artifact()
    artifact = Artifact.from_bytes(circuit.artifact(output).to_bytes())
    context_records = [record for record in artifact.tensors if record.name == "context_0"]
    assert len(context_records) == 1
    assert context_records[0].shape == (CONTEXT_ROWS, VOCAB_SIZE)
    return artifact


@pytest.mark.parametrize("move", [
    (52, 54, WP),       # legal e2e4
    (52, 55, WP),       # illegal e2e5
    (52, 54, WN),       # legal geometry with a false result-piece declaration
])
def test_initial_request_emits_the_exact_oracle_context(recurrent_artifact, move):
    before = build_context_0()
    request = encode_move_one_hot(*move)
    expected = transition_oracle(before, request)
    with torch.no_grad():
        actual = execute_artifact_reference(recurrent_artifact, _source(before, request))[0]
    assert torch.equal(actual, torch.from_numpy(expected))


def test_generated_legal_set_has_finite_nonzero_gradient_to_requested_move(
        recurrent_artifact):
    source = _source(
        build_context_0(),
        encode_move_one_hot(52, 54, WP),
    ).requires_grad_()
    actual = execute_artifact_reference(recurrent_artifact, source)
    weights = torch.linspace(-1.0, 1.0, 768 * VOCAB_SIZE).reshape(
        1, 768, VOCAB_SIZE
    )
    (actual[:, 1200:1968] * weights).sum().backward()

    request_gradient = source.grad[:, :3]
    assert torch.isfinite(request_gradient).all()
    assert torch.count_nonzero(request_gradient).item() > 0


def test_output_context_is_fed_back_unchanged_for_the_black_reply(recurrent_artifact):
    context = build_context_0()
    moves = ((52, 54, WP), (47, 45, BP))
    with torch.no_grad():
        for move in moves:
            request = encode_move_one_hot(*move)
            expected = transition_oracle(context, request)
            actual = execute_artifact_reference(recurrent_artifact, _source(context, request))[0]
            assert torch.equal(actual, torch.from_numpy(expected))
            context = actual.numpy()

    status_row = 1200 + 768
    assert context[status_row, STATUS_CHANNELS["OK"]] == 1.0


def test_fools_mate_emits_black_win_and_is_absorbing(recurrent_artifact):
    context = build_context_0()
    moves = ((62, 63, WP), (57, 55, BP), (72, 74, WP), (48, 84, BQ))
    with torch.no_grad():
        for ply, move in enumerate(moves, 1):
            request = encode_move_one_hot(*move)
            expected = transition_oracle(context, request)
            actual = execute_artifact_reference(recurrent_artifact, _source(context, request))[0]
            assert torch.equal(actual, torch.from_numpy(expected)), f"fools-mate ply {ply}"
            context = actual.numpy()

        before = context.copy()
        request = encode_move_one_hot(52, 54, WP)
        actual = execute_artifact_reference(recurrent_artifact, _source(context, request))[0]
    assert numpy.array_equal(actual.numpy(), before)


def test_shortest_stalemate_emits_draw_and_empty_legal_set(recurrent_artifact):
    moves = (
        "e2e3 a7a5 d1h5 a8a6 h5a5 h7h5 a5c7 a6h6 h2h4 f7f6 "
        "c7d7 e8f7 d7b7 d8d3 b7b8 d3h7 b8c8 f7g6 c8e6"
    ).split()
    context, request = _context_before_uci_request(moves)
    expected = transition_oracle(context, request)
    assert expected[1968, STATUS_CHANNELS["DRAW"]] == 1.0
    assert expected[1200:1968, 0].sum() == 768
    with torch.no_grad():
        actual = execute_artifact_reference(recurrent_artifact, _source(context, request))[0]
    assert torch.equal(actual, torch.from_numpy(expected))


def test_future_reply_signal_excludes_stalemating_move_but_keeps_ordinary_move():
    """The halfmove-99 lookahead must require a reply after the candidate."""

    moves = (
        "e2e3 a7a5 d1h5 a8a6 h5a5 h7h5 a5c7 a6h6 h2h4 f7f6 "
        "c7d7 e8f7 d7b7 d8d3 b7b8 d3h7 b8c8 f7g6"
    ).split()
    context, request = _context_before_uci_request(moves)
    circuit, output = build_recurrent_artifact()
    with torch.no_grad():
        _, values, _ = execute_artifact_reference_values(
            circuit.artifact(output), _source(context, request)
        )
    named = {name: values[value] for name, value in circuit.named_outputs.items()}
    legal = named["next_legal_set"][0].reshape(256, 3, VOCAB_SIZE).argmax(dim=2)
    reply_exists = named["future_has_legal_reply"][0, :, 0]

    board = chess.Board()
    for prior in moves:
        board.push_uci(prior)
    by_triple = {
        tuple(int(channel) for channel in _native_rows(board, move).argmax(axis=1)): move
        for move in board.legal_moves
    }
    for index, triple_tensor in enumerate(legal):
        triple = tuple(int(channel) for channel in triple_tensor.tolist())
        if triple[0] == 0:
            continue
        move = by_triple[triple]
        if board.is_zeroing(move):
            continue
        board.push(move)
        expected = float(any(board.generate_legal_moves()))
        board.pop()
        assert reply_exists[index].item() == expected, move.uci()

    def slot(uci):
        triple = torch.from_numpy(_native_rows(
            board, chess.Move.from_uci(uci)
        ).argmax(axis=1))
        matches = torch.all(legal == triple, dim=1).nonzero().flatten()
        assert matches.numel() == 1
        return int(matches.item())

    assert reply_exists[slot("c8e6")].item() == 0.0  # immediate stalemate
    assert reply_exists[slot("c8e8")].item() == 1.0  # black has a reply


def _full_legal_context():
    board = chess.Board()
    context = build_context_0()
    rng = random.Random(1)
    for ply in range(400):
        candidates = list(board.legal_moves)
        rng.shuffle(candidates)
        for move in candidates:
            board.push(move)
            ongoing = not board.is_game_over(claim_draw=True)
            board.pop()
            if ongoing:
                break
        else:
            raise AssertionError("fixture cannot continue through history capacity")
        context[ply * 3:ply * 3 + 3] = _native_rows(board, move)
        board.push(move)

    triples = sorted(tuple(_native_rows(board, move).argmax(axis=1)) for move in board.legal_moves)
    channels = [channel for triple in triples for channel in triple]
    channels.extend([0] * (768 - len(channels)))
    context[1200:1968] = numpy.eye(VOCAB_SIZE, dtype=numpy.float32)[channels]
    return context, encode_move_one_hot(*(int(channel) for channel in triples[0]))


def _context_before_uci_request(uci_moves):
    board = chess.Board()
    context = build_context_0()
    for ply, uci in enumerate(uci_moves[:-1]):
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        context[ply * 3:ply * 3 + 3] = _native_rows(board, move)
        board.push(move)
        assert board.outcome(claim_draw=True) is None

    triples = sorted(tuple(_native_rows(board, move).argmax(axis=1)) for move in board.legal_moves)
    channels = [int(channel) for triple in triples for channel in triple]
    channels.extend([0] * (768 - len(channels)))
    context[1200:1968] = numpy.eye(VOCAB_SIZE, dtype=numpy.float32)[channels]
    request_move = chess.Move.from_uci(uci_moves[-1])
    assert request_move in board.legal_moves
    return context, _native_rows(board, request_move)


def test_full_history_reports_overflow_without_mutating_context(recurrent_artifact):
    context, request = _full_legal_context()
    expected = transition_oracle(context, request)
    with torch.no_grad():
        actual = execute_artifact_reference(recurrent_artifact, _source(context, request))[0]
    assert torch.equal(actual, torch.from_numpy(expected))


def test_next_move_threefold_claim_terminates_automatically(recurrent_artifact):
    moves = "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1".split()
    context, request = _context_before_uci_request(moves)
    expected = transition_oracle(context, request)
    assert expected[1968, STATUS_CHANNELS["DRAW"]] == 1.0
    with torch.no_grad():
        actual = execute_artifact_reference(recurrent_artifact, _source(context, request))[0]
    assert torch.equal(actual, torch.from_numpy(expected))


def test_expired_legal_en_passant_does_not_count_as_the_same_repetition_key(
        recurrent_artifact):
    # After e2e4 the black d4 pawn has a legal EP capture. A reversible knight
    # cycle restores board/turn/rights but not that EP right, so it is a
    # different transposition key and must not create an early third occurrence.
    moves = (
        "g1f3 d7d5 f3g1 d5d4 e2e4 g8f6 g1f3 f6g8 "
        "f3g1 g8f6 g1f3 f6g8"
    ).split()
    context, request = _context_before_uci_request(moves)
    expected = transition_oracle(context, request)
    assert expected[1968, STATUS_CHANNELS["OK"]] == 1.0
    with torch.no_grad():
        actual = execute_artifact_reference(recurrent_artifact, _source(context, request))[0]
    assert torch.equal(actual, torch.from_numpy(expected))


def test_next_move_fifty_move_claim_terminates_automatically(recurrent_artifact):
    # Literal legal 99-halfmove fixture found offline. No pawn move or capture
    # resets the clock, and no earlier claimable repetition occurs.
    moves = (
        "g1h3 b8c6 h3g5 g8f6 g5h3 c6a5 h3f4 f6g4 f4e6 g4h6 "
        "b1c3 a5c6 c3a4 c6a5 a4c5 h6f5 c5d3 a5b3 e6d4 f5d6 "
        "h1g1 d6f5 g1h1 b3a5 a1b1 a5c6 d4b3 f5d6 b3a1 d6e4 "
        "d3e5 c6a5 h1g1 a8b8 e5d3 e4f6 d3b4 h8g8 b4d5 a5c4 "
        "d5c3 f6h5 c3d5 h5g3 d5e3 g8h8 e3d5 c4a5 d5b6 g3e4 "
        "b6a8 e4g3 g1h1 a5c4 a1b3 c4d6 b3a1 h8g8 a1b3 g8h8 "
        "b3d4 d6b5 b1a1 g3e4 d4f3 b5d6 f3g1 d6b5 a1b1 b5d6 "
        "b1a1 d6f5 g1h3 f5h6 h3f4 e4g5 f4e6 g5h3 a8b6 h6g8 "
        "b6c4 h3f4 e6d4 b8a8 c4a5 f4h5 a1b1 h5f4 a5b3 f4d5 "
        "d4c6 d5e3 c6b8 e3f5 b8c6 f5d6 b3a1 g8h6 c6a5"
    ).split()
    assert len(moves) == 99
    context, request = _context_before_uci_request(moves)
    expected = transition_oracle(context, request)
    assert expected[1968, STATUS_CHANNELS["DRAW"]] == 1.0
    with torch.no_grad():
        actual = execute_artifact_reference(recurrent_artifact, _source(context, request))[0]
    assert torch.equal(actual, torch.from_numpy(expected))


def test_insufficient_material_terminates_automatically(recurrent_artifact):
    moves = (
        "h2h4 h7h6 g2g3 e7e6 h1h3 f7f5 f2f4 h6h5 b2b4 d8h4 "
        "h3h4 g8e7 f1h3 h8h6 b1a3 b8a6 h3f5 e7f5 g3g4 f8b4 "
        "c2c4 f5h4 c4c5 b4d2 d1d2 e8e7 d2d7 e7d7 e1f2 b7b5 "
        "g4h5 h6h5 a3b5 h5c5 b5a7 c5c4 a1b1 c4c1 b1c1 a8a7 "
        "e2e4 a6b8 g1e2 a7a3 c1c4 a3b3 e2d4 d7e8 c4c7 b3h3 "
        "c7g7 h3e3 d4c6 e3c3 g7g5 e8f7 g5g8 c3c1 c6b8 c1c4 "
        "g8c8 c4c1 b8c6 c1e1 f2e1 h4f3 e1f1 f3e1 c6b8 f7g6 "
        "c8c7 e1f3 a2a3 f3h2 f1e1 g6h5 c7c1 h5g6 b8c6 g6f7 "
        "e1d1 h2g4 d1d2 g4h6 d2e1 f7f6 c6b4 e6e5 f4e5 f6e6 "
        "c1b1 h6f5 b4a2 f5d6 b1b2 d6b7 b2d2 e6e5 a3a4 e5e4 "
        "d2h2 b7a5 e1f2 a5b7 f2f1 e4d5 h2h3 b7a5 h3b3 d5e4 "
        "f1e1 a5b3 e1f1 e4d4 f1e2 b3a5 e2f1 a5b7 f1f2 b7c5 "
        "a4a5 c5e4 f2e1 e4f6 a2c1 d4c4 c1b3 c4d5 e1f1 f6h5 "
        "f1e1 h5g3 e1d2 d5c4 d2d1 g3h1 b3d4 h1g3 d4b3 c4b3 "
        "d1c1 g3e4 a5a6 e4g5 c1d1 g5f3 a6a7 b3a2 a7a8n f3g5 "
        "d1c2 g5e4 a8b6 e4c5 b6d7 c5d7"
    ).split()
    context, request = _context_before_uci_request(moves)
    expected = transition_oracle(context, request)
    assert expected[1968, STATUS_CHANNELS["DRAW"]] == 1.0
    with torch.no_grad():
        actual = execute_artifact_reference(recurrent_artifact, _source(context, request))[0]
    assert torch.equal(actual, torch.from_numpy(expected))
