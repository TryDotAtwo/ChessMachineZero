from __future__ import annotations

import torch

from chess_machine_zero.chess.board_io import STARTING_FEN, parse_fen
from chess_machine_zero.model.baseline import CMZOutcomeBaseline
from chess_machine_zero.trace.reconstruct import reconstruct_board_squares
from chess_machine_zero.trace.windows import TraceWindow
from chess_machine_zero.vm.interpreter import ChessMachineVM
from chess_machine_zero.vm.lookahead import InternalTraceLookahead
from chess_machine_zero.vm.trace_packet import TraceOp


def test_trace_negamax_depth1_emits_child_transitions_and_scores() -> None:
    torch.manual_seed(20260524)
    vm = ChessMachineVM(seed=20260524)
    lookahead = InternalTraceLookahead(vm=vm, baseline=CMZOutcomeBaseline(seed=20260524), max_trace_packets=4096)
    result = lookahead.trace_negamax(STARTING_FEN, depth=1, ply=0)
    legal_moves = vm.legal_moves(STARTING_FEN)

    assert tuple(move.to_uci() for move in result.legal_moves) == tuple(move.to_uci() for move in legal_moves)
    assert result.depth == 1
    assert sum(1 for packet in result.trace if packet.op is TraceOp.SCORE_SET) == len(legal_moves)
    assert sum(1 for packet in result.trace if packet.op is TraceOp.COMMIT_MOVE) == len(legal_moves)
    for child in result.children:
        expected_fen = vm.make_move(STARTING_FEN, child.move)
        assert reconstruct_board_squares(child.make_move_trace) == parse_fen(expected_fen).squares


def test_trace_negamax_depth_zero_uses_baseline_leaf_value() -> None:
    baseline = CMZOutcomeBaseline(seed=20260524)
    lookahead = InternalTraceLookahead(vm=ChessMachineVM(seed=20260524), baseline=baseline, max_trace_packets=256)
    result = lookahead.trace_negamax(STARTING_FEN, depth=0, ply=0)

    assert result.depth == 0
    assert result.children == ()
    assert isinstance(result.value, float)
    assert any(packet.op is TraceOp.SCORE_SET for packet in result.trace)


def test_trace_window_keeps_latest_packets_only() -> None:
    vm = ChessMachineVM(seed=20260524)
    full = vm.legal_move_trace(STARTING_FEN)
    window = TraceWindow(max_packets=8)
    for packet in full:
        window.append(packet)

    assert len(window.to_tuple()) == 8
    assert window.to_tuple() == tuple(full[-8:])
