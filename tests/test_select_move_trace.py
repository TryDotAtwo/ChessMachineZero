from __future__ import annotations

from chess_machine_zero.chess.board_io import STARTING_FEN
from chess_machine_zero.model.ranker import CMZMoveRanker
from chess_machine_zero.selfplay.actor import SelfPlayActor, SelfPlayConfig
from chess_machine_zero.vm.interpreter import ChessMachineVM
from chess_machine_zero.vm.trace_packet import TraceOp


def test_select_move_trace_scores_only_legal_candidates_and_commits_legal_move() -> None:
    actor = SelfPlayActor(
        vm=ChessMachineVM(seed=1234),
        ranker=CMZMoveRanker(seed=20260524),
        config=SelfPlayConfig(max_plies=8, temperature=1.0),
    )
    decision = actor.select_move(STARTING_FEN, ply=0, seed=777)
    ops = [packet.op for packet in decision.trace]

    assert TraceOp.CANDIDATE in ops
    assert TraceOp.LEGAL_SET in ops
    assert ops.count(TraceOp.SCORE_SET) == len(decision.legal_moves)
    assert ops.count(TraceOp.SAMPLE_SET) == 1
    assert ops.count(TraceOp.COMMIT_MOVE) == 1
    assert decision.chosen_move.to_uci() in {move.to_uci() for move in decision.legal_moves}
