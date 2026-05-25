from __future__ import annotations

from chess_machine_zero.model.ranker import CMZMoveRanker
from chess_machine_zero.selfplay.actor import SelfPlayActor, SelfPlayConfig
from chess_machine_zero.vm.interpreter import ChessMachineVM


def test_100_complete_selfplay_games_have_no_illegal_moves_and_record_outcomes() -> None:
    actor = SelfPlayActor(
        vm=ChessMachineVM(seed=20260524),
        ranker=CMZMoveRanker(seed=20260524),
        config=SelfPlayConfig(max_plies=8, temperature=1.0),
    )
    games = [actor.generate_game(game_id=game_id, seed=10_000 + game_id) for game_id in range(100)]

    assert len(games) == 100
    for game in games:
        assert game.terminal_status.is_terminal
        assert len(game.decisions) == len(game.replay_records)
        for decision in game.decisions:
            legal_uci = {move.to_uci() for move in decision.legal_moves}
            assert decision.chosen_move.to_uci() in legal_uci
        for record in game.replay_records:
            assert record.final_outcome_from_side_to_move in (-1.0, 0.0, 1.0)
            assert record.chosen_move.to_uci() in record.legal_uci
