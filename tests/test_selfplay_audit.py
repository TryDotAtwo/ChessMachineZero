from __future__ import annotations

from chess_machine_zero.model.ranker import CMZMoveRanker
from chess_machine_zero.selfplay.actor import SelfPlayActor, SelfPlayConfig
from chess_machine_zero.selfplay.audit import audit_game_record, game_record_trace_hash
from chess_machine_zero.vm.interpreter import ChessMachineVM


def _actor(max_plies: int) -> SelfPlayActor:
    return SelfPlayActor(
        vm=ChessMachineVM(seed=20260524),
        ranker=CMZMoveRanker(seed=20260524),
        config=SelfPlayConfig(max_plies=max_plies, temperature=1.0),
    )


def test_selfplay_audit_verifies_10000_generated_plies_without_illegal_commits() -> None:
    actor = _actor(max_plies=128)
    total_decisions = 0
    total_games = 0
    for game_id in range(128):
        game = actor.generate_game(game_id=game_id, seed=100_000 + game_id)
        audit = audit_game_record(game)
        assert audit.illegal_commit_count == 0
        assert audit.missing_outcome_count == 0
        assert audit.replay_count == audit.decision_count
        assert audit.terminal is True
        total_decisions += audit.decision_count
        total_games += 1
        if total_decisions >= 10_000:
            break

    assert total_decisions >= 10_000
    assert total_games <= 128


def test_same_seed_and_same_ranker_produce_same_game_trace_hash() -> None:
    actor_a = _actor(max_plies=48)
    actor_b = _actor(max_plies=48)

    game_a = actor_a.generate_game(game_id=1, seed=777)
    game_b = actor_b.generate_game(game_id=1, seed=777)

    assert audit_game_record(game_a) == audit_game_record(game_b)
    assert game_record_trace_hash(game_a) == game_record_trace_hash(game_b)
    assert [decision.chosen_move.to_uci() for decision in game_a.decisions] == [
        decision.chosen_move.to_uci() for decision in game_b.decisions
    ]
