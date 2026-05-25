from __future__ import annotations

import torch

from chess_machine_zero.model.baseline import CMZOutcomeBaseline
from chess_machine_zero.model.ranker import CMZMoveRanker
from chess_machine_zero.selfplay.actor import SelfPlayActor, SelfPlayConfig
from chess_machine_zero.selfplay.replay import ReplayStore
from chess_machine_zero.train.losses import train_ranker_baseline_step
from chess_machine_zero.vm.interpreter import ChessMachineVM


def test_training_step_consumes_replay_batch_and_updates_ranker_baseline() -> None:
    torch.manual_seed(20260524)
    ranker = CMZMoveRanker(seed=20260524)
    baseline = CMZOutcomeBaseline(seed=20260524)
    actor = SelfPlayActor(
        vm=ChessMachineVM(seed=20260524),
        ranker=ranker,
        config=SelfPlayConfig(max_plies=6, temperature=1.0),
    )
    store = ReplayStore()
    for game_id in range(4):
        store.add_game(actor.generate_game(game_id=game_id, seed=20_000 + game_id))
    batch = store.sample_deterministic(batch_size=12, seed=99)
    optimizer = torch.optim.AdamW(list(ranker.parameters()) + list(baseline.parameters()), lr=0.01)
    before = [parameter.detach().clone() for parameter in list(ranker.parameters()) + list(baseline.parameters())]

    stats = train_ranker_baseline_step(ranker, baseline, batch, optimizer)

    after = list(ranker.parameters()) + list(baseline.parameters())
    assert torch.isfinite(torch.tensor(stats.total_loss))
    assert torch.isfinite(torch.tensor(stats.policy_loss))
    assert torch.isfinite(torch.tensor(stats.baseline_loss))
    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, after, strict=True))
