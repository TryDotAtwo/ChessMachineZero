"""Trace-based self-play actor."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

from chess_machine_zero.chess.board_io import STARTING_FEN, parse_fen
from chess_machine_zero.chess.move_packet import MovePacket
from chess_machine_zero.chess.outcome import TerminalReason
from chess_machine_zero.model.ranker import CMZMoveRanker
from chess_machine_zero.selfplay.game_record import GameRecord, MoveDecision, ReplayRecord, outcome_from_side
from chess_machine_zero.vm.decision_program import select_move_program
from chess_machine_zero.vm.interpreter import ChessMachineVM, legal_moves_from_trace, position_key, terminal_status
from chess_machine_zero.vm.trace_hash import trace_hash_hex
from chess_machine_zero.vm.trace_packet import TraceOp, TracePacket, TraceTag


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    start_fen: str = STARTING_FEN
    max_plies: int = 256
    temperature: float = 1.0
    rules_version: str = "cmz-rules-v1"
    model_checkpoint_id: str = "untrained-ranker"


@dataclass(frozen=True, slots=True)
class SelectionResult:
    trace: tuple[TracePacket, ...]
    legal_moves: tuple[MovePacket, ...]
    chosen_move: MovePacket
    logprob_chosen: float
    entropy: float
    chosen_rank: int
    trace_hash_before: str
    trace_hash_after: str


class SelfPlayActor:
    """Generates games by scoring only legal VM-emitted candidates."""

    def __init__(self, vm: ChessMachineVM, ranker: CMZMoveRanker, config: SelfPlayConfig) -> None:
        self.vm = vm
        self.ranker = ranker
        self.config = config
        select_move_program()

    def select_move(self, fen: str, ply: int, seed: int, game_id: int = 0) -> MoveDecision:
        board = parse_fen(fen)
        selection = self._select(fen, seed)
        return MoveDecision(
            game_id=game_id,
            ply=ply,
            side_to_move=board.side_to_move,
            fen_before=fen,
            legal_moves=selection.legal_moves,
            chosen_move=selection.chosen_move,
            logprob_chosen=selection.logprob_chosen,
            entropy=selection.entropy,
            temperature=self.config.temperature,
            trace=selection.trace,
            trace_hash_before=selection.trace_hash_before,
            trace_hash_after=selection.trace_hash_after,
            stochastic_seed=seed,
        )

    def generate_game(self, game_id: int, seed: int) -> GameRecord:
        rng = random.Random(seed)
        fen = self.config.start_fen
        start_fen = fen
        game_trace: list[TracePacket] = []
        decisions: list[MoveDecision] = []
        repetitions = {position_key(parse_fen(fen)): 1}
        terminal = terminal_status(parse_fen(fen), 0, 1, False)
        for ply in range(self.config.max_plies):
            board = parse_fen(fen)
            terminal = terminal_status(board, ply, repetitions[position_key(board)], False)
            if terminal.is_terminal:
                break
            decision_seed = rng.randrange(0, 2**31 - 1)
            decision = self.select_move(fen, ply, decision_seed, game_id)
            decisions.append(decision)
            game_trace.extend(decision.trace)
            fen = self.vm.make_move(fen, decision.chosen_move)
            next_board = parse_fen(fen)
            repetitions[position_key(next_board)] = repetitions.get(position_key(next_board), 0) + 1
        else:
            terminal = terminal_status(parse_fen(fen), self.config.max_plies, repetitions[position_key(parse_fen(fen))], True)
        if not terminal.is_terminal:
            terminal = terminal_status(parse_fen(fen), len(decisions), repetitions[position_key(parse_fen(fen))], False)
        if not terminal.is_terminal:
            terminal = terminal_status(parse_fen(fen), len(decisions), repetitions[position_key(parse_fen(fen))], True)
        if terminal.reason is TerminalReason.ADJUDICATION_CAP:
            game_trace.extend(self.vm.terminal_trace(parse_fen(fen), len(decisions), adjudication_cap_reached=True))
        replay_records = tuple(self._replay_record(decision, terminal) for decision in decisions)
        return GameRecord(
            game_id=game_id,
            start_fen=start_fen,
            final_fen=fen,
            terminal_status=terminal,
            decisions=tuple(decisions),
            replay_records=replay_records,
            trace=tuple(game_trace),
        )

    def _select(self, fen: str, seed: int) -> SelectionResult:
        board = parse_fen(fen)
        legal_trace = self.vm.legal_move_trace(fen)
        legal_moves = tuple(legal_moves_from_trace(legal_trace))
        if not legal_moves:
            raise ValueError("select_move requires at least one legal move")
        before_hash = trace_hash_hex(legal_trace)
        with torch.no_grad():
            scores = self.ranker.score_moves(legal_moves, board.side_to_move).detach().float()
        probabilities = _probabilities(scores, self.config.temperature)
        chosen_rank = _sample_rank(probabilities, seed)
        chosen_move = legal_moves[chosen_rank]
        logprob = float(torch.log(probabilities[chosen_rank]).item())
        entropy = float((-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()).item())
        trace = list(legal_trace)
        for move, score in zip(legal_moves, scores, strict=True):
            trace.append(TracePacket(TraceOp.SCORE_SET, move.move_id, _score_bucket(float(score.item())), 0, 0, TraceTag.MOVE, 0))
        trace.append(
            TracePacket(
                TraceOp.SAMPLE_SET,
                seed % (2**31 - 1),
                int(self.config.temperature * 1000),
                chosen_rank,
                len(legal_moves),
                TraceTag.MOVE,
                0,
            )
        )
        trace.append(
            TracePacket(
                TraceOp.COMMIT_MOVE,
                chosen_move.move_id,
                chosen_move.from_sq,
                chosen_move.to_sq,
                int(chosen_move.promo),
                TraceTag.MOVE,
                int(chosen_move.flags),
            )
        )
        return SelectionResult(
            trace=tuple(trace),
            legal_moves=legal_moves,
            chosen_move=chosen_move,
            logprob_chosen=logprob,
            entropy=entropy,
            chosen_rank=chosen_rank,
            trace_hash_before=before_hash,
            trace_hash_after=trace_hash_hex(trace),
        )

    def _replay_record(self, decision: MoveDecision, terminal_status_value) -> ReplayRecord:
        return ReplayRecord(
            game_id=decision.game_id,
            ply=decision.ply,
            side_to_move=decision.side_to_move,
            fen_before=decision.fen_before,
            legal_moves=decision.legal_moves,
            chosen_move=decision.chosen_move,
            logprob_chosen=decision.logprob_chosen,
            entropy=decision.entropy,
            temperature=decision.temperature,
            final_outcome_from_side_to_move=outcome_from_side(terminal_status_value.result, decision.side_to_move),
            trace_hash_before=decision.trace_hash_before,
            trace_hash_after=decision.trace_hash_after,
            rules_version=self.config.rules_version,
            model_checkpoint_id=self.config.model_checkpoint_id,
            stochastic_seed=decision.stochastic_seed,
        )


def _probabilities(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        probabilities = torch.zeros_like(scores)
        probabilities[int(scores.argmax().item())] = 1.0
        return probabilities
    return torch.softmax(scores / temperature, dim=0)


def _sample_rank(probabilities: torch.Tensor, seed: int) -> int:
    rng = random.Random(seed)
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities.tolist()):
        cumulative += probability
        if threshold <= cumulative or math.isclose(cumulative, 1.0):
            return index
    return len(probabilities) - 1


def _score_bucket(score: float) -> int:
    return max(0, min(2**31 - 1, int(round((score + 32.0) * 1024.0))))
