"""Analytic rules plus learned strategy shell."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

from chess_machine_zero.chess.board_io import BoardState, parse_fen
from chess_machine_zero.chess.move_packet import MovePacket
from chess_machine_zero.chess.outcome import TerminalStatus
from chess_machine_zero.model.analytic_rules import AnalyticRulesTransformer, board_state_from_prompt_trace
from chess_machine_zero.model.ranker import CMZMoveRanker
from chess_machine_zero.vm.interpreter import position_key
from chess_machine_zero.vm.interpreter import legal_moves_from_trace
from chess_machine_zero.vm.trace_packet import TraceOp, TracePacket, TraceTag


@dataclass(frozen=True, slots=True)
class AnalyticMachineDecision:
    trace: tuple[TracePacket, ...]
    legal_moves: tuple[MovePacket, ...]
    chosen_move: MovePacket
    logprob_chosen: float
    entropy: float


@dataclass(frozen=True, slots=True)
class AnalyticGameRecord:
    start_fen: str
    final_board: BoardState
    terminal_status: TerminalStatus
    decisions: tuple[AnalyticMachineDecision, ...]
    trace: tuple[TracePacket, ...]
    illegal_commit_count: int

    @property
    def decision_count(self) -> int:
        return len(self.decisions)


@dataclass(frozen=True, slots=True)
class CMZAnalyticMachine:
    rules: AnalyticRulesTransformer
    ranker: CMZMoveRanker

    def select_move_from_prompt(
        self,
        prompt_trace: tuple[TracePacket, ...] | list[TracePacket],
        seed: int,
        temperature: float,
    ) -> AnalyticMachineDecision:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        board = board_state_from_prompt_trace(prompt_trace)
        trace = list(self.rules.legal_move_trace_from_prompt(prompt_trace, include_halt=False))
        legal_moves = tuple(legal_moves_from_trace(trace))
        if not legal_moves:
            raise ValueError("analytic machine requires at least one legal move")
        with torch.no_grad():
            scores = self.ranker.score_moves(legal_moves, board.side_to_move).detach().float()
        probabilities = torch.softmax(scores / temperature, dim=0)
        chosen_rank = _sample_rank(probabilities, seed)
        chosen_move = legal_moves[chosen_rank]
        logprob = float(torch.log(probabilities[chosen_rank]).item())
        entropy = float((-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()).item())
        for move, score in zip(legal_moves, scores, strict=True):
            trace.append(TracePacket(TraceOp.SCORE_SET, move.move_id, _score_bucket(float(score.item())), 0, 0, TraceTag.MOVE, 0))
        trace.append(TracePacket(TraceOp.SAMPLE_SET, seed % (2**31 - 1), int(temperature * 1000), chosen_rank, len(legal_moves), TraceTag.MOVE, 0))
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
        return AnalyticMachineDecision(
            trace=tuple(trace),
            legal_moves=legal_moves,
            chosen_move=chosen_move,
            logprob_chosen=logprob,
            entropy=entropy,
        )

    def play_game_from_fen(
        self,
        fen: str,
        seed: int,
        max_plies: int,
        temperature: float,
    ) -> AnalyticGameRecord:
        if max_plies <= 0:
            raise ValueError("max_plies must be positive")
        rng = random.Random(seed)
        board = parse_fen(fen)
        prompt = self.rules.prompt_trace_from_board(board)
        repetitions = {position_key(board): 1}
        decisions: list[AnalyticMachineDecision] = []
        trace: list[TracePacket] = []
        illegal_commit_count = 0
        terminal = self.rules.terminal_status_from_prompt(prompt, 0, repetitions[position_key(board)])
        for ply in range(max_plies):
            terminal = self.rules.terminal_status_from_prompt(prompt, ply, repetitions[position_key(board)])
            if terminal.is_terminal:
                trace.extend(self.rules.terminal_trace_from_prompt(prompt, ply, repetitions[position_key(board)]))
                break
            decision = self.select_move_from_prompt(prompt, rng.randrange(0, 2**31 - 1), temperature)
            if decision.chosen_move.to_uci() not in {move.to_uci() for move in decision.legal_moves}:
                illegal_commit_count += 1
            move_trace = self.rules.make_move_trace_from_prompt(prompt, decision.chosen_move, ply, include_terminal=False)
            board = self.rules.board_after_move_from_prompt(prompt, decision.chosen_move)
            prompt = self.rules.prompt_trace_from_board(board)
            repetitions[position_key(board)] = repetitions.get(position_key(board), 0) + 1
            decisions.append(decision)
            trace.extend(decision.trace)
            trace.extend(move_trace)
        else:
            terminal = self.rules.terminal_status_from_prompt(
                prompt,
                max_plies,
                repetitions[position_key(board)],
                adjudication_cap_reached=True,
            )
            trace.extend(
                self.rules.terminal_trace_from_prompt(
                    prompt,
                    max_plies,
                    repetitions[position_key(board)],
                    adjudication_cap_reached=True,
                )
            )
        return AnalyticGameRecord(
            start_fen=fen,
            final_board=board,
            terminal_status=terminal,
            decisions=tuple(decisions),
            trace=tuple(trace),
            illegal_commit_count=illegal_commit_count,
        )


def _sample_rank(probabilities: torch.Tensor, seed: int) -> int:
    threshold = random.Random(seed).random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities.tolist()):
        cumulative += probability
        if threshold <= cumulative or math.isclose(cumulative, 1.0):
            return index
    return len(probabilities) - 1


def _score_bucket(score: float) -> int:
    return max(0, min(2**31 - 1, int(round((score + 32.0) * 1024.0))))
