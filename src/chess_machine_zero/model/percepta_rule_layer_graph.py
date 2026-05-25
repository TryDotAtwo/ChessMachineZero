"""Executable rule layer graph backed by frozen tensor kernels."""

from __future__ import annotations

from collections import Counter

import torch

from chess_machine_zero.chess.board_io import BoardState
from chess_machine_zero.chess.move_packet import MoveFlag, MovePacket, Promo
from chess_machine_zero.chess.outcome import ResultCode, TerminalReason, TerminalStatus
from chess_machine_zero.model.percepta_attention_rule_kernels import (
    FrozenAttentionTensorRuleKernels,
    TensorBoardState,
    TensorCandidateSet,
)
from chess_machine_zero.model.weight_compiled_rules import board_state_from_prompt_trace, board_transition_trace
from chess_machine_zero.vm.trace_packet import TraceOp, TracePacket, TraceTag


RULE_PRIMITIVE_NAMES = (
    "PIECE_DISPATCH",
    "RAY_SCAN",
    "ATTACK_TEST",
    "LEGAL_FILTER",
    "MAKE_MOVE",
    "TERMINAL_PREDICATES",
)


class FrozenAttentionRuleLayerGraph:
    """Layer graph that routes chess rule execution through tensor kernels."""

    def __init__(self, rule_kernels: FrozenAttentionTensorRuleKernels) -> None:
        self.rule_kernels = rule_kernels
        self.execution_counts: Counter[str] = Counter({name: 0 for name in RULE_PRIMITIVE_NAMES})

    def legal_move_trace_from_prompt(
        self,
        prompt_trace: tuple[TracePacket, ...] | list[TracePacket],
        include_halt: bool = False,
    ) -> tuple[TracePacket, ...]:
        board = self.rule_kernels.board_to_tensor(board_state_from_prompt_trace(prompt_trace))
        candidates = self._legal_candidate_tensors(board)
        trace = list(prompt_trace)
        for move, legal in self._candidate_rows(candidates):
            trace.append(TracePacket(TraceOp.CANDIDATE, move.move_id, move.from_sq, move.to_sq, int(move.promo), TraceTag.MOVE, int(move.flags)))
            trace.append(TracePacket(TraceOp.LEGAL_SET, move.move_id, int(legal), 0, 0, TraceTag.LEGAL, 0))
        if include_halt:
            trace.append(TracePacket(TraceOp.PROGRAM_HALT, 0, 0, 0, 0, TraceTag.LEGAL, 1))
        return tuple(trace)

    def legal_moves_from_prompt(self, prompt_trace: tuple[TracePacket, ...] | list[TracePacket]) -> tuple[MovePacket, ...]:
        board = self.rule_kernels.board_to_tensor(board_state_from_prompt_trace(prompt_trace))
        candidates = self._legal_candidate_tensors(board)
        return tuple(move for move, legal in self._candidate_rows(candidates) if legal)

    def make_move_trace_from_prompt(
        self,
        prompt_trace: tuple[TracePacket, ...] | list[TracePacket],
        move: MovePacket | str,
        ply: int,
        include_terminal: bool = True,
    ) -> tuple[TracePacket, ...]:
        board_state = board_state_from_prompt_trace(prompt_trace)
        board = self.rule_kernels.board_to_tensor(board_state)
        selected = self._resolve_legal_move(board, move)
        next_board = self._make_move_state(board, selected)
        trace = list(prompt_trace)
        trace.append(TracePacket(TraceOp.COMMIT_MOVE, selected.move_id, selected.from_sq, selected.to_sq, int(selected.promo), TraceTag.MOVE, int(selected.flags)))
        trace.extend(board_transition_trace(board_state, next_board, ply + 1))
        if include_terminal:
            trace.extend(self.terminal_trace_from_board(next_board, ply + 1))
        return tuple(trace)

    def board_after_move_from_prompt(
        self,
        prompt_trace: tuple[TracePacket, ...] | list[TracePacket],
        move: MovePacket | str,
    ) -> BoardState:
        board = self.rule_kernels.board_to_tensor(board_state_from_prompt_trace(prompt_trace))
        selected = self._resolve_legal_move(board, move)
        return self._make_move_state(board, selected)

    def terminal_trace_from_prompt(
        self,
        prompt_trace: tuple[TracePacket, ...] | list[TracePacket],
        ply: int,
        repetition_count: int = 1,
        adjudication_cap_reached: bool = False,
    ) -> tuple[TracePacket, ...]:
        board = board_state_from_prompt_trace(prompt_trace)
        return self.terminal_trace_from_board(board, ply, repetition_count, adjudication_cap_reached)

    def terminal_status_from_prompt(
        self,
        prompt_trace: tuple[TracePacket, ...] | list[TracePacket],
        ply: int,
        repetition_count: int = 1,
        adjudication_cap_reached: bool = False,
    ) -> TerminalStatus:
        return self.terminal_status(board_state_from_prompt_trace(prompt_trace), ply, repetition_count, adjudication_cap_reached)

    def terminal_trace_from_board(
        self,
        board: BoardState,
        ply: int,
        repetition_count: int = 1,
        adjudication_cap_reached: bool = False,
    ) -> tuple[TracePacket, ...]:
        status = self.terminal_status(board, ply, repetition_count, adjudication_cap_reached)
        return (
            TracePacket(TraceOp.TERMINAL_SET, int(status.result), int(status.reason), status.ply, 0, TraceTag.TERMINAL, int(status.is_terminal)),
        )

    def generate_pseudo_legal_moves(self, board: BoardState) -> list[MovePacket]:
        candidates = self._legal_candidate_tensors(self.rule_kernels.board_to_tensor(board))
        return [move for move, _legal in self._candidate_rows(candidates)]

    def is_legal_move(self, board: BoardState, move: MovePacket) -> bool:
        move_uci = move.to_uci()
        legal_by_uci = {candidate.to_uci(): legal for candidate, legal in self._candidate_rows(self._legal_candidate_tensors(self.rule_kernels.board_to_tensor(board)))}
        return bool(legal_by_uci.get(move_uci, False))

    def make_move_state(self, board: BoardState, move: MovePacket) -> BoardState:
        return self._make_move_state(self.rule_kernels.board_to_tensor(board), move)

    def terminal_status(
        self,
        board: BoardState,
        ply: int = 0,
        repetition_count: int = 1,
        adjudication_cap_reached: bool = False,
    ) -> TerminalStatus:
        self._mark("TERMINAL_PREDICATES")
        tensor_board = self.rule_kernels.board_to_tensor(board)
        terminal = self.rule_kernels.terminal_predicates(
            tensor_board,
            torch.tensor(ply, dtype=torch.long),
            torch.tensor(repetition_count, dtype=torch.long),
            torch.tensor(adjudication_cap_reached, dtype=torch.bool),
        )
        return TerminalStatus(ResultCode(int(terminal.result.item())), TerminalReason(int(terminal.reason.item())), ply)

    def _legal_candidate_tensors(self, board: TensorBoardState) -> TensorCandidateSet:
        self._mark("PIECE_DISPATCH")
        self._mark("RAY_SCAN")
        self._mark("ATTACK_TEST")
        self._mark("LEGAL_FILTER")
        return self.rule_kernels.legal_candidate_tensors(board)

    def _make_move_state(self, board: TensorBoardState, move: MovePacket) -> BoardState:
        self._mark("MAKE_MOVE")
        next_board = self.rule_kernels.make_move(
            board,
            torch.tensor([move.from_sq, move.to_sq, int(move.promo), int(move.flags)], dtype=torch.long),
        )
        return self.rule_kernels.board_from_tensor(
            next_board.squares,
            next_board.side_id,
            next_board.castle_mask,
            next_board.ep_square,
            next_board.halfmove_clock,
            next_board.fullmove_number,
        )

    def _resolve_legal_move(self, board: TensorBoardState, move: MovePacket | str) -> MovePacket:
        move_uci = move if isinstance(move, str) else move.to_uci()
        legal_by_uci = {candidate.to_uci(): candidate for candidate, legal in self._candidate_rows(self._legal_candidate_tensors(board)) if legal}
        if move_uci not in legal_by_uci:
            raise ValueError(f"illegal frozen-attention move: {move_uci}")
        return legal_by_uci[move_uci]

    def _candidate_rows(self, candidates: TensorCandidateSet) -> list[tuple[MovePacket, bool]]:
        rows: list[tuple[MovePacket, bool]] = []
        for from_sq, to_sq, promo, flags, legal in zip(
            candidates.from_sq.tolist(),
            candidates.to_sq.tolist(),
            candidates.promo.tolist(),
            candidates.flags.tolist(),
            candidates.legal.tolist(),
            strict=True,
        ):
            rows.append((MovePacket(int(from_sq), int(to_sq), Promo(int(promo)), MoveFlag(int(flags))), bool(legal)))
        return rows

    def _mark(self, primitive_name: str) -> None:
        self.execution_counts[primitive_name] += 1
