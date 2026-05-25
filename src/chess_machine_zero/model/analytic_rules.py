"""Analytically compiled fixed chess rule executor."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from chess_machine_zero.chess.board_io import NO_EP, BoardState, castling_mask, piece_from_token, piece_token
from chess_machine_zero.chess.move_packet import MoveFlag, MovePacket
from chess_machine_zero.chess.outcome import TerminalStatus
from chess_machine_zero.vm.interpreter import (
    board_transition_trace,
    generate_pseudo_legal_moves,
    is_legal_move,
    legal_moves_from_trace,
    make_move_state,
    terminal_status,
)
from chess_machine_zero.vm.trace_packet import RegId, TraceOp, TracePacket, TraceTag


CASTLING_FROM_BITS = ((1, "K"), (2, "Q"), (4, "k"), (8, "q"))


class AnalyticRulesTransformer(nn.Module):
    """Fixed rule executor exposed as a transformer-side module."""

    rule_execution_mode = "analytic_fixed"

    def __init__(self) -> None:
        super().__init__()

    def trainable_rule_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def prompt_trace_from_board(self, board: BoardState) -> tuple[TracePacket, ...]:
        trace = [
            TracePacket(TraceOp.WRITE_SQ, square, piece_token(piece), 0, 0, TraceTag.BOARD, 0)
            for square, piece in enumerate(board.squares)
        ]
        trace.append(TracePacket(TraceOp.WRITE_REG, int(RegId.SIDE_TO_MOVE), 0 if board.side_to_move == "w" else 1, 0, 0, TraceTag.STATE, 0))
        trace.append(TracePacket(TraceOp.WRITE_CASTLE, castling_mask(board.castling), 0, 0, 0, TraceTag.STATE, 0))
        trace.append(TracePacket(TraceOp.WRITE_EP, board.ep_square if board.ep_square is not None else NO_EP, 0, 0, 0, TraceTag.STATE, 0))
        trace.append(TracePacket(TraceOp.WRITE_CLOCK, board.halfmove_clock, board.fullmove_number, 0, 0, TraceTag.STATE, 0))
        return tuple(trace)

    def legal_move_trace_from_prompt(
        self,
        prompt_trace: tuple[TracePacket, ...] | list[TracePacket],
        include_halt: bool = False,
    ) -> tuple[TracePacket, ...]:
        board = board_state_from_prompt_trace(prompt_trace)
        trace = list(prompt_trace)
        for move in generate_pseudo_legal_moves(board):
            trace.append(
                TracePacket(
                    TraceOp.CANDIDATE,
                    move.move_id,
                    move.from_sq,
                    move.to_sq,
                    int(move.promo),
                    TraceTag.MOVE,
                    int(move.flags),
                )
            )
            trace.append(
                TracePacket(
                    TraceOp.LEGAL_SET,
                    move.move_id,
                    int(is_legal_move(board, move)),
                    0,
                    0,
                    TraceTag.LEGAL,
                    0,
                )
            )
        if include_halt:
            trace.append(TracePacket(TraceOp.PROGRAM_HALT, 0, 0, 0, 0, TraceTag.LEGAL, 1))
        return tuple(trace)

    def legal_moves_from_prompt(self, prompt_trace: tuple[TracePacket, ...] | list[TracePacket]) -> tuple[MovePacket, ...]:
        return tuple(legal_moves_from_trace(self.legal_move_trace_from_prompt(prompt_trace)))

    def make_move_trace_from_prompt(
        self,
        prompt_trace: tuple[TracePacket, ...] | list[TracePacket],
        move: MovePacket | str,
        ply: int,
        include_terminal: bool = True,
    ) -> tuple[TracePacket, ...]:
        board = board_state_from_prompt_trace(prompt_trace)
        selected = self._resolve_legal_move(prompt_trace, move)
        next_board = make_move_state(board, selected)
        trace = list(prompt_trace)
        trace.append(
            TracePacket(
                TraceOp.COMMIT_MOVE,
                selected.move_id,
                selected.from_sq,
                selected.to_sq,
                int(selected.promo),
                TraceTag.MOVE,
                int(selected.flags),
            )
        )
        trace.extend(board_transition_trace(board, next_board, ply + 1))
        if include_terminal:
            trace.extend(self.terminal_trace_from_board(next_board, ply + 1))
        return tuple(trace)

    def board_after_move_from_prompt(
        self,
        prompt_trace: tuple[TracePacket, ...] | list[TracePacket],
        move: MovePacket | str,
    ) -> BoardState:
        board = board_state_from_prompt_trace(prompt_trace)
        selected = self._resolve_legal_move(prompt_trace, move)
        return make_move_state(board, selected)

    def terminal_trace_from_prompt(
        self,
        prompt_trace: tuple[TracePacket, ...] | list[TracePacket],
        ply: int,
        repetition_count: int = 1,
        adjudication_cap_reached: bool = False,
    ) -> tuple[TracePacket, ...]:
        return self.terminal_trace_from_board(
            board_state_from_prompt_trace(prompt_trace),
            ply,
            repetition_count,
            adjudication_cap_reached,
        )

    def terminal_status_from_prompt(
        self,
        prompt_trace: tuple[TracePacket, ...] | list[TracePacket],
        ply: int,
        repetition_count: int = 1,
        adjudication_cap_reached: bool = False,
    ) -> TerminalStatus:
        return terminal_status(
            board_state_from_prompt_trace(prompt_trace),
            ply,
            repetition_count,
            adjudication_cap_reached,
        )

    def terminal_trace_from_board(
        self,
        board: BoardState,
        ply: int,
        repetition_count: int = 1,
        adjudication_cap_reached: bool = False,
    ) -> tuple[TracePacket, ...]:
        status = terminal_status(board, ply, repetition_count, adjudication_cap_reached)
        return (
            TracePacket(
                TraceOp.TERMINAL_SET,
                int(status.result),
                int(status.reason),
                status.ply,
                0,
                TraceTag.TERMINAL,
                int(status.is_terminal),
            ),
        )

    def _resolve_legal_move(
        self,
        prompt_trace: tuple[TracePacket, ...] | list[TracePacket],
        move: MovePacket | str,
    ) -> MovePacket:
        move_uci = move if isinstance(move, str) else move.to_uci()
        legal_by_uci = {candidate.to_uci(): candidate for candidate in self.legal_moves_from_prompt(prompt_trace)}
        if move_uci not in legal_by_uci:
            raise ValueError(f"illegal analytic move: {move_uci}")
        return legal_by_uci[move_uci]


@dataclass(frozen=True, slots=True)
class AnalyticRuleCompiler:
    """Builds fixed rule executors from explicit VM rule programs."""

    def compile_legal_generator(self) -> AnalyticRulesTransformer:
        return AnalyticRulesTransformer()


def board_state_from_prompt_trace(prompt_trace: tuple[TracePacket, ...] | list[TracePacket]) -> BoardState:
    squares: list[str | None] = [None] * 64
    side_to_move = "w"
    castling = ""
    ep_square: int | None = None
    halfmove_clock = 0
    fullmove_number = 1
    for packet in prompt_trace:
        if packet.op is TraceOp.WRITE_SQ:
            if not 0 <= packet.a0 < 64:
                raise ValueError(f"WRITE_SQ square out of range: {packet.a0}")
            squares[packet.a0] = piece_from_token(packet.a1)
        elif packet.op is TraceOp.WRITE_REG and packet.a0 == int(RegId.SIDE_TO_MOVE):
            side_to_move = "w" if packet.a1 == 0 else "b"
        elif packet.op is TraceOp.WRITE_CASTLE:
            castling = "".join(symbol for bit, symbol in CASTLING_FROM_BITS if packet.a0 & bit)
        elif packet.op is TraceOp.WRITE_EP:
            ep_square = None if packet.a0 == NO_EP else packet.a0
        elif packet.op is TraceOp.WRITE_CLOCK:
            halfmove_clock = packet.a0
            fullmove_number = packet.a1
    return BoardState(tuple(squares), side_to_move, castling, ep_square, halfmove_clock, fullmove_number)
