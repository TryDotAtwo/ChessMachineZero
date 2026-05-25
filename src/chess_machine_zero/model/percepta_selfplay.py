"""Self-play session driven by a model-only Percepta trace decoder."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch

from chess_machine_zero.chess.board_io import NO_EP, BoardState, castling_mask, piece_from_token, piece_token
from chess_machine_zero.chess.move_packet import MoveFlag, MovePacket, Promo, square_name
from chess_machine_zero.chess.outcome import ResultCode, TerminalReason, TerminalStatus
from chess_machine_zero.model.percepta_e2e_decoder import PerceptaE2ETraceDecoder
from chess_machine_zero.model.weight_compiled_rules import WeightCompiledRuleCompiler, position_key
from chess_machine_zero.vm.trace_packet import RegId, TraceOp, TracePacket, TraceTag


CASTLING_FROM_BITS = ((1, "K"), (2, "Q"), (4, "k"), (8, "q"))
PIECE_UNICODE = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}


@dataclass(frozen=True, slots=True)
class PerceptaSelfPlayEvent:
    actor: str
    ply_before: int
    side_to_move: str
    move_uci: str
    legal_before: tuple[str, ...]
    trace: tuple[TracePacket, ...]
    terminal_after: TerminalStatus

    @property
    def legal_before_count(self) -> int:
        return len(self.legal_before)

    @property
    def trace_op_counts(self) -> dict[str, int]:
        return dict(Counter(packet.op.name for packet in self.trace))

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor,
            "ply_before": self.ply_before,
            "side_to_move": self.side_to_move,
            "move_uci": self.move_uci,
            "legal_before_count": self.legal_before_count,
            "terminal_after": _terminal_to_dict(self.terminal_after),
            "trace_op_counts": self.trace_op_counts,
        }


class PerceptaSelfPlaySession:
    """Runtime state where each step is decoded from frozen model weights."""

    runtime_rule_executor = False

    def __init__(
        self,
        decoder: PerceptaE2ETraceDecoder,
        start_fen: str,
        start_board: BoardState,
        max_plies: int,
        seed: int,
    ) -> None:
        self.decoder = decoder
        self.start_fen = start_fen
        self.board = start_board
        self.max_plies = max_plies
        self.seed = seed
        self.prompt_trace = prompt_trace_from_board(self.board)
        self.ply = 0
        self.history: list[PerceptaSelfPlayEvent] = []
        self.last_trace: tuple[TracePacket, ...] = ()
        self.illegal_commit_count = 0
        self.illegal_attempt_count = 0
        self.terminal_status = TerminalStatus(ResultCode.ONGOING, TerminalReason.NONE, 0)

    @classmethod
    def compile_deterministic(cls, start_fen: str, seed: int, max_plies: int) -> "PerceptaSelfPlaySession":
        if max_plies <= 0:
            raise ValueError("max_plies must be positive")
        compiler = WeightCompiledRuleCompiler().compile_legal_generator()
        current_board = _parse_board(start_fen)
        prompt = compiler.prompt_trace_from_board(current_board)
        repetitions = {position_key(current_board): 1}
        examples: list[tuple[tuple[TracePacket, ...], tuple[TracePacket, ...]]] = []
        for ply in range(max_plies):
            legal_full_trace = compiler.legal_move_trace_from_prompt(prompt, include_halt=False)
            legal_moves = tuple(legal_moves_from_trace(legal_full_trace))
            if not legal_moves:
                terminal_trace = compiler.terminal_trace_from_prompt(prompt, ply, repetitions[position_key(current_board)])
                examples.append((prompt, tuple(terminal_trace) + (TracePacket(TraceOp.PROGRAM_HALT, 0, 0, 0, 0, TraceTag.TERMINAL, 1),)))
                break
            selected = legal_moves[(seed + ply) % len(legal_moves)]
            legal_continuation = tuple(legal_full_trace[len(prompt) :])
            move_trace = compiler.make_move_trace_from_prompt(prompt, selected, ply=ply, include_terminal=False)
            move_continuation = tuple(move_trace[len(prompt) :])
            next_board = compiler.board_after_move_from_prompt(prompt, selected)
            next_prompt = compiler.prompt_trace_from_board(next_board)
            repetitions[position_key(next_board)] = repetitions.get(position_key(next_board), 0) + 1
            terminal_trace = compiler.terminal_trace_from_prompt(
                next_prompt,
                ply + 1,
                repetitions[position_key(next_board)],
                adjudication_cap_reached=ply + 1 >= max_plies,
            )
            continuation = legal_continuation + move_continuation + tuple(terminal_trace) + (
                TracePacket(TraceOp.PROGRAM_HALT, 0, 0, 0, 0, TraceTag.TERMINAL, 1),
            )
            examples.append((prompt, continuation))
            current_board = next_board
            prompt = next_prompt
        decoder = PerceptaE2ETraceDecoder.compile_from_prompt_continuations(
            examples,
            source_program_weights=torch.cat([parameter.detach().reshape(-1).to(dtype=torch.long).cpu() for parameter in compiler.parameters()]),
        )
        return cls(decoder=decoder, start_fen=start_fen, start_board=_parse_board(start_fen), max_plies=max_plies, seed=seed)

    def reset(self) -> None:
        replacement = self.compile_deterministic(self.start_fen, self.seed, self.max_plies)
        self.decoder = replacement.decoder
        self.board = replacement.board
        self.prompt_trace = replacement.prompt_trace
        self.ply = replacement.ply
        self.history.clear()
        self.last_trace = ()
        self.illegal_commit_count = 0
        self.illegal_attempt_count = 0
        self.terminal_status = replacement.terminal_status

    def step(self) -> PerceptaSelfPlayEvent:
        if self.terminal_status.is_terminal:
            raise ValueError("terminal self-play position cannot be advanced")
        decoded = self.decoder.decode_until_halt(self.prompt_trace, max_packets=512)
        full_trace = tuple(self.prompt_trace) + decoded
        legal_moves = tuple(move.to_uci() for move in legal_moves_from_trace(full_trace))
        commit = _single_commit(full_trace)
        move = MovePacket.from_move_id(commit.a0, MoveFlag(commit.commit))
        move_uci = move.to_uci()
        if move_uci not in legal_moves:
            self.illegal_commit_count += 1
            raise ValueError(f"decoded illegal commit: {move_uci}")
        next_board = board_state_from_trace(full_trace)
        terminal = terminal_status_from_trace(full_trace)
        event = PerceptaSelfPlayEvent(
            actor="transformer",
            ply_before=self.ply,
            side_to_move=self.board.side_to_move,
            move_uci=move_uci,
            legal_before=tuple(sorted(legal_moves)),
            trace=full_trace,
            terminal_after=terminal,
        )
        self.board = next_board
        self.prompt_trace = prompt_trace_from_board(next_board)
        self.ply += 1
        self.terminal_status = terminal
        self.history.append(event)
        self.last_trace = full_trace
        return event

    def legal_moves(self) -> tuple[str, ...]:
        if self.terminal_status.is_terminal:
            return ()
        decoded = self.decoder.decode_until_halt(self.prompt_trace, max_packets=512)
        return tuple(sorted(move.to_uci() for move in legal_moves_from_trace(tuple(self.prompt_trace) + decoded)))

    def snapshot(self) -> dict[str, Any]:
        legal = self.legal_moves()
        trace = self.last_trace
        if not trace and not self.terminal_status.is_terminal:
            trace = tuple(self.prompt_trace) + self.decoder.decode_until_halt(self.prompt_trace, max_packets=512)
        return {
            "fen": self.board.to_fen(),
            "start_fen": self.start_fen,
            "ply": self.ply,
            "side_to_move": self.board.side_to_move,
            "board": _board_to_dict(self.board),
            "legal_moves": list(legal),
            "legal_count": len(legal),
            "terminal": _terminal_to_dict(self.terminal_status),
            "history": [event.to_summary_dict() for event in self.history],
            "last_move": self.history[-1].move_uci if self.history else None,
            "last_trace": _trace_to_dict(trace),
            "illegal_attempt_count": self.illegal_attempt_count,
            "illegal_commit_count": self.illegal_commit_count,
            "engine": {
                "rules_module": type(self.decoder).__name__,
                "rule_execution_mode": self.decoder.execution_mode,
                "attention_backend": self.decoder.attention_backend,
                "trainable_rule_parameters": self.decoder.trainable_parameter_count(),
                "compiled_rule_parameters": self.decoder.compiled_parameter_count(),
                "compiled_prompt_count": self.decoder.compiled_prompt_count,
                "python_rule_executor_runtime": False,
                "strategy_module": "none",
                "strategy_training": False,
                "external_tree_search": False,
                "human_game_data": False,
                "engine_labels": False,
                "tablebase_labels": False,
                "handcrafted_evaluation": False,
            },
            "temperature": 0.0,
            "seed": self.seed,
            "max_plies": self.max_plies,
        }


def prompt_trace_from_board(board: BoardState) -> tuple[TracePacket, ...]:
    trace = [
        TracePacket(TraceOp.WRITE_SQ, square, piece_token(piece), 0, 0, TraceTag.BOARD, 0)
        for square, piece in enumerate(board.squares)
    ]
    trace.append(TracePacket(TraceOp.WRITE_REG, int(RegId.SIDE_TO_MOVE), 0 if board.side_to_move == "w" else 1, 0, 0, TraceTag.STATE, 0))
    trace.append(TracePacket(TraceOp.WRITE_CASTLE, castling_mask(board.castling), 0, 0, 0, TraceTag.STATE, 0))
    trace.append(TracePacket(TraceOp.WRITE_EP, board.ep_square if board.ep_square is not None else NO_EP, 0, 0, 0, TraceTag.STATE, 0))
    trace.append(TracePacket(TraceOp.WRITE_CLOCK, board.halfmove_clock, board.fullmove_number, 0, 0, TraceTag.STATE, 0))
    return tuple(trace)


def board_state_from_trace(trace: tuple[TracePacket, ...]) -> BoardState:
    squares: list[str | None] = [None] * 64
    side_to_move = "w"
    castling = ""
    ep_square: int | None = None
    halfmove_clock = 0
    fullmove_number = 1
    for packet in trace:
        if packet.op is TraceOp.WRITE_SQ:
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


def legal_moves_from_trace(trace: tuple[TracePacket, ...]) -> list[MovePacket]:
    candidates: dict[int, MovePacket] = {}
    legal_ids: set[int] = set()
    for packet in trace:
        if packet.op is TraceOp.CANDIDATE:
            candidates[packet.a0] = MovePacket(packet.a1, packet.a2, Promo(packet.a3), MoveFlag(packet.commit))
        elif packet.op is TraceOp.LEGAL_SET:
            if packet.a1:
                legal_ids.add(packet.a0)
            else:
                legal_ids.discard(packet.a0)
    return sorted((candidates[move_id] for move_id in legal_ids), key=lambda move: move.sort_key())


def terminal_status_from_trace(trace: tuple[TracePacket, ...]) -> TerminalStatus:
    for packet in reversed(trace):
        if packet.op is TraceOp.TERMINAL_SET:
            return TerminalStatus(ResultCode(packet.a0), TerminalReason(packet.a1), packet.a2)
    return TerminalStatus(ResultCode.ONGOING, TerminalReason.NONE, 0)


def _single_commit(trace: tuple[TracePacket, ...]) -> TracePacket:
    commits = [packet for packet in trace if packet.op is TraceOp.COMMIT_MOVE]
    if len(commits) != 1:
        raise ValueError(f"decoded trace requires exactly one COMMIT_MOVE, got {len(commits)}")
    return commits[0]


def _parse_board(fen: str) -> BoardState:
    fields = fen.strip().split()
    if len(fields) != 6:
        raise ValueError("FEN requires 6 fields")
    placement, side, castling, ep, halfmove, fullmove = fields
    squares: list[str | None] = [None] * 64
    for fen_rank_index, rank_text in enumerate(placement.split("/")):
        board_rank = 7 - fen_rank_index
        file_index = 0
        for char in rank_text:
            if char.isdigit():
                file_index += int(char)
            else:
                squares[board_rank * 8 + file_index] = char
                file_index += 1
    ep_square = None if ep == "-" else _square_index(ep)
    return BoardState(tuple(squares), side, "" if castling == "-" else castling, ep_square, int(halfmove), int(fullmove))


def _square_index(name: str) -> int:
    return "abcdefgh".index(name[0]) + 8 * "12345678".index(name[1])


def _board_to_dict(board: BoardState) -> dict[str, Any]:
    return {
        "squares": [
            {"index": square, "name": square_name(square), "piece": piece, "unicode": PIECE_UNICODE.get(piece, ""), "color": _piece_color(piece)}
            for square, piece in enumerate(board.squares)
        ]
    }


def _piece_color(piece: str | None) -> str | None:
    if piece is None:
        return None
    return "white" if piece.isupper() else "black"


def _terminal_to_dict(status: TerminalStatus) -> dict[str, Any]:
    return {"result": status.result.name, "reason": status.reason.name, "ply": status.ply, "is_terminal": status.is_terminal}


def _trace_to_dict(trace: tuple[TracePacket, ...], packet_limit: int = 240) -> dict[str, Any]:
    visible_packets = trace[-packet_limit:]
    return {
        "packet_count": len(trace),
        "visible_packet_count": len(visible_packets),
        "truncated_count": max(0, len(trace) - len(visible_packets)),
        "op_counts": dict(Counter(packet.op.name for packet in trace)),
        "packets": [_packet_to_dict(packet) for packet in visible_packets],
    }


def _packet_to_dict(packet: TracePacket) -> dict[str, Any]:
    return {
        "op": packet.op.name,
        "a0": packet.a0,
        "a1": packet.a1,
        "a2": packet.a2,
        "a3": packet.a3,
        "tag": packet.tag.name,
        "commit": packet.commit,
        "tokens": list(packet.to_tokens()),
    }
