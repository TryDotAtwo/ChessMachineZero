"""Self-play session driven by parametric rule weights over arbitrary boards."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from chess_machine_zero.chess.board_io import NO_EP, BoardState, castling_mask, parse_fen, piece_from_token, piece_token
from chess_machine_zero.chess.move_packet import MoveFlag, MovePacket, square_name
from chess_machine_zero.chess.outcome import ResultCode, TerminalReason, TerminalStatus
from chess_machine_zero.model.percepta_frozen_attention_vm import PerceptaFrozenAttentionRuleCompiler, PerceptaFrozenAttentionRuleComputer
from chess_machine_zero.model.weight_compiled_rules import board_state_from_prompt_trace, legal_moves_from_trace, position_key
from chess_machine_zero.vm.trace_packet import RegId, TraceOp, TracePacket, TraceTag


CASTLING_FROM_BITS = ((1, "K"), (2, "Q"), (4, "k"), (8, "q"))
PIECE_UNICODE = {
    "P": "\u2659",
    "N": "\u2658",
    "B": "\u2657",
    "R": "\u2656",
    "Q": "\u2655",
    "K": "\u2654",
    "p": "\u265f",
    "n": "\u265e",
    "b": "\u265d",
    "r": "\u265c",
    "q": "\u265b",
    "k": "\u265a",
}


@dataclass(frozen=True, slots=True)
class PerceptaParametricMoveEvent:
    actor: str
    transformer_id: str
    ply_before: int
    side_to_move: str
    move_uci: str
    legal_before: tuple[str, ...]
    trace: tuple[TracePacket, ...]
    terminal_after: TerminalStatus
    trace_verified_legal: bool

    @property
    def legal_before_count(self) -> int:
        return len(self.legal_before)

    @property
    def trace_op_counts(self) -> dict[str, int]:
        return dict(Counter(packet.op.name for packet in self.trace))

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor,
            "transformer_id": self.transformer_id,
            "ply_before": self.ply_before,
            "side_to_move": self.side_to_move,
            "move_uci": self.move_uci,
            "legal_before_count": self.legal_before_count,
            "terminal_after": _terminal_to_dict(self.terminal_after),
            "trace_op_counts": self.trace_op_counts,
            "trace_verified_legal": self.trace_verified_legal,
            "emitted_token_count": len(self.trace),
            "emitted_tokens": [list(packet.to_tokens()) for packet in self.trace],
        }


class PerceptaParametricSelfPlaySession:
    """Runtime session with parametric rule weights and no finite board lookup."""

    runtime_rule_executor = False

    def __init__(
        self,
        rules: PerceptaFrozenAttentionRuleComputer,
        start_fen: str,
        max_plies: int,
        seed: int,
        black_rules: PerceptaFrozenAttentionRuleComputer | None = None,
    ) -> None:
        if max_plies <= 0:
            raise ValueError("max_plies must be positive")
        self.white_rules = rules
        self.black_rules = black_rules or PerceptaFrozenAttentionRuleCompiler().compile_trace_computer()
        self.rules = self.white_rules
        self.start_fen = start_fen
        self.board = parse_fen(start_fen)
        self.max_plies = int(max_plies)
        self.seed = int(seed)
        self.prompt_tensor_trace = self.white_rules.prompt_tensor_from_board(self.board)
        self.prompt_trace = self.white_rules.tensor_trace_to_packets(self.prompt_tensor_trace)
        self.repetitions = {position_key(self.board): 1}
        self.ply = 0
        self.history: list[PerceptaParametricMoveEvent] = []
        self.last_tensor_trace = None
        self.last_trace: tuple[TracePacket, ...] = ()
        self.last_transformer_traces: dict[str, tuple[TracePacket, ...]] = {"white": (), "black": ()}
        self.last_trace_verified_legal: bool | None = None
        self.illegal_commit_count = 0
        self.illegal_attempt_count = 0
        self.terminal_status = self.white_rules.terminal_status_from_prompt(
            self.prompt_trace,
            self.ply,
            self.repetitions[position_key(self.board)],
        )

    @classmethod
    def create(cls, start_fen: str, seed: int, max_plies: int) -> "PerceptaParametricSelfPlaySession":
        rules = PerceptaFrozenAttentionRuleCompiler().compile_trace_computer()
        return cls(rules=rules, start_fen=start_fen, max_plies=max_plies, seed=seed)

    def reset(self) -> None:
        replacement = self.create(self.start_fen, self.seed, self.max_plies)
        self.white_rules = replacement.white_rules
        self.black_rules = replacement.black_rules
        self.rules = replacement.rules
        self.board = replacement.board
        self.prompt_tensor_trace = replacement.prompt_tensor_trace
        self.prompt_trace = replacement.prompt_trace
        self.repetitions = replacement.repetitions
        self.ply = replacement.ply
        self.history.clear()
        self.last_tensor_trace = None
        self.last_trace = ()
        self.last_transformer_traces = {"white": (), "black": ()}
        self.last_trace_verified_legal = None
        self.illegal_commit_count = 0
        self.illegal_attempt_count = 0
        self.terminal_status = replacement.terminal_status

    def legal_moves(self) -> tuple[str, ...]:
        if self.terminal_status.is_terminal:
            return ()
        rules = self._active_rules()
        legal_tensor_trace = rules.decode_legal_tensor_trace_host_append_only(self.prompt_tensor_trace, max_packets=512)
        legal_trace = rules.tensor_trace_to_packets(legal_tensor_trace)
        return tuple(sorted(move.to_uci() for move in legal_moves_from_trace(legal_trace)))

    def step(self) -> PerceptaParametricMoveEvent:
        if self.terminal_status.is_terminal:
            raise ValueError("terminal self-play position cannot be advanced")
        side = self.board.side_to_move
        rules = self._rules_for_side(side)
        legal_tensor_trace = rules.decode_legal_tensor_trace_host_append_only(self.prompt_tensor_trace, max_packets=512)
        legal_trace = rules.tensor_trace_to_packets(legal_tensor_trace)
        legal_moves = tuple(legal_moves_from_trace(legal_trace))
        if not legal_moves:
            self.terminal_status = rules.terminal_status_from_prompt(
                self.prompt_trace,
                self.ply,
                self.repetitions[position_key(self.board)],
            )
            raise ValueError("terminal position has no legal move")
        selected = legal_moves[(self.seed + self.ply) % len(legal_moves)]
        return self._commit_selected_move(selected, legal_trace, actor=self._transformer_id_for_side(side), rules=rules)

    def play_human_move(self, move_uci: str) -> PerceptaParametricMoveEvent:
        if self.terminal_status.is_terminal:
            self.illegal_attempt_count += 1
            raise ValueError("terminal position cannot accept a human move")
        rules = self._active_rules()
        legal_tensor_trace = rules.decode_legal_tensor_trace_host_append_only(self.prompt_tensor_trace, max_packets=512)
        legal_trace = rules.tensor_trace_to_packets(legal_tensor_trace)
        legal_by_uci = {move.to_uci(): move for move in legal_moves_from_trace(legal_trace)}
        if move_uci not in legal_by_uci:
            self.illegal_attempt_count += 1
            raise ValueError(f"illegal move: {move_uci}")
        return self._commit_selected_move(legal_by_uci[move_uci], legal_trace, actor="human", rules=rules)

    def snapshot(self) -> dict[str, Any]:
        rules = self._active_rules()
        legal = self.legal_moves()
        trace = self.last_trace
        if not trace and not self.terminal_status.is_terminal:
            trace = rules.tensor_trace_to_packets(rules.decode_legal_tensor_trace_host_append_only(self.prompt_tensor_trace, max_packets=512))
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
            "last_trace_actor": self.history[-1].actor if self.history else None,
            "transformers": {
                "mode": "two_transformer_selfplay",
                "active": self._transformer_id_for_side(self.board.side_to_move),
                "white": _rule_summary(self.white_rules),
                "black": _rule_summary(self.black_rules),
            },
            "transformer_token_streams": {
                "white": _trace_to_dict(self.last_transformer_traces["white"], packet_limit=512),
                "black": _trace_to_dict(self.last_transformer_traces["black"], packet_limit=512),
            },
            "trace_legal_verification": {
                "selected_move_in_legal_set": self.last_trace_verified_legal,
                "illegal_commit_count": self.illegal_commit_count,
                "runtime_oracle_used": False,
            },
            "illegal_attempt_count": self.illegal_attempt_count,
            "illegal_commit_count": self.illegal_commit_count,
            "engine": _rule_summary(rules),
            "temperature": 0.0,
            "seed": self.seed,
            "max_plies": self.max_plies,
        }

    def _rules_for_side(self, side: str) -> PerceptaFrozenAttentionRuleComputer:
        if side == "w":
            return self.white_rules
        if side == "b":
            return self.black_rules
        raise ValueError(f"unknown side: {side}")

    def _active_rules(self) -> PerceptaFrozenAttentionRuleComputer:
        return self._rules_for_side(self.board.side_to_move)

    def _transformer_id_for_side(self, side: str) -> str:
        if side == "w":
            return "transformer_white"
        if side == "b":
            return "transformer_black"
        raise ValueError(f"unknown side: {side}")

    def _commit_selected_move(
        self,
        selected: MovePacket,
        legal_trace: tuple[TracePacket, ...],
        actor: str,
        rules: PerceptaFrozenAttentionRuleComputer,
    ) -> PerceptaParametricMoveEvent:
        legal_moves = tuple(legal_moves_from_trace(legal_trace))
        legal_uci = tuple(sorted(move.to_uci() for move in legal_moves))
        trace_verified_legal = selected.to_uci() in legal_uci
        if not trace_verified_legal:
            self.illegal_commit_count += 1
            raise ValueError(f"decoded illegal commit: {selected.to_uci()}")
        selected_tensor = rules.resolve_legal_move_tensor(self.prompt_tensor_trace, selected)
        next_board = rules.board_after_move_from_prompt(self.prompt_trace, selected)
        next_tensor_prompt = rules.prompt_tensor_from_board(next_board)
        next_prompt = rules.tensor_trace_to_packets(next_tensor_prompt)
        next_key = position_key(next_board)
        next_repetition_count = self.repetitions.get(next_key, 0) + 1
        move_tensor_trace = rules.decode_make_move_tensor_trace_host_append_only(
            self.prompt_tensor_trace,
            selected_tensor,
            ply=self.ply,
            repetition_count=next_repetition_count,
            adjudication_cap_reached=self.ply + 1 >= self.max_plies,
            max_packets=128,
        )
        move_trace = rules.tensor_trace_to_packets(move_tensor_trace)
        move_continuation = tuple(move_trace[len(self.prompt_trace) :])
        full_trace = (
            tuple(legal_trace[:-1])
            + move_continuation
        )
        terminal = _terminal_from_trace(full_trace)
        event = PerceptaParametricMoveEvent(
            actor=actor,
            transformer_id=actor if actor.startswith("transformer_") else "human",
            ply_before=self.ply,
            side_to_move=self.board.side_to_move,
            move_uci=selected.to_uci(),
            legal_before=legal_uci,
            trace=full_trace,
            terminal_after=terminal,
            trace_verified_legal=trace_verified_legal,
        )
        self.board = next_board
        self.prompt_tensor_trace = next_tensor_prompt
        self.prompt_trace = next_prompt
        self.repetitions[next_key] = next_repetition_count
        self.ply += 1
        self.terminal_status = terminal
        self.history.append(event)
        self.last_tensor_trace = move_tensor_trace
        self.last_trace = full_trace
        self.last_trace_verified_legal = trace_verified_legal
        if actor == "transformer_white":
            self.last_transformer_traces["white"] = full_trace
        elif actor == "transformer_black":
            self.last_transformer_traces["black"] = full_trace
        return event


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


def _terminal_from_trace(trace: tuple[TracePacket, ...]) -> TerminalStatus:
    for packet in reversed(trace):
        if packet.op is TraceOp.TERMINAL_SET:
            return TerminalStatus(ResultCode(packet.a0), TerminalReason(packet.a1), packet.a2)
    return TerminalStatus(ResultCode.ONGOING, TerminalReason.NONE, 0)


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


def _rule_summary(rules: PerceptaFrozenAttentionRuleComputer) -> dict[str, Any]:
    return {
        "rules_module": type(rules).__name__,
        "rule_execution_mode": rules.rule_execution_mode,
        "attention_backend": rules.attention_backend,
        "lookup_complexity": rules.lookup_complexity,
        "rule_core_execution_mode": rules.rule_core_execution_mode,
        "primitive_kernel_execution_mode": rules.primitive_kernel_execution_mode,
        "core_trace_runtime": rules.core_trace_runtime,
        "core_rule_compute_backend": rules.core_rule_compute_backend,
        "tensor_kernel_shortcut_runtime": rules.tensor_kernel_shortcut_runtime,
        "compiled_attention_block_stack": rules.compiled_attention_block_stack,
        "compiled_attention_block_count": rules.compiled_attention_block_count,
        "compiled_attention_head_count": rules.compiled_attention_head_count,
        "residual_trace_write_count": rules.residual_trace_write_count,
        "percepta_compiler_pipeline": rules.percepta_compiler_pipeline,
        "rule_compiler_backend": rules.rule_compiler_backend,
        "rule_microprogram_source": rules.rule_microprogram_source,
        "rule_microprogram_instruction_count": rules.rule_microprogram_instruction_count,
        "compiled_rule_program_weight_count": rules.compiled_rule_program_weight_count,
        "unified_rule_executor_runtime": rules.unified_rule_executor_runtime,
        "handwritten_stack_primitive_runtime": rules.handwritten_stack_primitive_runtime,
        "matrix_attention_interpreter_runtime": rules.matrix_attention_interpreter_runtime,
        "executor_substrate": rules.executor_substrate,
        "attention_step_operator": rules.attention_step_operator,
        "pytorch_domain_shortcut_runtime": rules.pytorch_domain_shortcut_runtime,
        "matrix_attention_step_count": rules.matrix_attention_step_count,
        "matrix_residual_write_count": rules.matrix_residual_write_count,
        "python_host_boundary_role": rules.python_host_boundary_role,
        "tensor_trace_core_runtime": rules.tensor_trace_core_runtime,
        "tracepacket_core_runtime": rules.tracepacket_core_runtime,
        "python_rule_primitive_runtime": rules.python_rule_primitive_runtime,
        "python_control_flow_rule_primitives": rules.python_control_flow_rule_primitives,
        "compiled_rule_primitives": list(rules.compiled_rule_primitives),
        "compiled_rule_primitive_count": rules.compiled_rule_primitive_count,
        "tensor_kernel_count": rules.tensor_kernel_count,
        "graph_execution_counts": dict(rules.graph_execution_counts),
        "parametric_rule_weights": rules.parametric_rule_weights,
        "host_append_only": rules.host_append_only,
        "token_streaming": rules.token_streaming,
        "uses_mlp": rules.uses_mlp,
        "uses_dense_scan": rules.uses_dense_scan,
        "compiled_layer_graph_serialized": rules.compiled_layer_graph_serialized,
        "position_lookup": rules.position_lookup,
        "finite_prompt_lookup": rules.finite_prompt_lookup,
        "compiled_prompt_count": rules.compiled_prompt_count,
        "compiled_isa_instruction_count": rules.compiled_isa_instruction_count,
        "compiled_microprogram_step_count": rules.compiled_microprogram_step_count,
        "compiled_attention_layer_count": rules.compiled_attention_layer_count,
        "max_lookup_steps": rules.max_lookup_steps,
        "last_lookup_steps": rules.last_lookup_steps,
        "trainable_rule_parameters": rules.trainable_rule_parameter_count(),
        "compiled_rule_parameters": rules.compiled_rule_parameter_count(),
        "python_rule_executor_runtime": False,
        "strategy_module": "none",
        "strategy_training": False,
        "external_tree_search": False,
        "human_game_data": False,
        "engine_labels": False,
        "tablebase_labels": False,
        "handcrafted_evaluation": False,
    }


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
