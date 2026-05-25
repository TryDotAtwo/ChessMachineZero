"""Frozen-weight chess rule executor.

The module stores chess geometry and rule constants in non-trainable model state.
Runtime methods execute trace-producing chess computation by indexing those
frozen tensors instead of calling an outside rules engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chess_machine_zero.chess.board_io import NO_EP, BoardState, castling_mask, piece_from_token, piece_token
from chess_machine_zero.chess.move_packet import MoveFlag, MovePacket, Promo
from chess_machine_zero.chess.outcome import ResultCode, TerminalReason, TerminalStatus
from chess_machine_zero.vm.trace_packet import RegId, TraceOp, TracePacket, TraceTag


WHITE = "w"
BLACK = "b"
PIECE_KIND_EMPTY = 0
PIECE_KIND_PAWN = 1
PIECE_KIND_KNIGHT = 2
PIECE_KIND_BISHOP = 3
PIECE_KIND_ROOK = 4
PIECE_KIND_QUEEN = 5
PIECE_KIND_KING = 6
NO_SQUARE = -1
PROMOTION_ORDER = (Promo.KNIGHT, Promo.BISHOP, Promo.ROOK, Promo.QUEEN)
CASTLING_FROM_BITS = ((1, "K"), (2, "Q"), (4, "k"), (8, "q"))


class CompiledChessRuleWeights(nn.Module):
    """Non-trainable state tensors containing chess rule tables."""

    def __init__(self) -> None:
        super().__init__()
        self.piece_kind = _frozen_weight(_piece_kind_table())
        self.piece_color = _frozen_weight(_piece_color_table())
        self.knight_targets = _frozen_weight(_leaper_targets(((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))))
        self.king_targets = _frozen_weight(_leaper_targets(((1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1))))
        self.ray_squares = _frozen_weight(_ray_squares())
        self.pawn_single_push = _frozen_weight(_pawn_single_push())
        self.pawn_double_push = _frozen_weight(_pawn_double_push())
        self.pawn_captures = _frozen_weight(_pawn_captures())
        self.promotion_rank = _frozen_weight(torch.tensor([6, 1], dtype=torch.long))
        self.pawn_start_rank = _frozen_weight(torch.tensor([1, 6], dtype=torch.long))
        self.castle_king_from_to = _frozen_weight(torch.tensor([[4, 6], [4, 2], [60, 62], [60, 58]], dtype=torch.long))
        self.castle_rook_from_to = _frozen_weight(torch.tensor([[7, 5], [0, 3], [63, 61], [56, 59]], dtype=torch.long))
        self.castle_empty_squares = _frozen_weight(_castle_empty_squares())
        self.castle_transit_squares = _frozen_weight(torch.tensor([5, 3, 61, 59], dtype=torch.long))
        self.castle_right_bits = _frozen_weight(torch.tensor([1, 2, 4, 8], dtype=torch.long))
        self.rook_home_squares = _frozen_weight(torch.tensor([0, 7, 56, 63], dtype=torch.long))
        self.rook_home_right_bits = _frozen_weight(torch.tensor([2, 1, 8, 4], dtype=torch.long))
        self.bishop_square_colors = _frozen_weight(_bishop_square_colors())

    def compiled_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class WeightCompiledRulesTransformer(nn.Module):
    """Fixed chess rules hosted as frozen model state."""

    rule_execution_mode = "weight_compiled"

    def __init__(self) -> None:
        super().__init__()
        self.rule_weights = CompiledChessRuleWeights()

    def trainable_rule_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def compiled_rule_parameter_count(self) -> int:
        return self.rule_weights.compiled_parameter_count()

    def prompt_trace_from_board(self, board: BoardState) -> tuple[TracePacket, ...]:
        trace = [
            TracePacket(TraceOp.WRITE_SQ, square, piece_token(piece), 0, 0, TraceTag.BOARD, 0)
            for square, piece in enumerate(board.squares)
        ]
        trace.append(TracePacket(TraceOp.WRITE_REG, int(RegId.SIDE_TO_MOVE), 0 if board.side_to_move == WHITE else 1, 0, 0, TraceTag.STATE, 0))
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
        for move in self.generate_pseudo_legal_moves(board):
            trace.append(TracePacket(TraceOp.CANDIDATE, move.move_id, move.from_sq, move.to_sq, int(move.promo), TraceTag.MOVE, int(move.flags)))
            trace.append(TracePacket(TraceOp.LEGAL_SET, move.move_id, int(self.is_legal_move(board, move)), 0, 0, TraceTag.LEGAL, 0))
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
        next_board = self.make_move_state(board, selected)
        trace = list(prompt_trace)
        trace.append(TracePacket(TraceOp.COMMIT_MOVE, selected.move_id, selected.from_sq, selected.to_sq, int(selected.promo), TraceTag.MOVE, int(selected.flags)))
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
        return self.make_move_state(board, selected)

    def terminal_trace_from_prompt(
        self,
        prompt_trace: tuple[TracePacket, ...] | list[TracePacket],
        ply: int,
        repetition_count: int = 1,
        adjudication_cap_reached: bool = False,
    ) -> tuple[TracePacket, ...]:
        return self.terminal_trace_from_board(board_state_from_prompt_trace(prompt_trace), ply, repetition_count, adjudication_cap_reached)

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
        moves: list[MovePacket] = []
        for square, piece in enumerate(board.squares):
            if piece is None or _color_of(piece) != board.side_to_move:
                continue
            token = piece_token(piece)
            kind = self._piece_kind_from_token(token)
            if kind == PIECE_KIND_PAWN:
                moves.extend(self._pawn_moves(board, square, piece))
            elif kind == PIECE_KIND_KNIGHT:
                moves.extend(self._leaper_moves(board, square, piece, self.rule_weights.knight_targets))
            elif kind == PIECE_KIND_BISHOP:
                moves.extend(self._ray_moves(board, square, piece, range(0, 4)))
            elif kind == PIECE_KIND_ROOK:
                moves.extend(self._ray_moves(board, square, piece, range(4, 8)))
            elif kind == PIECE_KIND_QUEEN:
                moves.extend(self._ray_moves(board, square, piece, range(0, 8)))
            elif kind == PIECE_KIND_KING:
                moves.extend(self._king_moves(board, square, piece))
        return sorted(moves, key=lambda move: move.sort_key())

    def is_legal_move(self, board: BoardState, move: MovePacket) -> bool:
        piece = board.squares[move.from_sq]
        if piece is None or _color_of(piece) != board.side_to_move:
            return False
        target = board.squares[move.to_sq]
        if target is not None and (_color_of(target) == board.side_to_move or target.lower() == "k"):
            return False
        if move.flags & MoveFlag.CASTLE and not self._castle_path_is_safe(board, move):
            return False
        next_squares = self.apply_move_squares(board, move)
        king_square = find_king(next_squares, board.side_to_move)
        if king_square is None:
            return False
        return not self.is_square_attacked(next_squares, king_square, _opposite(board.side_to_move))

    def make_move_state(self, board: BoardState, move: MovePacket) -> BoardState:
        piece = board.squares[move.from_sq]
        target = board.squares[move.to_sq]
        if piece is None:
            raise ValueError(f"cannot move empty square: {move.from_sq}")
        next_squares = self.apply_move_squares(board, move)
        next_side = _opposite(board.side_to_move)
        next_castling = self.update_castling_rights(board.castling, piece, move.from_sq, move.to_sq, target)
        next_ep = self.next_ep_square(piece, move)
        next_halfmove = 0 if piece.lower() == "p" or target is not None or (move.flags & MoveFlag.EP) else board.halfmove_clock + 1
        next_fullmove = board.fullmove_number + 1 if board.side_to_move == BLACK else board.fullmove_number
        return BoardState(next_squares, next_side, next_castling, next_ep, next_halfmove, next_fullmove)

    def terminal_status(
        self,
        board: BoardState,
        ply: int = 0,
        repetition_count: int = 1,
        adjudication_cap_reached: bool = False,
    ) -> TerminalStatus:
        if adjudication_cap_reached:
            return TerminalStatus(ResultCode.DRAW, TerminalReason.ADJUDICATION_CAP, ply)
        if repetition_count >= 3:
            return TerminalStatus(ResultCode.DRAW, TerminalReason.THREEFOLD, ply)
        if board.halfmove_clock >= 100:
            return TerminalStatus(ResultCode.DRAW, TerminalReason.FIFTY_MOVE, ply)
        legal = [move for move in self.generate_pseudo_legal_moves(board) if self.is_legal_move(board, move)]
        if not legal:
            king_square = find_king(board.squares, board.side_to_move)
            in_check = king_square is not None and self.is_square_attacked(board.squares, king_square, _opposite(board.side_to_move))
            if in_check:
                winner = ResultCode.BLACK_WIN if board.side_to_move == WHITE else ResultCode.WHITE_WIN
                return TerminalStatus(winner, TerminalReason.CHECKMATE, ply)
            return TerminalStatus(ResultCode.DRAW, TerminalReason.STALEMATE, ply)
        if self.insufficient_material(board.squares):
            return TerminalStatus(ResultCode.DRAW, TerminalReason.INSUFFICIENT_MATERIAL, ply)
        return TerminalStatus(ResultCode.ONGOING, TerminalReason.NONE, ply)

    def apply_move_squares(self, board: BoardState, move: MovePacket) -> tuple[str | None, ...]:
        squares = list(board.squares)
        piece = squares[move.from_sq]
        if piece is None:
            raise ValueError(f"cannot move empty square: {move.from_sq}")
        squares[move.from_sq] = None
        if move.flags & MoveFlag.EP:
            capture_square = move.to_sq - 8 if _color_of(piece) == WHITE else move.to_sq + 8
            squares[capture_square] = None
        if move.flags & MoveFlag.CASTLE:
            self._apply_castle_rook_move(squares, move)
        squares[move.to_sq] = self.promotion_piece(piece, move.promo)
        return tuple(squares)

    def update_castling_rights(self, castling: str, piece: str, from_sq: int, to_sq: int, captured: str | None) -> str:
        mask = castling_mask(castling)
        if piece == "K":
            mask &= ~3
        elif piece == "k":
            mask &= ~12
        for index, square in enumerate(self.rule_weights.rook_home_squares.tolist()):
            if from_sq == square or to_sq == square or (captured is not None and to_sq == square):
                mask &= ~int(self.rule_weights.rook_home_right_bits[index].item())
        return "".join(symbol for bit, symbol in CASTLING_FROM_BITS if mask & bit)

    def next_ep_square(self, piece: str, move: MovePacket) -> int | None:
        if piece.lower() != "p" or abs(move.to_sq - move.from_sq) != 16:
            return None
        return (move.from_sq + move.to_sq) // 2

    def insufficient_material(self, squares: tuple[str | None, ...] | list[str | None]) -> bool:
        pieces = [piece for piece in squares if piece is not None and piece.lower() != "k"]
        if not pieces:
            return True
        if any(piece.lower() in ("p", "r", "q") for piece in pieces):
            return False
        bishops = [index for index, piece in enumerate(squares) if piece is not None and piece.lower() == "b"]
        knights = [piece for piece in pieces if piece.lower() == "n"]
        if len(pieces) == 1 and (bishops or knights):
            return True
        if len(pieces) == len(bishops):
            colors = {int(self.rule_weights.bishop_square_colors[square].item()) for square in bishops}
            return len(colors) <= 1
        if len(pieces) == len(knights) and len(knights) <= 2:
            return True
        return False

    def is_square_attacked(self, squares: tuple[str | None, ...] | list[str | None], target: int, by_color: str) -> bool:
        for square, piece in enumerate(squares):
            if piece is None or _color_of(piece) != by_color:
                continue
            if self._piece_attacks_square(squares, square, piece, target):
                return True
        return False

    def promotion_piece(self, piece: str, promo: Promo) -> str:
        if promo is Promo.NONE:
            return piece
        promoted = {
            Promo.KNIGHT: "N",
            Promo.BISHOP: "B",
            Promo.ROOK: "R",
            Promo.QUEEN: "Q",
        }[promo]
        return promoted if _color_of(piece) == WHITE else promoted.lower()

    def _pawn_moves(self, board: BoardState, square: int, piece: str) -> list[MovePacket]:
        color_id = 0 if _color_of(piece) == WHITE else 1
        rank = square // 8
        moves: list[MovePacket] = []
        one_step = int(self.rule_weights.pawn_single_push[color_id, square].item())
        if one_step != NO_SQUARE and board.squares[one_step] is None:
            if rank == int(self.rule_weights.promotion_rank[color_id].item()):
                moves.extend(_promotion_moves(square, one_step, MoveFlag.QUIET))
            else:
                moves.append(MovePacket(square, one_step))
                two_step = int(self.rule_weights.pawn_double_push[color_id, square].item())
                if two_step != NO_SQUARE and board.squares[two_step] is None:
                    moves.append(MovePacket(square, two_step))
        for target in self.rule_weights.pawn_captures[color_id, square].tolist():
            if target == NO_SQUARE:
                continue
            target_piece = board.squares[target]
            if target_piece is not None and _color_of(target_piece) != _color_of(piece) and target_piece.lower() != "k":
                if rank == int(self.rule_weights.promotion_rank[color_id].item()):
                    moves.extend(_promotion_moves(square, target, MoveFlag.CAPTURE))
                else:
                    moves.append(MovePacket(square, target, Promo.NONE, MoveFlag.CAPTURE))
            if board.ep_square == target:
                moves.append(MovePacket(square, target, Promo.NONE, MoveFlag.EP | MoveFlag.CAPTURE))
        return moves

    def _leaper_moves(self, board: BoardState, square: int, piece: str, target_table: torch.Tensor) -> list[MovePacket]:
        moves: list[MovePacket] = []
        own_color = _color_of(piece)
        for target in target_table[square].tolist():
            if target == NO_SQUARE:
                continue
            flags = _target_move_flags(board.squares[target], own_color)
            if flags is not None:
                moves.append(MovePacket(square, target, Promo.NONE, flags))
        return moves

    def _ray_moves(self, board: BoardState, square: int, piece: str, ray_indices: range) -> list[MovePacket]:
        moves: list[MovePacket] = []
        own_color = _color_of(piece)
        for ray_index in ray_indices:
            for target in self.rule_weights.ray_squares[square, ray_index].tolist():
                if target == NO_SQUARE:
                    break
                target_piece = board.squares[target]
                flags = _target_move_flags(target_piece, own_color)
                if flags is not None:
                    moves.append(MovePacket(square, target, Promo.NONE, flags))
                if target_piece is not None:
                    break
        return moves

    def _king_moves(self, board: BoardState, square: int, piece: str) -> list[MovePacket]:
        moves = self._leaper_moves(board, square, piece, self.rule_weights.king_targets)
        color = _color_of(piece)
        if color == WHITE and square == 4:
            if self._castle_candidate_available(board, castle_index=0):
                moves.append(MovePacket(4, 6, Promo.NONE, MoveFlag.CASTLE))
            if self._castle_candidate_available(board, castle_index=1):
                moves.append(MovePacket(4, 2, Promo.NONE, MoveFlag.CASTLE))
        if color == BLACK and square == 60:
            if self._castle_candidate_available(board, castle_index=2):
                moves.append(MovePacket(60, 62, Promo.NONE, MoveFlag.CASTLE))
            if self._castle_candidate_available(board, castle_index=3):
                moves.append(MovePacket(60, 58, Promo.NONE, MoveFlag.CASTLE))
        return moves

    def _castle_candidate_available(self, board: BoardState, castle_index: int) -> bool:
        if not (castling_mask(board.castling) & int(self.rule_weights.castle_right_bits[castle_index].item())):
            return False
        rook_from = int(self.rule_weights.castle_rook_from_to[castle_index, 0].item())
        expected_rook = "R" if castle_index < 2 else "r"
        if board.squares[rook_from] != expected_rook:
            return False
        for square in self.rule_weights.castle_empty_squares[castle_index].tolist():
            if square != NO_SQUARE and board.squares[square] is not None:
                return False
        return True

    def _castle_path_is_safe(self, board: BoardState, move: MovePacket) -> bool:
        castle_index = self._castle_index_for_move(move)
        enemy = _opposite(board.side_to_move)
        if self.is_square_attacked(board.squares, move.from_sq, enemy):
            return False
        transit_square = int(self.rule_weights.castle_transit_squares[castle_index].item())
        transit_squares = list(board.squares)
        king = transit_squares[move.from_sq]
        transit_squares[move.from_sq] = None
        transit_squares[transit_square] = king
        if self.is_square_attacked(transit_squares, transit_square, enemy):
            return False
        final_squares = self.apply_move_squares(board, move)
        return not self.is_square_attacked(final_squares, move.to_sq, enemy)

    def _apply_castle_rook_move(self, squares: list[str | None], move: MovePacket) -> None:
        castle_index = self._castle_index_for_move(move)
        rook_from = int(self.rule_weights.castle_rook_from_to[castle_index, 0].item())
        rook_to = int(self.rule_weights.castle_rook_from_to[castle_index, 1].item())
        squares[rook_to] = squares[rook_from]
        squares[rook_from] = None

    def _castle_index_for_move(self, move: MovePacket) -> int:
        for index, (from_sq, to_sq) in enumerate(self.rule_weights.castle_king_from_to.tolist()):
            if move.from_sq == from_sq and move.to_sq == to_sq:
                return index
        raise ValueError(f"unknown castle move: {move.to_uci()}")

    def _piece_attacks_square(self, squares: tuple[str | None, ...] | list[str | None], square: int, piece: str, target: int) -> bool:
        kind = self._piece_kind_from_token(piece_token(piece))
        color_id = 0 if _color_of(piece) == WHITE else 1
        if kind == PIECE_KIND_PAWN:
            return target in self.rule_weights.pawn_captures[color_id, square].tolist()
        if kind == PIECE_KIND_KNIGHT:
            return target in self.rule_weights.knight_targets[square].tolist()
        if kind == PIECE_KIND_KING:
            return target in self.rule_weights.king_targets[square].tolist()
        ray_indices = range(0, 4) if kind == PIECE_KIND_BISHOP else range(4, 8) if kind == PIECE_KIND_ROOK else range(0, 8)
        for ray_index in ray_indices:
            for current in self.rule_weights.ray_squares[square, ray_index].tolist():
                if current == NO_SQUARE:
                    break
                if current == target:
                    return True
                if squares[current] is not None:
                    break
        return False

    def _piece_kind_from_token(self, token: int) -> int:
        return int(self.rule_weights.piece_kind[token].item())

    def _resolve_legal_move(
        self,
        prompt_trace: tuple[TracePacket, ...] | list[TracePacket],
        move: MovePacket | str,
    ) -> MovePacket:
        move_uci = move if isinstance(move, str) else move.to_uci()
        legal_by_uci = {candidate.to_uci(): candidate for candidate in self.legal_moves_from_prompt(prompt_trace)}
        if move_uci not in legal_by_uci:
            raise ValueError(f"illegal weight-compiled move: {move_uci}")
        return legal_by_uci[move_uci]


@dataclass(frozen=True, slots=True)
class WeightCompiledRuleCompiler:
    """Build frozen-weight rule executors."""

    def compile_legal_generator(self) -> WeightCompiledRulesTransformer:
        return WeightCompiledRulesTransformer()


def legal_moves_from_trace(trace: tuple[TracePacket, ...] | list[TracePacket]) -> list[MovePacket]:
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


def board_transition_trace(before: BoardState, after: BoardState, ply: int) -> list[TracePacket]:
    trace: list[TracePacket] = []
    for square, (old_piece, new_piece) in enumerate(zip(before.squares, after.squares, strict=True)):
        if old_piece != new_piece:
            trace.append(TracePacket(TraceOp.WRITE_SQ, square, piece_token(new_piece), ply, 0, TraceTag.BOARD, 1))
    trace.append(TracePacket(TraceOp.WRITE_REG, int(RegId.SIDE_TO_MOVE), 0 if after.side_to_move == WHITE else 1, ply, 0, TraceTag.STATE, 1))
    trace.append(TracePacket(TraceOp.WRITE_CASTLE, castling_mask(after.castling), ply, 0, 0, TraceTag.STATE, 1))
    trace.append(TracePacket(TraceOp.WRITE_EP, after.ep_square if after.ep_square is not None else NO_EP, ply, 0, 0, TraceTag.STATE, 1))
    trace.append(TracePacket(TraceOp.WRITE_CLOCK, after.halfmove_clock, after.fullmove_number, ply, 0, TraceTag.STATE, 1))
    return trace


def board_state_from_prompt_trace(prompt_trace: tuple[TracePacket, ...] | list[TracePacket]) -> BoardState:
    squares: list[str | None] = [None] * 64
    side_to_move = WHITE
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
            side_to_move = WHITE if packet.a1 == 0 else BLACK
        elif packet.op is TraceOp.WRITE_CASTLE:
            castling = "".join(symbol for bit, symbol in CASTLING_FROM_BITS if packet.a0 & bit)
        elif packet.op is TraceOp.WRITE_EP:
            ep_square = None if packet.a0 == NO_EP else packet.a0
        elif packet.op is TraceOp.WRITE_CLOCK:
            halfmove_clock = packet.a0
            fullmove_number = packet.a1
    return BoardState(tuple(squares), side_to_move, castling, ep_square, halfmove_clock, fullmove_number)


def find_king(squares: tuple[str | None, ...] | list[str | None], color: str) -> int | None:
    king = "K" if color == WHITE else "k"
    for square, piece in enumerate(squares):
        if piece == king:
            return square
    return None


def position_key(board: BoardState) -> tuple[str, str, str, int | None]:
    return ("".join(piece or "." for piece in board.squares), board.side_to_move, board.castling, board.ep_square)


def _promotion_moves(from_sq: int, to_sq: int, flags: MoveFlag) -> list[MovePacket]:
    return [MovePacket(from_sq, to_sq, promo, flags | MoveFlag.PROMOTION) for promo in PROMOTION_ORDER]


def _target_move_flags(target_piece: str | None, own_color: str) -> MoveFlag | None:
    if target_piece is None:
        return MoveFlag.QUIET
    if _color_of(target_piece) == own_color or target_piece.lower() == "k":
        return None
    return MoveFlag.CAPTURE


def _color_of(piece: str) -> str:
    return WHITE if piece.isupper() else BLACK


def _opposite(color: str) -> str:
    return BLACK if color == WHITE else WHITE


def _piece_kind_table() -> torch.Tensor:
    table = torch.zeros(13, dtype=torch.long)
    for token, kind in {
        1: PIECE_KIND_PAWN,
        2: PIECE_KIND_KNIGHT,
        3: PIECE_KIND_BISHOP,
        4: PIECE_KIND_ROOK,
        5: PIECE_KIND_QUEEN,
        6: PIECE_KIND_KING,
        7: PIECE_KIND_PAWN,
        8: PIECE_KIND_KNIGHT,
        9: PIECE_KIND_BISHOP,
        10: PIECE_KIND_ROOK,
        11: PIECE_KIND_QUEEN,
        12: PIECE_KIND_KING,
    }.items():
        table[token] = kind
    return table


def _frozen_weight(tensor: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(tensor, requires_grad=False)


def _piece_color_table() -> torch.Tensor:
    table = torch.full((13,), -1, dtype=torch.long)
    table[1:7] = 0
    table[7:13] = 1
    return table


def _leaper_targets(deltas: tuple[tuple[int, int], ...]) -> torch.Tensor:
    table = torch.full((64, len(deltas)), NO_SQUARE, dtype=torch.long)
    for square in range(64):
        file_index = square % 8
        rank = square // 8
        for index, (df, dr) in enumerate(deltas):
            target_file = file_index + df
            target_rank = rank + dr
            if 0 <= target_file < 8 and 0 <= target_rank < 8:
                table[square, index] = target_rank * 8 + target_file
    return table


def _ray_squares() -> torch.Tensor:
    directions = ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1))
    table = torch.full((64, len(directions), 7), NO_SQUARE, dtype=torch.long)
    for square in range(64):
        file_index = square % 8
        rank = square // 8
        for direction_index, (df, dr) in enumerate(directions):
            target_file = file_index + df
            target_rank = rank + dr
            step = 0
            while 0 <= target_file < 8 and 0 <= target_rank < 8:
                table[square, direction_index, step] = target_rank * 8 + target_file
                target_file += df
                target_rank += dr
                step += 1
    return table


def _pawn_single_push() -> torch.Tensor:
    table = torch.full((2, 64), NO_SQUARE, dtype=torch.long)
    for square in range(64):
        file_index = square % 8
        rank = square // 8
        if rank < 7:
            table[0, square] = (rank + 1) * 8 + file_index
        if rank > 0:
            table[1, square] = (rank - 1) * 8 + file_index
    return table


def _pawn_double_push() -> torch.Tensor:
    table = torch.full((2, 64), NO_SQUARE, dtype=torch.long)
    for file_index in range(8):
        table[0, 8 + file_index] = 24 + file_index
        table[1, 48 + file_index] = 32 + file_index
    return table


def _pawn_captures() -> torch.Tensor:
    table = torch.full((2, 64, 2), NO_SQUARE, dtype=torch.long)
    for square in range(64):
        file_index = square % 8
        rank = square // 8
        for color_id, direction in ((0, 1), (1, -1)):
            target_rank = rank + direction
            write_index = 0
            for file_delta in (-1, 1):
                target_file = file_index + file_delta
                if 0 <= target_file < 8 and 0 <= target_rank < 8:
                    table[color_id, square, write_index] = target_rank * 8 + target_file
                write_index += 1
    return table


def _castle_empty_squares() -> torch.Tensor:
    table = torch.full((4, 3), NO_SQUARE, dtype=torch.long)
    table[0, :2] = torch.tensor([5, 6], dtype=torch.long)
    table[1, :3] = torch.tensor([3, 2, 1], dtype=torch.long)
    table[2, :2] = torch.tensor([61, 62], dtype=torch.long)
    table[3, :3] = torch.tensor([59, 58, 57], dtype=torch.long)
    return table


def _bishop_square_colors() -> torch.Tensor:
    return torch.tensor([(square % 8 + square // 8) % 2 for square in range(64)], dtype=torch.long)
