"""Frozen tensor kernels for Percepta-style chess rule primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chess_machine_zero.chess.board_io import BoardState, piece_from_token, piece_token
from chess_machine_zero.chess.move_packet import MoveFlag, MovePacket, Promo
from chess_machine_zero.chess.outcome import ResultCode, TerminalReason


WHITE_ID = 0
BLACK_ID = 1
EMPTY_TOKEN = 0
WHITE_KING_TOKEN = 6
BLACK_KING_TOKEN = 12
PAWN_KIND = 1
KNIGHT_KIND = 2
BISHOP_KIND = 3
ROOK_KIND = 4
QUEEN_KIND = 5
KING_KIND = 6
NO_EP = 64


@dataclass(frozen=True, slots=True)
class TensorBoardState:
    squares: torch.Tensor
    side_id: torch.Tensor
    castle_mask: torch.Tensor
    ep_square: torch.Tensor
    halfmove_clock: torch.Tensor
    fullmove_number: torch.Tensor


@dataclass(frozen=True, slots=True)
class TensorCandidateSet:
    from_sq: torch.Tensor
    to_sq: torch.Tensor
    promo: torch.Tensor
    flags: torch.Tensor
    legal: torch.Tensor

    @property
    def move_id(self) -> torch.Tensor:
        return self.from_sq * 64 * 5 + self.to_sq * 5 + self.promo


@dataclass(frozen=True, slots=True)
class TensorTerminalStatus:
    result: torch.Tensor
    reason: torch.Tensor


class FrozenAttentionTensorRuleKernels(nn.Module):
    """Rule primitives lowered to frozen tensor operations."""

    kernel_execution_mode = "pure_frozen_attention_tensor_layers"
    kernel_names = (
        "PIECE_DISPATCH",
        "RAY_SCAN",
        "ATTACK_TEST",
        "LEGAL_FILTER",
        "MAKE_MOVE",
        "TERMINAL_PREDICATES",
    )

    def __init__(self) -> None:
        super().__init__()
        candidate_from, candidate_to, candidate_promo = _candidate_universe()
        self.candidate_from = _frozen(candidate_from)
        self.candidate_to = _frozen(candidate_to)
        self.candidate_promo = _frozen(candidate_promo)
        self.candidate_from_rank = _frozen(candidate_from // 8)
        self.candidate_to_rank = _frozen(candidate_to // 8)
        self.piece_kind = _frozen(_piece_kind_table())
        self.piece_color = _frozen(_piece_color_table())
        self.knight_attack_bool = _frozen(_leaper_bool(((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))))
        self.king_attack_bool = _frozen(_leaper_bool(((1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1))))
        self.pawn_attack_bool = _frozen(_pawn_attack_bool())
        bishop_line, rook_line, between = _line_tables()
        self.bishop_line_bool = _frozen(bishop_line)
        self.rook_line_bool = _frozen(rook_line)
        self.between_from_target = _frozen(between)
        self.candidate_bishop_line = _frozen(bishop_line[candidate_from, candidate_to])
        self.candidate_rook_line = _frozen(rook_line[candidate_from, candidate_to])
        self.candidate_between = _frozen(between[candidate_from, candidate_to])
        self.candidate_castle_index = _frozen(_candidate_castle_index(candidate_from, candidate_to))
        self.castle_right_bits = _frozen(torch.tensor([1, 2, 4, 8], dtype=torch.long))
        self.castle_rook_from = _frozen(torch.tensor([7, 0, 63, 56], dtype=torch.long))
        self.castle_rook_to = _frozen(torch.tensor([5, 3, 61, 59], dtype=torch.long))
        self.castle_rook_token = _frozen(torch.tensor([4, 4, 10, 10], dtype=torch.long))
        self.castle_transit = _frozen(torch.tensor([5, 3, 61, 59], dtype=torch.long))
        self.castle_empty_mask = _frozen(_castle_empty_mask())
        self.rook_home_squares = _frozen(torch.tensor([0, 7, 56, 63], dtype=torch.long))
        self.rook_home_right_bits = _frozen(torch.tensor([2, 1, 8, 4], dtype=torch.long))
        self.bishop_square_color = _frozen(torch.tensor([(square % 8 + square // 8) % 2 for square in range(64)], dtype=torch.long))

    @property
    def tensor_kernel_count(self) -> int:
        return len(self.kernel_names)

    def board_to_tensor(self, board: BoardState) -> TensorBoardState:
        return TensorBoardState(
            squares=torch.tensor([piece_token(piece) for piece in board.squares], dtype=torch.long),
            side_id=torch.tensor(WHITE_ID if board.side_to_move == "w" else BLACK_ID, dtype=torch.long),
            castle_mask=torch.tensor(_castle_mask_from_string(board.castling), dtype=torch.long),
            ep_square=torch.tensor(board.ep_square if board.ep_square is not None else NO_EP, dtype=torch.long),
            halfmove_clock=torch.tensor(board.halfmove_clock, dtype=torch.long),
            fullmove_number=torch.tensor(board.fullmove_number, dtype=torch.long),
        )

    def board_from_tensor(
        self,
        squares: torch.Tensor,
        side_id: torch.Tensor,
        castle_mask: torch.Tensor,
        ep_square: torch.Tensor,
        halfmove_clock: torch.Tensor,
        fullmove_number: torch.Tensor,
    ) -> BoardState:
        square_tuple = tuple(piece_from_token(int(token)) for token in squares.tolist())
        side = "w" if int(side_id.item()) == WHITE_ID else "b"
        castling = "".join(symbol for bit, symbol in ((1, "K"), (2, "Q"), (4, "k"), (8, "q")) if int(castle_mask.item()) & bit)
        ep_value = int(ep_square.item())
        return BoardState(
            square_tuple,
            side,
            castling,
            None if ep_value == NO_EP else ep_value,
            int(halfmove_clock.item()),
            int(fullmove_number.item()),
        )

    def piece_dispatch(self, board_tokens: torch.Tensor, side_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from_tokens = board_tokens[self.candidate_from]
        target_tokens = board_tokens[self.candidate_to]
        from_kind = self.piece_kind[from_tokens]
        target_kind = self.piece_kind[target_tokens]
        from_color = self.piece_color[from_tokens]
        target_color = self.piece_color[target_tokens]
        own_piece = from_color.eq(side_id)
        target_enemy = target_color.eq(1 - side_id)
        target_own = target_color.eq(side_id)
        target_king = target_kind.eq(KING_KIND)
        target_available = torch.logical_not(torch.logical_or(target_own, target_king))
        return from_kind, own_piece, target_enemy, target_available

    def ray_scan(self, board_tokens: torch.Tensor) -> torch.Tensor:
        occupied = board_tokens.ne(EMPTY_TOKEN)
        blocked = torch.logical_and(self.candidate_between.bool(), occupied.unsqueeze(0)).any(dim=1)
        return torch.logical_not(blocked)

    def attack_test(self, squares_tokens: torch.Tensor, targets: torch.Tensor, by_side_id: torch.Tensor) -> torch.Tensor:
        batch = squares_tokens.shape[0]
        side = torch.as_tensor(by_side_id, dtype=torch.long).reshape(-1).expand(batch)
        target = targets.to(torch.long).clamp(0, 63)
        colors = self.piece_color[squares_tokens]
        kinds = self.piece_kind[squares_tokens]
        attacker_color = colors.eq(side[:, None])
        pawn_by_white = self.pawn_attack_bool[WHITE_ID, :, target].transpose(0, 1).bool()
        pawn_by_black = self.pawn_attack_bool[BLACK_ID, :, target].transpose(0, 1).bool()
        pawn_hits = torch.where(side[:, None].eq(WHITE_ID), pawn_by_white, pawn_by_black)
        knight_hits = self.knight_attack_bool[:, target].transpose(0, 1).bool()
        king_hits = self.king_attack_bool[:, target].transpose(0, 1).bool()
        bishop_line = self.bishop_line_bool[:, target].transpose(0, 1).bool()
        rook_line = self.rook_line_bool[:, target].transpose(0, 1).bool()
        between = self.between_from_target[:, target, :].permute(1, 0, 2).bool()
        occupied = squares_tokens.ne(EMPTY_TOKEN)
        blocked = torch.logical_and(between, occupied[:, None, :]).any(dim=2)
        pawn_attackers = torch.logical_and(kinds.eq(PAWN_KIND), pawn_hits)
        knight_attackers = torch.logical_and(kinds.eq(KNIGHT_KIND), knight_hits)
        king_attackers = torch.logical_and(kinds.eq(KING_KIND), king_hits)
        bishop_attackers = torch.logical_and(torch.logical_or(kinds.eq(BISHOP_KIND), kinds.eq(QUEEN_KIND)), torch.logical_and(bishop_line, torch.logical_not(blocked)))
        rook_attackers = torch.logical_and(torch.logical_or(kinds.eq(ROOK_KIND), kinds.eq(QUEEN_KIND)), torch.logical_and(rook_line, torch.logical_not(blocked)))
        attackers = torch.logical_and(
            attacker_color,
            torch.logical_or(
                torch.logical_or(torch.logical_or(pawn_attackers, knight_attackers), king_attackers),
                torch.logical_or(bishop_attackers, rook_attackers),
            ),
        )
        return attackers.any(dim=1)

    def legal_filter(
        self,
        board_tokens: torch.Tensor,
        side_id: torch.Tensor,
        candidate_from: torch.Tensor,
        candidate_to: torch.Tensor,
        candidate_promo: torch.Tensor,
        candidate_flags: torch.Tensor,
    ) -> torch.Tensor:
        next_tokens = self._apply_move_batch(board_tokens, candidate_from, candidate_to, candidate_promo, candidate_flags)
        own_king = torch.where(side_id.eq(WHITE_ID), torch.tensor(WHITE_KING_TOKEN), torch.tensor(BLACK_KING_TOKEN))
        enemy_side = 1 - side_id
        king_mask = next_tokens.eq(own_king)
        king_present = king_mask.any(dim=1)
        king_square = king_mask.to(torch.long).argmax(dim=1)
        final_attacked = self.attack_test(next_tokens, king_square, enemy_side)
        castle_move = candidate_flags.bitwise_and(int(MoveFlag.CASTLE)).ne(0)
        current_tokens = board_tokens.unsqueeze(0).repeat(candidate_from.shape[0], 1)
        castle_index = self._castle_index_from_move(candidate_from, candidate_to)
        safe_castle_index = castle_index.clamp(0, 3)
        transit = self.castle_transit[safe_castle_index]
        transit_tokens = self._apply_king_transit_batch(board_tokens, candidate_from, transit)
        from_attacked = self.attack_test(current_tokens, candidate_from, enemy_side)
        transit_attacked = self.attack_test(transit_tokens, transit, enemy_side)
        castle_path_ok = torch.logical_or(torch.logical_not(castle_move), torch.logical_and(torch.logical_not(from_attacked), torch.logical_not(transit_attacked)))
        return torch.logical_and(torch.logical_and(king_present, torch.logical_not(final_attacked)), castle_path_ok)

    def make_move(self, board: TensorBoardState, move_tensor: torch.Tensor) -> TensorBoardState:
        move = move_tensor.to(torch.long).reshape(1, 4)
        from_sq = move[:, 0]
        to_sq = move[:, 1]
        promo = move[:, 2]
        flags = move[:, 3]
        next_squares = self._apply_move_batch(board.squares, from_sq, to_sq, promo, flags)[0]
        moving_token = board.squares[from_sq][0]
        captured_token = board.squares[to_sq][0]
        moving_kind = self.piece_kind[moving_token]
        capture_flag = flags[0].bitwise_and(int(MoveFlag.CAPTURE)).ne(0)
        next_side = 1 - board.side_id
        next_castle = self._next_castle_mask(board.castle_mask, moving_token, from_sq[0], to_sq[0], captured_token)
        double_pawn = torch.logical_and(moving_kind.eq(PAWN_KIND), torch.abs(to_sq[0] - from_sq[0]).eq(16))
        next_ep = torch.where(double_pawn, (from_sq[0] + to_sq[0]) // 2, torch.tensor(NO_EP))
        zero_halfmove = torch.logical_or(moving_kind.eq(PAWN_KIND), capture_flag)
        next_halfmove = torch.where(zero_halfmove, torch.tensor(0), board.halfmove_clock + 1)
        next_fullmove = board.fullmove_number + board.side_id.eq(BLACK_ID).to(torch.long)
        return TensorBoardState(next_squares, next_side, next_castle, next_ep, next_halfmove, next_fullmove)

    def terminal_predicates(
        self,
        board: TensorBoardState,
        ply: torch.Tensor,
        repetition_count: torch.Tensor,
        adjudication_cap_reached: torch.Tensor,
    ) -> TensorTerminalStatus:
        candidates = self.legal_candidate_tensors(board)
        no_legal = torch.logical_not(candidates.legal.any())
        own_king = torch.where(board.side_id.eq(WHITE_ID), torch.tensor(WHITE_KING_TOKEN), torch.tensor(BLACK_KING_TOKEN))
        king_mask = board.squares.eq(own_king)
        king_square = king_mask.to(torch.long).argmax().reshape(1)
        in_check = torch.logical_and(king_mask.any(), self.attack_test(board.squares.reshape(1, 64), king_square, 1 - board.side_id)[0])
        mate_result = torch.where(board.side_id.eq(WHITE_ID), torch.tensor(int(ResultCode.BLACK_WIN)), torch.tensor(int(ResultCode.WHITE_WIN)))
        no_legal_result = torch.where(in_check, mate_result, torch.tensor(int(ResultCode.DRAW)))
        no_legal_reason = torch.where(in_check, torch.tensor(int(TerminalReason.CHECKMATE)), torch.tensor(int(TerminalReason.STALEMATE)))
        insufficient = self._insufficient_material(board.squares)
        base_result = torch.where(no_legal, no_legal_result, torch.where(insufficient, torch.tensor(int(ResultCode.DRAW)), torch.tensor(int(ResultCode.ONGOING))))
        base_reason = torch.where(no_legal, no_legal_reason, torch.where(insufficient, torch.tensor(int(TerminalReason.INSUFFICIENT_MATERIAL)), torch.tensor(int(TerminalReason.NONE))))
        fifty = board.halfmove_clock.ge(100)
        threefold = repetition_count.ge(3)
        cap = adjudication_cap_reached.to(torch.bool)
        result_after_fifty = torch.where(fifty, torch.tensor(int(ResultCode.DRAW)), base_result)
        reason_after_fifty = torch.where(fifty, torch.tensor(int(TerminalReason.FIFTY_MOVE)), base_reason)
        result_after_threefold = torch.where(threefold, torch.tensor(int(ResultCode.DRAW)), result_after_fifty)
        reason_after_threefold = torch.where(threefold, torch.tensor(int(TerminalReason.THREEFOLD)), reason_after_fifty)
        result = torch.where(cap, torch.tensor(int(ResultCode.DRAW)), result_after_threefold)
        reason = torch.where(cap, torch.tensor(int(TerminalReason.ADJUDICATION_CAP)), reason_after_threefold)
        return TensorTerminalStatus(result, reason)

    def legal_candidate_tensors(self, board: TensorBoardState) -> TensorCandidateSet:
        board_tokens = board.squares
        side_id = board.side_id
        from_kind, own_piece, target_enemy, target_available = self.piece_dispatch(board_tokens, side_id)
        target_tokens = board_tokens[self.candidate_to]
        target_empty = target_tokens.eq(EMPTY_TOKEN)
        clear_ray = self.ray_scan(board_tokens)
        white_to_move = side_id.eq(WHITE_ID)
        start_rank = torch.where(white_to_move, torch.tensor(1), torch.tensor(6))
        last_from_rank = torch.where(white_to_move, torch.tensor(6), torch.tensor(1))
        one_step = torch.where(white_to_move, self.candidate_from + 8, self.candidate_from - 8)
        two_step = torch.where(white_to_move, self.candidate_from + 16, self.candidate_from - 16)
        mid_step = one_step.clamp(0, 63)
        diagonal_white = self.pawn_attack_bool[WHITE_ID, self.candidate_from, self.candidate_to].bool()
        diagonal_black = self.pawn_attack_bool[BLACK_ID, self.candidate_from, self.candidate_to].bool()
        diagonal = torch.where(white_to_move, diagonal_white, diagonal_black)
        ep_capture_square = torch.where(white_to_move, self.candidate_to - 8, self.candidate_to + 8).clamp(0, 63)
        ep_captured_token = board_tokens[ep_capture_square]
        ep_captured_color = self.piece_color[ep_captured_token]
        ep_capture = torch.logical_and(
            torch.logical_and(torch.logical_and(diagonal, self.candidate_to.eq(board.ep_square)), target_empty),
            torch.logical_and(board.ep_square.lt(NO_EP), torch.logical_and(self.piece_kind[ep_captured_token].eq(PAWN_KIND), ep_captured_color.eq(1 - side_id))),
        )
        reaches_last = self.candidate_from_rank.eq(last_from_rank)
        pawn_promo_ok = torch.where(reaches_last, self.candidate_promo.ne(0), self.candidate_promo.eq(0))
        pawn_quiet = torch.logical_or(
            torch.logical_and(self.candidate_to.eq(one_step), target_empty),
            torch.logical_and(
                torch.logical_and(self.candidate_from_rank.eq(start_rank), self.candidate_to.eq(two_step)),
                torch.logical_and(target_empty, board_tokens[mid_step].eq(EMPTY_TOKEN)),
            ),
        )
        pawn_capture = torch.logical_or(torch.logical_and(diagonal, target_enemy), ep_capture)
        pawn_move = torch.logical_and(torch.logical_and(from_kind.eq(PAWN_KIND), pawn_promo_ok), torch.logical_or(pawn_quiet, pawn_capture))
        promo_zero = self.candidate_promo.eq(0)
        knight_move = torch.logical_and(torch.logical_and(from_kind.eq(KNIGHT_KIND), promo_zero), torch.logical_and(self.knight_attack_bool[self.candidate_from, self.candidate_to].bool(), target_available))
        bishop_move = torch.logical_and(torch.logical_and(from_kind.eq(BISHOP_KIND), promo_zero), torch.logical_and(torch.logical_and(self.candidate_bishop_line.bool(), clear_ray), target_available))
        rook_move = torch.logical_and(torch.logical_and(from_kind.eq(ROOK_KIND), promo_zero), torch.logical_and(torch.logical_and(self.candidate_rook_line.bool(), clear_ray), target_available))
        queen_line = torch.logical_or(self.candidate_bishop_line.bool(), self.candidate_rook_line.bool())
        queen_move = torch.logical_and(torch.logical_and(from_kind.eq(QUEEN_KIND), promo_zero), torch.logical_and(torch.logical_and(queen_line, clear_ray), target_available))
        king_normal = torch.logical_and(torch.logical_and(from_kind.eq(KING_KIND), promo_zero), torch.logical_and(self.king_attack_bool[self.candidate_from, self.candidate_to].bool(), target_available))
        castle_index = self.candidate_castle_index
        safe_castle_index = castle_index.clamp(0, 3)
        castle_combo = castle_index.ge(0)
        castle_right = board.castle_mask.bitwise_and(self.castle_right_bits[safe_castle_index]).ne(0)
        castle_rook_present = board_tokens[self.castle_rook_from[safe_castle_index]].eq(self.castle_rook_token[safe_castle_index])
        castle_empty = torch.logical_not(torch.logical_and(self.castle_empty_mask[safe_castle_index].bool(), board_tokens.ne(EMPTY_TOKEN).unsqueeze(0)).any(dim=1))
        castle_move = torch.logical_and(torch.logical_and(torch.logical_and(torch.logical_and(from_kind.eq(KING_KIND), promo_zero), castle_combo), castle_right), torch.logical_and(castle_rook_present, castle_empty))
        pseudo = torch.logical_and(
            own_piece,
            torch.logical_or(
                torch.logical_or(torch.logical_or(pawn_move, knight_move), torch.logical_or(bishop_move, rook_move)),
                torch.logical_or(queen_move, torch.logical_or(king_normal, castle_move)),
            ),
        )
        capture_flag = torch.logical_or(target_enemy, ep_capture).to(torch.long) * int(MoveFlag.CAPTURE)
        ep_flag = ep_capture.to(torch.long) * int(MoveFlag.EP)
        castle_flag = castle_move.to(torch.long) * int(MoveFlag.CASTLE)
        promotion_flag = torch.logical_and(from_kind.eq(PAWN_KIND), torch.logical_and(reaches_last, self.candidate_promo.ne(0))).to(torch.long) * int(MoveFlag.PROMOTION)
        flags = capture_flag + ep_flag + castle_flag + promotion_flag
        indices = torch.nonzero(pseudo, as_tuple=False).flatten()
        candidate_from = self.candidate_from[indices]
        candidate_to = self.candidate_to[indices]
        candidate_promo = self.candidate_promo[indices]
        candidate_flags = flags[indices]
        legal = self.legal_filter(board_tokens, side_id, candidate_from, candidate_to, candidate_promo, candidate_flags)
        return TensorCandidateSet(candidate_from, candidate_to, candidate_promo, candidate_flags, legal)

    def _apply_move_batch(
        self,
        board_tokens: torch.Tensor,
        from_sq: torch.Tensor,
        to_sq: torch.Tensor,
        promo: torch.Tensor,
        flags: torch.Tensor,
    ) -> torch.Tensor:
        batch = from_sq.shape[0]
        next_tokens = board_tokens.unsqueeze(0).repeat(batch, 1)
        moving_tokens = board_tokens[from_sq]
        moving_colors = self.piece_color[moving_tokens]
        next_tokens.scatter_(1, from_sq.reshape(batch, 1), torch.zeros((batch, 1), dtype=torch.long))
        ep_capture = flags.bitwise_and(int(MoveFlag.EP)).ne(0)
        ep_square = torch.where(moving_colors.eq(WHITE_ID), to_sq - 8, to_sq + 8).clamp(0, 63)
        next_tokens.scatter_(1, ep_square.reshape(batch, 1), torch.where(ep_capture, torch.zeros(batch, dtype=torch.long), next_tokens.gather(1, ep_square.reshape(batch, 1)).flatten()).reshape(batch, 1))
        castle = flags.bitwise_and(int(MoveFlag.CASTLE)).ne(0)
        castle_index = self._castle_index_from_move(from_sq, to_sq).clamp(0, 3)
        rook_from = self.castle_rook_from[castle_index]
        rook_to = self.castle_rook_to[castle_index]
        rook_tokens = next_tokens.gather(1, rook_from.reshape(batch, 1)).flatten()
        next_tokens.scatter_(1, rook_to.reshape(batch, 1), torch.where(castle, rook_tokens, next_tokens.gather(1, rook_to.reshape(batch, 1)).flatten()).reshape(batch, 1))
        next_tokens.scatter_(1, rook_from.reshape(batch, 1), torch.where(castle, torch.zeros(batch, dtype=torch.long), next_tokens.gather(1, rook_from.reshape(batch, 1)).flatten()).reshape(batch, 1))
        promo_token = torch.where(moving_colors.eq(WHITE_ID), promo + 1, promo + 7)
        placed_token = torch.where(promo.ne(0), promo_token, moving_tokens)
        next_tokens.scatter_(1, to_sq.reshape(batch, 1), placed_token.reshape(batch, 1))
        return next_tokens

    def _apply_king_transit_batch(self, board_tokens: torch.Tensor, from_sq: torch.Tensor, transit: torch.Tensor) -> torch.Tensor:
        batch = from_sq.shape[0]
        transit_tokens = board_tokens.unsqueeze(0).repeat(batch, 1)
        king_tokens = board_tokens[from_sq]
        transit_tokens.scatter_(1, from_sq.reshape(batch, 1), torch.zeros((batch, 1), dtype=torch.long))
        transit_tokens.scatter_(1, transit.reshape(batch, 1), king_tokens.reshape(batch, 1))
        return transit_tokens

    def _castle_index_from_move(self, from_sq: torch.Tensor, to_sq: torch.Tensor) -> torch.Tensor:
        index = torch.full_like(from_sq, -1)
        index = torch.where(torch.logical_and(from_sq.eq(4), to_sq.eq(6)), torch.zeros_like(index), index)
        index = torch.where(torch.logical_and(from_sq.eq(4), to_sq.eq(2)), torch.ones_like(index), index)
        index = torch.where(torch.logical_and(from_sq.eq(60), to_sq.eq(62)), torch.full_like(index, 2), index)
        return torch.where(torch.logical_and(from_sq.eq(60), to_sq.eq(58)), torch.full_like(index, 3), index)

    def _next_castle_mask(self, castle_mask: torch.Tensor, moving_token: torch.Tensor, from_sq: torch.Tensor, to_sq: torch.Tensor, captured_token: torch.Tensor) -> torch.Tensor:
        white_king_moved = moving_token.eq(WHITE_KING_TOKEN)
        black_king_moved = moving_token.eq(BLACK_KING_TOKEN)
        after_kings = castle_mask.bitwise_and(torch.where(white_king_moved, torch.tensor(~3), torch.tensor(-1))).bitwise_and(torch.where(black_king_moved, torch.tensor(~12), torch.tensor(-1)))
        home_from = self.rook_home_squares.eq(from_sq)
        home_to = self.rook_home_squares.eq(to_sq)
        captured_home_rook = torch.logical_and(captured_token.ne(EMPTY_TOKEN), home_to)
        touched = torch.logical_or(torch.logical_or(home_from, home_to), captured_home_rook)
        removed_bits = (self.rook_home_right_bits * touched.to(torch.long)).sum()
        return after_kings.bitwise_and(15 - removed_bits)

    def _insufficient_material(self, board_tokens: torch.Tensor) -> torch.Tensor:
        kinds = self.piece_kind[board_tokens]
        non_king_piece = torch.logical_and(board_tokens.ne(EMPTY_TOKEN), kinds.ne(KING_KIND))
        minor_piece = torch.logical_or(kinds.eq(BISHOP_KIND), kinds.eq(KNIGHT_KIND))
        pawn_rook_queen = torch.logical_or(kinds.eq(PAWN_KIND), torch.logical_or(kinds.eq(ROOK_KIND), kinds.eq(QUEEN_KIND)))
        piece_count = non_king_piece.to(torch.long).sum()
        bishop_mask = torch.logical_and(non_king_piece, kinds.eq(BISHOP_KIND))
        knight_mask = torch.logical_and(non_king_piece, kinds.eq(KNIGHT_KIND))
        bishop_count = bishop_mask.to(torch.long).sum()
        knight_count = knight_mask.to(torch.long).sum()
        any_major_or_pawn = torch.logical_and(non_king_piece, pawn_rook_queen).any()
        only_kings = piece_count.eq(0)
        one_minor = torch.logical_and(piece_count.eq(1), torch.logical_and(non_king_piece, minor_piece).any())
        bishop_colors_present = self.bishop_square_color[bishop_mask].unique().numel()
        same_color_bishops = torch.logical_and(piece_count.eq(bishop_count), torch.tensor(bishop_colors_present <= 1))
        only_small_knights = torch.logical_and(piece_count.eq(knight_count), knight_count.le(2))
        return torch.logical_and(torch.logical_not(any_major_or_pawn), torch.logical_or(torch.logical_or(only_kings, one_minor), torch.logical_or(same_color_bishops, only_small_knights)))


def _candidate_universe() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from_sq: list[int] = []
    to_sq: list[int] = []
    promo: list[int] = []
    for source in range(64):
        for target in range(64):
            for promotion in range(5):
                from_sq.append(source)
                to_sq.append(target)
                promo.append(promotion)
    return torch.tensor(from_sq, dtype=torch.long), torch.tensor(to_sq, dtype=torch.long), torch.tensor(promo, dtype=torch.long)


def _piece_kind_table() -> torch.Tensor:
    table = torch.zeros(13, dtype=torch.long)
    table[1:7] = torch.arange(1, 7, dtype=torch.long)
    table[7:13] = torch.arange(1, 7, dtype=torch.long)
    return table


def _piece_color_table() -> torch.Tensor:
    table = torch.full((13,), -1, dtype=torch.long)
    table[1:7] = WHITE_ID
    table[7:13] = BLACK_ID
    return table


def _leaper_bool(deltas: tuple[tuple[int, int], ...]) -> torch.Tensor:
    table = torch.zeros((64, 64), dtype=torch.int8)
    for square in range(64):
        file_index = square % 8
        rank = square // 8
        for df, dr in deltas:
            target_file = file_index + df
            target_rank = rank + dr
            if 0 <= target_file < 8 and 0 <= target_rank < 8:
                table[square, target_rank * 8 + target_file] = 1
    return table


def _pawn_attack_bool() -> torch.Tensor:
    table = torch.zeros((2, 64, 64), dtype=torch.int8)
    for square in range(64):
        file_index = square % 8
        rank = square // 8
        for color_id, direction in ((WHITE_ID, 1), (BLACK_ID, -1)):
            target_rank = rank + direction
            for file_delta in (-1, 1):
                target_file = file_index + file_delta
                if 0 <= target_file < 8 and 0 <= target_rank < 8:
                    table[color_id, square, target_rank * 8 + target_file] = 1
    return table


def _line_tables() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bishop = torch.zeros((64, 64), dtype=torch.int8)
    rook = torch.zeros((64, 64), dtype=torch.int8)
    between = torch.zeros((64, 64, 64), dtype=torch.int8)
    for source in range(64):
        source_file = source % 8
        source_rank = source // 8
        for target in range(64):
            target_file = target % 8
            target_rank = target // 8
            df = target_file - source_file
            dr = target_rank - source_rank
            step_file = 0 if df == 0 else (1 if df > 0 else -1)
            step_rank = 0 if dr == 0 else (1 if dr > 0 else -1)
            rook_line = (df == 0) ^ (dr == 0)
            bishop_line = abs(df) == abs(dr) and df != 0
            if rook_line or bishop_line:
                rook[source, target] = int(rook_line)
                bishop[source, target] = int(bishop_line)
                current_file = source_file + step_file
                current_rank = source_rank + step_rank
                while current_file != target_file or current_rank != target_rank:
                    between[source, target, current_rank * 8 + current_file] = 1
                    current_file += step_file
                    current_rank += step_rank
    return bishop, rook, between


def _candidate_castle_index(candidate_from: torch.Tensor, candidate_to: torch.Tensor) -> torch.Tensor:
    index = torch.full_like(candidate_from, -1)
    index = torch.where(torch.logical_and(candidate_from.eq(4), candidate_to.eq(6)), torch.zeros_like(index), index)
    index = torch.where(torch.logical_and(candidate_from.eq(4), candidate_to.eq(2)), torch.ones_like(index), index)
    index = torch.where(torch.logical_and(candidate_from.eq(60), candidate_to.eq(62)), torch.full_like(index, 2), index)
    return torch.where(torch.logical_and(candidate_from.eq(60), candidate_to.eq(58)), torch.full_like(index, 3), index)


def _castle_empty_mask() -> torch.Tensor:
    mask = torch.zeros((4, 64), dtype=torch.int8)
    for index, squares in enumerate(((5, 6), (3, 2, 1), (61, 62), (59, 58, 57))):
        for square in squares:
            mask[index, square] = 1
    return mask


def _castle_mask_from_string(castling: str) -> int:
    mask = 0
    for symbol, bit in (("K", 1), ("Q", 2), ("k", 4), ("q", 8)):
        if symbol in castling:
            mask |= bit
    return mask


def _frozen(tensor: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(tensor, requires_grad=False)
