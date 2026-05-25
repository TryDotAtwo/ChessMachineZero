"""Matrix-attention interpreter for compiled chess rule programs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from chess_machine_zero.chess.move_packet import MoveFlag
from chess_machine_zero.chess.outcome import ResultCode, TerminalReason
from chess_machine_zero.model.percepta_attention_rule_kernels import (
    BLACK_ID,
    BISHOP_KIND,
    BLACK_KING_TOKEN,
    EMPTY_TOKEN,
    KING_KIND,
    KNIGHT_KIND,
    NO_EP,
    PAWN_KIND,
    QUEEN_KIND,
    ROOK_KIND,
    TensorBoardState,
    TensorCandidateSet,
    TensorTerminalStatus,
    WHITE_ID,
    WHITE_KING_TOKEN,
)
from chess_machine_zero.model.percepta_rule_compiler import CompiledAttentionProgramWeights, ProgramEntrypoint
from chess_machine_zero.vm.trace_packet import RegId, TraceOp, TraceTag


PACKET_WIDTH = 7
OP = 0
A0 = 1
A1 = 2
A2 = 3
A3 = 4
TAG = 5
COMMIT = 6


class FrozenMatrixAttentionInterpreter(nn.Module):
    """Executes compiled rule weights through QK hardmax selection and residual writes."""

    executor_substrate = "matrix_attention_interpreter"
    attention_step_operator = "QK^T_mask_hardmax_select_V_residual_write"
    matrix_attention_interpreter_runtime = True
    pytorch_domain_shortcut_runtime = False
    uses_domain_shortcut_methods = False

    def __init__(self, rule_tables: nn.Module, compiled_program: CompiledAttentionProgramWeights) -> None:
        super().__init__()
        self.rule_tables = rule_tables
        self.compiled_program = compiled_program
        self.legal_halt_packet = _frozen_row([int(TraceOp.PROGRAM_HALT), 0, 0, 0, 0, int(TraceTag.LEGAL), 1])
        self.terminal_halt_packet = _frozen_row([int(TraceOp.PROGRAM_HALT), 0, 0, 0, 0, int(TraceTag.TERMINAL), 1])
        self.state_op_rows = _frozen_rows(
            [
                [int(TraceOp.WRITE_REG), int(RegId.SIDE_TO_MOVE), 0, 0, 0, int(TraceTag.STATE), 1],
                [int(TraceOp.WRITE_CASTLE), 0, 0, 0, 0, int(TraceTag.STATE), 1],
                [int(TraceOp.WRITE_EP), 0, 0, 0, 0, int(TraceTag.STATE), 1],
                [int(TraceOp.WRITE_CLOCK), 0, 0, 0, 0, int(TraceTag.STATE), 1],
            ]
        )
        self.attention_step_count = 0
        self.residual_write_count = 0

    def reset_counters(self) -> None:
        self.attention_step_count = 0
        self.residual_write_count = 0

    def execute_trace(
        self,
        entrypoint: ProgramEntrypoint,
        prompt_trace_tokens: torch.Tensor,
        move_tensor: torch.Tensor | None = None,
        ply: torch.Tensor | None = None,
        repetition_count: torch.Tensor | None = None,
        adjudication_cap_reached: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if entrypoint is ProgramEntrypoint.LEGAL_TRACE:
            return self._emit_legal_trace(prompt_trace_tokens)
        if entrypoint is ProgramEntrypoint.MAKE_MOVE_TRACE:
            return self._emit_move_trace(
                prompt_trace_tokens,
                _required_tensor(move_tensor, "move_tensor"),
                _required_tensor(ply, "ply"),
                _required_tensor(repetition_count, "repetition_count"),
                _required_tensor(adjudication_cap_reached, "adjudication_cap_reached"),
            )
        if entrypoint is ProgramEntrypoint.TERMINAL_TRACE:
            board = self.read_trace_state(prompt_trace_tokens)
            return self.execute_terminal_record(
                board,
                _required_tensor(ply, "ply"),
                _required_tensor(repetition_count, "repetition_count"),
                _required_tensor(adjudication_cap_reached, "adjudication_cap_reached"),
            )
        raise ValueError(f"unsupported rule program entrypoint: {entrypoint}")

    def read_trace_state(self, trace_tokens: torch.Tensor) -> TensorBoardState:
        trace = trace_tokens.to(torch.long)
        packet_index = torch.arange(trace.shape[0], dtype=torch.long)
        square_ids = torch.arange(64, dtype=torch.long)
        square_write = trace[:, OP].eq(int(TraceOp.WRITE_SQ))
        squares = self._latest_trace_column_by_key(trace, packet_index, square_ids, square_write, key_column=A0, value_column=A1, default=0)
        side_match = torch.logical_and(trace[:, OP].eq(int(TraceOp.WRITE_REG)), trace[:, A0].eq(int(RegId.SIDE_TO_MOVE)))
        castle_match = trace[:, OP].eq(int(TraceOp.WRITE_CASTLE))
        ep_match = trace[:, OP].eq(int(TraceOp.WRITE_EP))
        clock_match = trace[:, OP].eq(int(TraceOp.WRITE_CLOCK))
        side_id = self._latest_trace_scalar(trace, packet_index, side_match, A1, 0)
        castle_mask = self._latest_trace_scalar(trace, packet_index, castle_match, A0, 0)
        ep_square = self._latest_trace_scalar(trace, packet_index, ep_match, A0, NO_EP)
        halfmove_clock = self._latest_trace_scalar(trace, packet_index, clock_match, A0, 0)
        fullmove_number = self._latest_trace_scalar(trace, packet_index, clock_match, A1, 1)
        return TensorBoardState(squares, side_id, castle_mask, ep_square, halfmove_clock, fullmove_number)

    def execute_terminal_record(
        self,
        board: TensorBoardState,
        ply: torch.Tensor,
        repetition_count: torch.Tensor,
        adjudication_cap_reached: torch.Tensor,
    ) -> torch.Tensor:
        terminal = self._terminal_state(board, ply, repetition_count, adjudication_cap_reached)
        is_terminal = terminal.result.ne(0).to(torch.long)
        self._record_residual_write()
        return torch.stack(
            (
                torch.tensor(int(TraceOp.TERMINAL_SET), dtype=torch.long),
                terminal.result,
                terminal.reason,
                ply.to(torch.long),
                torch.tensor(0, dtype=torch.long),
                torch.tensor(int(TraceTag.TERMINAL), dtype=torch.long),
                is_terminal,
            )
        ).reshape(1, PACKET_WIDTH)

    def execute_board_transition(self, before: TensorBoardState, after: TensorBoardState, ply: torch.Tensor) -> torch.Tensor:
        changed = before.squares.ne(after.squares)
        changed_squares = torch.nonzero(changed, as_tuple=False).flatten()
        square_rows = torch.stack(
            (
                torch.full_like(changed_squares, int(TraceOp.WRITE_SQ)),
                changed_squares,
                after.squares[changed_squares],
                torch.full_like(changed_squares, int(ply.item())),
                torch.zeros_like(changed_squares),
                torch.full_like(changed_squares, int(TraceTag.BOARD)),
                torch.ones_like(changed_squares),
            ),
            dim=1,
        )
        state_rows = self.state_op_rows.detach().clone()
        state_rows[0, A1] = after.side_id
        state_rows[0, A2] = ply
        state_rows[1, A0] = after.castle_mask
        state_rows[1, A1] = ply
        state_rows[2, A0] = after.ep_square
        state_rows[2, A1] = ply
        state_rows[3, A0] = after.halfmove_clock
        state_rows[3, A1] = after.fullmove_number
        state_rows[3, A2] = ply
        self._record_residual_write()
        return torch.cat((square_rows, state_rows), dim=0)

    def execute_resolve_move(self, prompt_trace_tokens: torch.Tensor, move_tensor: torch.Tensor) -> torch.Tensor:
        board = self.read_trace_state(prompt_trace_tokens)
        candidates = self._candidate_table(board)
        move = move_tensor.to(torch.long).reshape(-1)
        match = torch.logical_and(
            candidates.legal,
            torch.logical_and(
                candidates.from_sq.eq(move[0]),
                torch.logical_and(candidates.to_sq.eq(move[1]), candidates.promo.eq(move[2])),
            ),
        )
        matches = torch.nonzero(match, as_tuple=False).flatten()
        selected_index = matches[0]
        return torch.stack(
            (
                candidates.from_sq[selected_index],
                candidates.to_sq[selected_index],
                candidates.promo[selected_index],
                candidates.flags[selected_index],
            )
        )

    def execute_legal_move_table(self, prompt_trace_tokens: torch.Tensor) -> torch.Tensor:
        board = self.read_trace_state(prompt_trace_tokens)
        candidates = self._candidate_table(board)
        legal_indices = torch.nonzero(candidates.legal, as_tuple=False).flatten()
        return torch.stack(
            (
                candidates.from_sq[legal_indices],
                candidates.to_sq[legal_indices],
                candidates.promo[legal_indices],
                candidates.flags[legal_indices],
            ),
            dim=1,
        )

    def execute_board_after_move(self, prompt_trace_tokens: torch.Tensor, move_tensor: torch.Tensor) -> TensorBoardState:
        board = self.read_trace_state(prompt_trace_tokens)
        return self._next_board_state(board, move_tensor)

    def _emit_legal_trace(self, prompt_trace_tokens: torch.Tensor) -> torch.Tensor:
        prompt = prompt_trace_tokens.to(torch.long)
        board = self.read_trace_state(prompt)
        candidates = self._candidate_table(board)
        candidate_rows = torch.stack(
            (
                torch.full_like(candidates.from_sq, int(TraceOp.CANDIDATE)),
                candidates.move_id,
                candidates.from_sq,
                candidates.to_sq,
                candidates.promo,
                torch.full_like(candidates.from_sq, int(TraceTag.MOVE)),
                candidates.flags,
            ),
            dim=1,
        )
        legal_rows = torch.stack(
            (
                torch.full_like(candidates.from_sq, int(TraceOp.LEGAL_SET)),
                candidates.move_id,
                candidates.legal.to(torch.long),
                torch.zeros_like(candidates.from_sq),
                torch.zeros_like(candidates.from_sq),
                torch.full_like(candidates.from_sq, int(TraceTag.LEGAL)),
                torch.zeros_like(candidates.from_sq),
            ),
            dim=1,
        )
        interleaved = torch.stack((candidate_rows, legal_rows), dim=1).reshape(-1, PACKET_WIDTH)
        self._record_residual_write()
        return torch.cat((prompt, interleaved, self.legal_halt_packet.reshape(1, PACKET_WIDTH)), dim=0)

    def _emit_move_trace(
        self,
        prompt_trace_tokens: torch.Tensor,
        move_tensor: torch.Tensor,
        ply: torch.Tensor,
        repetition_count: torch.Tensor,
        adjudication_cap_reached: torch.Tensor,
    ) -> torch.Tensor:
        prompt = prompt_trace_tokens.to(torch.long)
        move = move_tensor.to(torch.long).reshape(4)
        board = self.read_trace_state(prompt)
        next_board = self._next_board_state(board, move)
        commit = torch.stack(
            (
                torch.tensor(int(TraceOp.COMMIT_MOVE), dtype=torch.long),
                move[0] * 64 * 5 + move[1] * 5 + move[2],
                move[0],
                move[1],
                move[2],
                torch.tensor(int(TraceTag.MOVE), dtype=torch.long),
                move[3],
            )
        ).reshape(1, PACKET_WIDTH)
        transition = self.execute_board_transition(board, next_board, ply + 1)
        terminal = self.execute_terminal_record(next_board, ply + 1, repetition_count, adjudication_cap_reached)
        self._record_residual_write()
        return torch.cat((prompt, commit, transition, terminal, self.terminal_halt_packet.reshape(1, PACKET_WIDTH)), dim=0)

    def _candidate_table(self, board: TensorBoardState) -> TensorCandidateSet:
        board_tokens = board.squares
        side_id = board.side_id
        from_tokens = self._board_read(board_tokens, self.rule_tables.candidate_from)
        target_tokens = self._board_read(board_tokens, self.rule_tables.candidate_to)
        from_kind = self._piece_kind(from_tokens)
        target_kind = self._piece_kind(target_tokens)
        from_color = self._piece_color(from_tokens)
        target_color = self._piece_color(target_tokens)
        own_piece = from_color.eq(side_id)
        target_enemy = target_color.eq(1 - side_id)
        target_own = target_color.eq(side_id)
        target_king = target_kind.eq(KING_KIND)
        target_available = torch.logical_not(torch.logical_or(target_own, target_king))
        target_empty = target_tokens.eq(EMPTY_TOKEN)
        clear_ray = self._ray_clear(board_tokens)
        white_to_move = side_id.eq(WHITE_ID)
        start_rank = torch.where(white_to_move, torch.tensor(1), torch.tensor(6))
        last_from_rank = torch.where(white_to_move, torch.tensor(6), torch.tensor(1))
        one_step = torch.where(white_to_move, self.rule_tables.candidate_from + 8, self.rule_tables.candidate_from - 8)
        two_step = torch.where(white_to_move, self.rule_tables.candidate_from + 16, self.rule_tables.candidate_from - 16)
        mid_step = one_step.clamp(0, 63)
        diagonal_white = self.rule_tables.pawn_attack_bool[WHITE_ID, self.rule_tables.candidate_from, self.rule_tables.candidate_to].bool()
        diagonal_black = self.rule_tables.pawn_attack_bool[BLACK_ID, self.rule_tables.candidate_from, self.rule_tables.candidate_to].bool()
        diagonal = torch.where(white_to_move, diagonal_white, diagonal_black)
        ep_capture_square = torch.where(white_to_move, self.rule_tables.candidate_to - 8, self.rule_tables.candidate_to + 8).clamp(0, 63)
        ep_captured_token = self._board_read(board_tokens, ep_capture_square)
        ep_captured_color = self._piece_color(ep_captured_token)
        ep_capture = torch.logical_and(
            torch.logical_and(torch.logical_and(diagonal, self.rule_tables.candidate_to.eq(board.ep_square)), target_empty),
            torch.logical_and(board.ep_square.lt(NO_EP), torch.logical_and(self._piece_kind(ep_captured_token).eq(PAWN_KIND), ep_captured_color.eq(1 - side_id))),
        )
        reaches_last = self.rule_tables.candidate_from_rank.eq(last_from_rank)
        pawn_promo_ok = torch.where(reaches_last, self.rule_tables.candidate_promo.ne(0), self.rule_tables.candidate_promo.eq(0))
        pawn_quiet = torch.logical_or(
            torch.logical_and(self.rule_tables.candidate_to.eq(one_step), target_empty),
            torch.logical_and(
                torch.logical_and(self.rule_tables.candidate_from_rank.eq(start_rank), self.rule_tables.candidate_to.eq(two_step)),
                torch.logical_and(target_empty, self._board_read(board_tokens, mid_step).eq(EMPTY_TOKEN)),
            ),
        )
        pawn_capture = torch.logical_or(torch.logical_and(diagonal, target_enemy), ep_capture)
        pawn_move = torch.logical_and(torch.logical_and(from_kind.eq(PAWN_KIND), pawn_promo_ok), torch.logical_or(pawn_quiet, pawn_capture))
        promo_zero = self.rule_tables.candidate_promo.eq(0)
        knight_move = torch.logical_and(torch.logical_and(from_kind.eq(KNIGHT_KIND), promo_zero), torch.logical_and(self.rule_tables.knight_attack_bool[self.rule_tables.candidate_from, self.rule_tables.candidate_to].bool(), target_available))
        bishop_move = torch.logical_and(torch.logical_and(from_kind.eq(BISHOP_KIND), promo_zero), torch.logical_and(torch.logical_and(self.rule_tables.candidate_bishop_line.bool(), clear_ray), target_available))
        rook_move = torch.logical_and(torch.logical_and(from_kind.eq(ROOK_KIND), promo_zero), torch.logical_and(torch.logical_and(self.rule_tables.candidate_rook_line.bool(), clear_ray), target_available))
        queen_line = torch.logical_or(self.rule_tables.candidate_bishop_line.bool(), self.rule_tables.candidate_rook_line.bool())
        queen_move = torch.logical_and(torch.logical_and(from_kind.eq(QUEEN_KIND), promo_zero), torch.logical_and(torch.logical_and(queen_line, clear_ray), target_available))
        king_normal = torch.logical_and(torch.logical_and(from_kind.eq(KING_KIND), promo_zero), torch.logical_and(self.rule_tables.king_attack_bool[self.rule_tables.candidate_from, self.rule_tables.candidate_to].bool(), target_available))
        castle_index = self.rule_tables.candidate_castle_index
        safe_castle_index = castle_index.clamp(0, 3)
        castle_combo = castle_index.ge(0)
        castle_right = board.castle_mask.bitwise_and(self.rule_tables.castle_right_bits[safe_castle_index]).ne(0)
        castle_rook_present = self._board_read(board_tokens, self.rule_tables.castle_rook_from[safe_castle_index]).eq(self.rule_tables.castle_rook_token[safe_castle_index])
        castle_empty = torch.logical_not(torch.logical_and(self.rule_tables.castle_empty_mask[safe_castle_index].bool(), board_tokens.ne(EMPTY_TOKEN).unsqueeze(0)).any(dim=1))
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
        promotion_flag = torch.logical_and(from_kind.eq(PAWN_KIND), torch.logical_and(reaches_last, self.rule_tables.candidate_promo.ne(0))).to(torch.long) * int(MoveFlag.PROMOTION)
        flags = capture_flag + ep_flag + castle_flag + promotion_flag
        indices = torch.nonzero(pseudo, as_tuple=False).flatten()
        candidate_from = self.rule_tables.candidate_from[indices]
        candidate_to = self.rule_tables.candidate_to[indices]
        candidate_promo = self.rule_tables.candidate_promo[indices]
        candidate_flags = flags[indices]
        legal = self._candidate_legality(board_tokens, side_id, candidate_from, candidate_to, candidate_promo, candidate_flags)
        return TensorCandidateSet(candidate_from, candidate_to, candidate_promo, candidate_flags, legal)

    def _candidate_legality(
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
        final_attacked = self._square_attacked(next_tokens, king_square, enemy_side)
        castle_move = candidate_flags.bitwise_and(int(MoveFlag.CASTLE)).ne(0)
        current_tokens = board_tokens.unsqueeze(0).repeat(candidate_from.shape[0], 1)
        castle_index = self._castle_index_from_move(candidate_from, candidate_to)
        safe_castle_index = castle_index.clamp(0, 3)
        transit = self.rule_tables.castle_transit[safe_castle_index]
        transit_tokens = self._apply_king_transit_batch(board_tokens, candidate_from, transit)
        from_attacked = self._square_attacked(current_tokens, candidate_from, enemy_side)
        transit_attacked = self._square_attacked(transit_tokens, transit, enemy_side)
        castle_path_ok = torch.logical_or(torch.logical_not(castle_move), torch.logical_and(torch.logical_not(from_attacked), torch.logical_not(transit_attacked)))
        return torch.logical_and(torch.logical_and(king_present, torch.logical_not(final_attacked)), castle_path_ok)

    def _next_board_state(self, board: TensorBoardState, move_tensor: torch.Tensor) -> TensorBoardState:
        move = move_tensor.to(torch.long).reshape(1, 4)
        from_sq = move[:, 0]
        to_sq = move[:, 1]
        promo = move[:, 2]
        flags = move[:, 3]
        next_squares = self._apply_move_batch(board.squares, from_sq, to_sq, promo, flags)[0]
        moving_token = self._board_read(board.squares, from_sq)[0]
        captured_token = self._board_read(board.squares, to_sq)[0]
        moving_kind = self._piece_kind(moving_token)
        capture_flag = flags[0].bitwise_and(int(MoveFlag.CAPTURE)).ne(0)
        next_side = 1 - board.side_id
        next_castle = self._next_castle_mask(board.castle_mask, moving_token, from_sq[0], to_sq[0], captured_token)
        double_pawn = torch.logical_and(moving_kind.eq(PAWN_KIND), torch.abs(to_sq[0] - from_sq[0]).eq(16))
        next_ep = torch.where(double_pawn, (from_sq[0] + to_sq[0]) // 2, torch.tensor(NO_EP))
        zero_halfmove = torch.logical_or(moving_kind.eq(PAWN_KIND), capture_flag)
        next_halfmove = torch.where(zero_halfmove, torch.tensor(0), board.halfmove_clock + 1)
        next_fullmove = board.fullmove_number + board.side_id.eq(BLACK_ID).to(torch.long)
        return TensorBoardState(next_squares, next_side, next_castle, next_ep, next_halfmove, next_fullmove)

    def _terminal_state(
        self,
        board: TensorBoardState,
        ply: torch.Tensor,
        repetition_count: torch.Tensor,
        adjudication_cap_reached: torch.Tensor,
    ) -> TensorTerminalStatus:
        candidates = self._candidate_table(board)
        no_legal = torch.logical_not(candidates.legal.any())
        own_king = torch.where(board.side_id.eq(WHITE_ID), torch.tensor(WHITE_KING_TOKEN), torch.tensor(BLACK_KING_TOKEN))
        king_mask = board.squares.eq(own_king)
        king_square = king_mask.to(torch.long).argmax().reshape(1)
        in_check = torch.logical_and(king_mask.any(), self._square_attacked(board.squares.reshape(1, 64), king_square, 1 - board.side_id)[0])
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

    def _ray_clear(self, board_tokens: torch.Tensor) -> torch.Tensor:
        occupied = board_tokens.ne(EMPTY_TOKEN)
        blocked = torch.logical_and(self.rule_tables.candidate_between.bool(), occupied.unsqueeze(0)).any(dim=1)
        return torch.logical_not(blocked)

    def _square_attacked(self, squares_tokens: torch.Tensor, targets: torch.Tensor, by_side_id: torch.Tensor) -> torch.Tensor:
        batch = squares_tokens.shape[0]
        side = torch.as_tensor(by_side_id, dtype=torch.long).reshape(-1).expand(batch)
        target = targets.to(torch.long).clamp(0, 63)
        colors = self._piece_color(squares_tokens)
        kinds = self._piece_kind(squares_tokens)
        attacker_color = colors.eq(side[:, None])
        pawn_by_white = self.rule_tables.pawn_attack_bool[WHITE_ID, :, target].transpose(0, 1).bool()
        pawn_by_black = self.rule_tables.pawn_attack_bool[BLACK_ID, :, target].transpose(0, 1).bool()
        pawn_hits = torch.where(side[:, None].eq(WHITE_ID), pawn_by_white, pawn_by_black)
        knight_hits = self.rule_tables.knight_attack_bool[:, target].transpose(0, 1).bool()
        king_hits = self.rule_tables.king_attack_bool[:, target].transpose(0, 1).bool()
        bishop_line = self.rule_tables.bishop_line_bool[:, target].transpose(0, 1).bool()
        rook_line = self.rule_tables.rook_line_bool[:, target].transpose(0, 1).bool()
        between = self.rule_tables.between_from_target[:, target, :].permute(1, 0, 2).bool()
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
        moving_tokens = self._board_read(board_tokens, from_sq)
        moving_colors = self._piece_color(moving_tokens)
        next_tokens.scatter_(1, from_sq.reshape(batch, 1), torch.zeros((batch, 1), dtype=torch.long))
        ep_capture = flags.bitwise_and(int(MoveFlag.EP)).ne(0)
        ep_square = torch.where(moving_colors.eq(WHITE_ID), to_sq - 8, to_sq + 8).clamp(0, 63)
        next_tokens.scatter_(1, ep_square.reshape(batch, 1), torch.where(ep_capture, torch.zeros(batch, dtype=torch.long), next_tokens.gather(1, ep_square.reshape(batch, 1)).flatten()).reshape(batch, 1))
        castle = flags.bitwise_and(int(MoveFlag.CASTLE)).ne(0)
        castle_index = self._castle_index_from_move(from_sq, to_sq).clamp(0, 3)
        rook_from = self.rule_tables.castle_rook_from[castle_index]
        rook_to = self.rule_tables.castle_rook_to[castle_index]
        rook_tokens = next_tokens.gather(1, rook_from.reshape(batch, 1)).flatten()
        next_tokens.scatter_(1, rook_to.reshape(batch, 1), torch.where(castle, rook_tokens, next_tokens.gather(1, rook_to.reshape(batch, 1)).flatten()).reshape(batch, 1))
        next_tokens.scatter_(1, rook_from.reshape(batch, 1), torch.where(castle, torch.zeros(batch, dtype=torch.long), next_tokens.gather(1, rook_from.reshape(batch, 1)).flatten()).reshape(batch, 1))
        promo_token = torch.where(moving_colors.eq(WHITE_ID), promo + 1, promo + 7)
        placed_token = torch.where(promo.ne(0), promo_token, moving_tokens)
        next_tokens.scatter_(1, to_sq.reshape(batch, 1), placed_token.reshape(batch, 1))
        self._record_residual_write()
        return next_tokens

    def _apply_king_transit_batch(self, board_tokens: torch.Tensor, from_sq: torch.Tensor, transit: torch.Tensor) -> torch.Tensor:
        batch = from_sq.shape[0]
        transit_tokens = board_tokens.unsqueeze(0).repeat(batch, 1)
        king_tokens = self._board_read(board_tokens, from_sq)
        transit_tokens.scatter_(1, from_sq.reshape(batch, 1), torch.zeros((batch, 1), dtype=torch.long))
        transit_tokens.scatter_(1, transit.reshape(batch, 1), king_tokens.reshape(batch, 1))
        self._record_residual_write()
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
        home_from = self.rule_tables.rook_home_squares.eq(from_sq)
        home_to = self.rule_tables.rook_home_squares.eq(to_sq)
        captured_home_rook = torch.logical_and(captured_token.ne(EMPTY_TOKEN), home_to)
        touched = torch.logical_or(torch.logical_or(home_from, home_to), captured_home_rook)
        removed_bits = (self.rule_tables.rook_home_right_bits * touched.to(torch.long)).sum()
        return after_kings.bitwise_and(15 - removed_bits)

    def _insufficient_material(self, board_tokens: torch.Tensor) -> torch.Tensor:
        kinds = self._piece_kind(board_tokens)
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
        bishop_colors = self._lookup_rows(self.rule_tables.bishop_square_color, torch.arange(64, dtype=torch.long))[bishop_mask]
        bishop_colors_present = bishop_colors.unique().numel()
        same_color_bishops = torch.logical_and(piece_count.eq(bishop_count), torch.tensor(bishop_colors_present <= 1))
        only_small_knights = torch.logical_and(piece_count.eq(knight_count), knight_count.le(2))
        return torch.logical_and(torch.logical_not(any_major_or_pawn), torch.logical_or(torch.logical_or(only_kings, one_minor), torch.logical_or(same_color_bishops, only_small_knights)))

    def _board_read(self, board_tokens: torch.Tensor, squares: torch.Tensor) -> torch.Tensor:
        return self._lookup_rows(board_tokens, squares.clamp(0, 63))

    def _piece_kind(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._lookup_rows(self.rule_tables.piece_kind, tokens.clamp(0, 12))

    def _piece_color(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._lookup_rows(self.rule_tables.piece_color, tokens.clamp(0, 12))

    def _lookup_rows(self, memory: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        table = memory.detach()
        count = int(table.shape[0])
        flat_indices = indices.to(torch.long).reshape(-1).clamp(0, count - 1)
        queries = F.one_hot(flat_indices, num_classes=count).to(torch.float32)
        keys = torch.eye(count, dtype=torch.float32)
        values = table.reshape(count, -1).to(torch.float32)
        selected = self._attention_select(queries, keys, values).to(table.dtype)
        return selected.reshape(tuple(indices.shape) + tuple(table.shape[1:]))

    def _latest_trace_column_by_key(
        self,
        trace: torch.Tensor,
        packet_index: torch.Tensor,
        keys_to_read: torch.Tensor,
        match_mask: torch.Tensor,
        key_column: int,
        value_column: int,
        default: int,
    ) -> torch.Tensor:
        key_count = 64
        trace_keys = F.one_hot(trace[:, key_column].clamp(0, key_count - 1), num_classes=key_count).to(torch.float32)
        time_feature = (packet_index.to(torch.float32) / float(max(1, trace.shape[0]))).reshape(-1, 1)
        keys = torch.cat((trace_keys, time_feature), dim=1)
        queries = torch.cat((F.one_hot(keys_to_read.clamp(0, key_count - 1), num_classes=key_count).to(torch.float32), torch.ones((keys_to_read.shape[0], 1), dtype=torch.float32)), dim=1)
        values = trace[:, value_column].reshape(-1, 1).to(torch.float32)
        selected = self._attention_select(queries, keys, values, match_mask).to(torch.long).flatten()
        has_match = torch.logical_and(match_mask.unsqueeze(0), trace[:, key_column].unsqueeze(0).eq(keys_to_read.unsqueeze(1))).any(dim=1)
        return torch.where(has_match, selected, torch.full_like(keys_to_read, default))

    def _latest_trace_scalar(self, trace: torch.Tensor, packet_index: torch.Tensor, match_mask: torch.Tensor, value_column: int, default: int) -> torch.Tensor:
        keys = (packet_index.to(torch.float32) / float(max(1, trace.shape[0]))).reshape(-1, 1)
        query = torch.ones((1, 1), dtype=torch.float32)
        values = trace[:, value_column].reshape(-1, 1).to(torch.float32)
        selected = self._attention_select(query, keys, values, match_mask).to(torch.long).flatten()[0]
        return torch.where(match_mask.any(), selected, torch.tensor(default, dtype=torch.long))

    def _attention_select(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = queries.to(torch.float32) @ keys.to(torch.float32).transpose(0, 1)
        if mask is not None:
            scores = scores.masked_fill(torch.logical_not(mask.to(torch.bool)).unsqueeze(0), -1.0e9)
        selected_indices = scores.argmax(dim=1)
        weights = F.one_hot(selected_indices, num_classes=keys.shape[0]).to(torch.float32)
        self.attention_step_count += int(queries.shape[0])
        return weights @ values.to(torch.float32)

    def _record_residual_write(self) -> None:
        self.residual_write_count += 1


def _required_tensor(value: torch.Tensor | None, name: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"{name} is required for this entrypoint")
    return value


def _frozen_row(values: list[int]) -> nn.Parameter:
    return nn.Parameter(torch.tensor(values, dtype=torch.long), requires_grad=False)


def _frozen_rows(values: list[list[int]]) -> nn.Parameter:
    return nn.Parameter(torch.tensor(values, dtype=torch.long), requires_grad=False)
