"""Host ChessMachineVM for deterministic trace generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from chess_machine_zero.chess.board_io import (
    NO_EP,
    BoardState,
    castling_mask,
    parse_fen,
    piece_from_token,
    piece_token,
)
from chess_machine_zero.chess.move_packet import MoveFlag, MovePacket, Promo
from chess_machine_zero.chess.outcome import ResultCode, TerminalReason, TerminalStatus
from chess_machine_zero.rng import DEFAULT_SEED
from chess_machine_zero.vm.chess_program import legal_generator_program, make_move_program, terminal_check_program
from chess_machine_zero.vm.trace_packet import RegId, TraceOp, TracePacket, TraceTag


WHITE = "w"
BLACK = "b"
KNIGHT_DELTAS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
KING_DELTAS = ((1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1))
BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
ROOK_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
QUEEN_DIRS = BISHOP_DIRS + ROOK_DIRS
PROMOTION_ORDER = (Promo.KNIGHT, Promo.BISHOP, Promo.ROOK, Promo.QUEEN)


@dataclass(frozen=True, slots=True)
class ChessMachineVM:
    """Deterministic host VM used to emit executable chess trace records."""

    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        legal_generator_program()
        make_move_program()
        terminal_check_program()

    def legal_move_trace(self, fen: str, include_halt: bool = False) -> list[TracePacket]:
        board = parse_fen(fen)
        trace = self.initial_board_trace(board)
        candidates = generate_pseudo_legal_moves(board)
        for move in candidates:
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
            legal = is_legal_move(board, move)
            trace.append(
                TracePacket(
                    TraceOp.LEGAL_SET,
                    move.move_id,
                    int(legal),
                    0,
                    0,
                    TraceTag.LEGAL,
                    0,
                )
            )
        if include_halt:
            trace.append(TracePacket(TraceOp.PROGRAM_HALT, 0, 0, 0, 0, TraceTag.LEGAL, 1))
        return trace

    def legal_moves(self, fen: str) -> list[MovePacket]:
        return legal_moves_from_trace(self.legal_move_trace(fen))

    def make_move(self, fen: str, move: MovePacket | str) -> str:
        board = parse_fen(fen)
        selected = self._resolve_legal_move(board, move)
        next_board = make_move_state(board, selected)
        return next_board.to_fen()

    def make_move_trace(self, fen: str, move: MovePacket | str, ply: int = 0) -> list[TracePacket]:
        board = parse_fen(fen)
        selected = self._resolve_legal_move(board, move)
        next_board = make_move_state(board, selected)
        trace = self.initial_board_trace(board)
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
        trace.extend(self.terminal_trace(next_board, ply + 1))
        return trace

    def terminal_trace(
        self,
        board: BoardState,
        ply: int = 0,
        repetition_count: int = 1,
        adjudication_cap_reached: bool = False,
    ) -> list[TracePacket]:
        status = terminal_status(board, ply, repetition_count, adjudication_cap_reached)
        return [
            TracePacket(
                TraceOp.TERMINAL_SET,
                int(status.result),
                int(status.reason),
                status.ply,
                0,
                TraceTag.TERMINAL,
                int(status.is_terminal),
            )
        ]

    def play_random_game_trace(self, start_fen: str, max_plies: int = 256) -> list[TracePacket]:
        import random

        rng = random.Random(self.seed)
        board = parse_fen(start_fen)
        trace = self.initial_board_trace(board)
        repetitions = {position_key(board): 1}
        for ply in range(max_plies):
            status = terminal_status(board, ply, repetitions[position_key(board)], False)
            trace.extend(self.terminal_trace(board, ply, repetitions[position_key(board)], False))
            if status.is_terminal:
                return trace
            legal = self.legal_moves(board.to_fen())
            if not legal:
                return trace
            selected = legal[rng.randrange(len(legal))]
            next_board = make_move_state(board, selected)
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
            board = next_board
            repetitions[position_key(board)] = repetitions.get(position_key(board), 0) + 1
        trace.extend(self.terminal_trace(board, max_plies, repetitions[position_key(board)], True))
        return trace

    def initial_board_trace(self, board: BoardState) -> list[TracePacket]:
        trace = [
            TracePacket(TraceOp.WRITE_SQ, square, piece_token(piece), 0, 0, TraceTag.BOARD, 0)
            for square, piece in enumerate(board.squares)
        ]
        trace.append(TracePacket(TraceOp.WRITE_REG, int(RegId.SIDE_TO_MOVE), 0 if board.side_to_move == WHITE else 1, 0, 0, TraceTag.STATE, 0))
        trace.append(TracePacket(TraceOp.WRITE_CASTLE, castling_mask(board.castling), 0, 0, 0, TraceTag.STATE, 0))
        trace.append(TracePacket(TraceOp.WRITE_EP, board.ep_square if board.ep_square is not None else NO_EP, 0, 0, 0, TraceTag.STATE, 0))
        trace.append(TracePacket(TraceOp.WRITE_CLOCK, board.halfmove_clock, board.fullmove_number, 0, 0, TraceTag.STATE, 0))
        return trace

    def _resolve_legal_move(self, board: BoardState, move: MovePacket | str) -> MovePacket:
        move_uci = move if isinstance(move, str) else move.to_uci()
        legal_by_uci = {candidate.to_uci(): candidate for candidate in self.legal_moves(board.to_fen())}
        if move_uci not in legal_by_uci:
            raise ValueError(f"illegal move for VM make_move: {move_uci}")
        return legal_by_uci[move_uci]


def legal_moves_from_trace(trace: Iterable[TracePacket]) -> list[MovePacket]:
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


def legal_uci_set_from_trace(trace: Iterable[TracePacket]) -> set[str]:
    return {move.to_uci() for move in legal_moves_from_trace(trace)}


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


def make_move_state(board: BoardState, move: MovePacket) -> BoardState:
    piece = board.squares[move.from_sq]
    target = board.squares[move.to_sq]
    if piece is None:
        raise ValueError(f"cannot move empty square: {move.from_sq}")
    next_squares = apply_move_squares(board, move)
    next_side = opposite(board.side_to_move)
    next_castling = update_castling_rights(board.castling, piece, move.from_sq, move.to_sq, target)
    next_ep = next_ep_square(piece, move)
    next_halfmove = 0 if piece.lower() == "p" or target is not None or (move.flags & MoveFlag.EP) else board.halfmove_clock + 1
    next_fullmove = board.fullmove_number + 1 if board.side_to_move == BLACK else board.fullmove_number
    return BoardState(next_squares, next_side, next_castling, next_ep, next_halfmove, next_fullmove)


def terminal_status(
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
    legal = [move for move in generate_pseudo_legal_moves(board) if is_legal_move(board, move)]
    if not legal:
        king_square = find_king(board.squares, board.side_to_move)
        in_check = king_square is not None and is_square_attacked(board.squares, king_square, opposite(board.side_to_move))
        if in_check:
            winner = ResultCode.BLACK_WIN if board.side_to_move == WHITE else ResultCode.WHITE_WIN
            return TerminalStatus(winner, TerminalReason.CHECKMATE, ply)
        return TerminalStatus(ResultCode.DRAW, TerminalReason.STALEMATE, ply)
    if insufficient_material(board.squares):
        return TerminalStatus(ResultCode.DRAW, TerminalReason.INSUFFICIENT_MATERIAL, ply)
    return TerminalStatus(ResultCode.ONGOING, TerminalReason.NONE, ply)


def position_key(board: BoardState) -> tuple[str, str, str, int | None]:
    return ("".join(piece or "." for piece in board.squares), board.side_to_move, board.castling, board.ep_square)


def generate_pseudo_legal_moves(board: BoardState) -> list[MovePacket]:
    moves: list[MovePacket] = []
    for square, piece in enumerate(board.squares):
        if piece is None or color_of(piece) != board.side_to_move:
            continue
        kind = piece.lower()
        if kind == "p":
            moves.extend(_pawn_moves(board, square, piece))
        elif kind == "n":
            moves.extend(_jump_moves(board, square, piece, KNIGHT_DELTAS))
        elif kind == "b":
            moves.extend(_slide_moves(board, square, piece, BISHOP_DIRS))
        elif kind == "r":
            moves.extend(_slide_moves(board, square, piece, ROOK_DIRS))
        elif kind == "q":
            moves.extend(_slide_moves(board, square, piece, QUEEN_DIRS))
        elif kind == "k":
            moves.extend(_king_moves(board, square, piece))
    return sorted(moves, key=lambda move: move.sort_key())


def is_legal_move(board: BoardState, move: MovePacket) -> bool:
    piece = board.squares[move.from_sq]
    if piece is None or color_of(piece) != board.side_to_move:
        return False
    target = board.squares[move.to_sq]
    if target is not None and (color_of(target) == board.side_to_move or target.lower() == "k"):
        return False
    if move.flags & MoveFlag.CASTLE and not _castle_path_is_safe(board, move):
        return False
    next_squares = apply_move_squares(board, move)
    king_square = find_king(next_squares, board.side_to_move)
    if king_square is None:
        return False
    return not is_square_attacked(next_squares, king_square, opposite(board.side_to_move))


def apply_move_squares(board: BoardState, move: MovePacket) -> tuple[str | None, ...]:
    squares = list(board.squares)
    piece = squares[move.from_sq]
    if piece is None:
        raise ValueError(f"cannot move empty square: {move.from_sq}")
    squares[move.from_sq] = None
    if move.flags & MoveFlag.EP:
        capture_square = move.to_sq - 8 if color_of(piece) == WHITE else move.to_sq + 8
        squares[capture_square] = None
    if move.flags & MoveFlag.CASTLE:
        _apply_castle_rook_move(squares, move)
    placed_piece = promotion_piece(piece, move.promo)
    squares[move.to_sq] = placed_piece
    return tuple(squares)


def update_castling_rights(castling: str, piece: str, from_sq: int, to_sq: int, captured: str | None) -> str:
    rights = set(castling)
    if piece == "K":
        rights.discard("K")
        rights.discard("Q")
    elif piece == "k":
        rights.discard("k")
        rights.discard("q")
    if from_sq == 0 or to_sq == 0:
        rights.discard("Q")
    if from_sq == 7 or to_sq == 7:
        rights.discard("K")
    if from_sq == 56 or to_sq == 56:
        rights.discard("q")
    if from_sq == 63 or to_sq == 63:
        rights.discard("k")
    if captured is not None:
        if to_sq == 0:
            rights.discard("Q")
        elif to_sq == 7:
            rights.discard("K")
        elif to_sq == 56:
            rights.discard("q")
        elif to_sq == 63:
            rights.discard("k")
    return "".join(right for right in "KQkq" if right in rights)


def next_ep_square(piece: str, move: MovePacket) -> int | None:
    if piece.lower() != "p" or abs(move.to_sq - move.from_sq) != 16:
        return None
    return (move.from_sq + move.to_sq) // 2


def insufficient_material(squares: tuple[str | None, ...] | list[str | None]) -> bool:
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
        colors = {(square % 8 + square // 8) % 2 for square in bishops}
        return len(colors) <= 1
    if len(pieces) == len(knights) and len(knights) <= 2:
        return True
    return False


def is_square_attacked(squares: tuple[str | None, ...] | list[str | None], target: int, by_color: str) -> bool:
    for square, piece in enumerate(squares):
        if piece is None or color_of(piece) != by_color:
            continue
        if _piece_attacks_square(squares, square, piece, target):
            return True
    return False


def find_king(squares: tuple[str | None, ...] | list[str | None], color: str) -> int | None:
    king = "K" if color == WHITE else "k"
    for square, piece in enumerate(squares):
        if piece == king:
            return square
    return None


def color_of(piece: str) -> str:
    return WHITE if piece.isupper() else BLACK


def opposite(color: str) -> str:
    return BLACK if color == WHITE else WHITE


def _pawn_moves(board: BoardState, square: int, piece: str) -> list[MovePacket]:
    color = color_of(piece)
    direction = 1 if color == WHITE else -1
    rank = square // 8
    file_index = square % 8
    start_rank = 1 if color == WHITE else 6
    promotion_from_rank = 6 if color == WHITE else 1
    moves: list[MovePacket] = []
    one_rank = rank + direction
    if 0 <= one_rank < 8:
        one_step = one_rank * 8 + file_index
        if board.squares[one_step] is None:
            if rank == promotion_from_rank:
                moves.extend(_promotion_moves(square, one_step, MoveFlag.QUIET))
            else:
                moves.append(MovePacket(square, one_step))
                two_rank = rank + 2 * direction
                two_step = two_rank * 8 + file_index
                if rank == start_rank and board.squares[two_step] is None:
                    moves.append(MovePacket(square, two_step))
    for file_delta in (-1, 1):
        target_file = file_index + file_delta
        target_rank = rank + direction
        if not _inside(target_file, target_rank):
            continue
        target = target_rank * 8 + target_file
        target_piece = board.squares[target]
        if target_piece is not None and color_of(target_piece) != color and target_piece.lower() != "k":
            if rank == promotion_from_rank:
                moves.extend(_promotion_moves(square, target, MoveFlag.CAPTURE))
            else:
                moves.append(MovePacket(square, target, Promo.NONE, MoveFlag.CAPTURE))
        if board.ep_square == target:
            moves.append(MovePacket(square, target, Promo.NONE, MoveFlag.EP | MoveFlag.CAPTURE))
    return moves


def _promotion_moves(from_sq: int, to_sq: int, flags: MoveFlag) -> list[MovePacket]:
    return [MovePacket(from_sq, to_sq, promo, flags | MoveFlag.PROMOTION) for promo in PROMOTION_ORDER]


def _jump_moves(board: BoardState, square: int, piece: str, deltas: tuple[tuple[int, int], ...]) -> list[MovePacket]:
    color = color_of(piece)
    file_index = square % 8
    rank = square // 8
    moves: list[MovePacket] = []
    for df, dr in deltas:
        target_file = file_index + df
        target_rank = rank + dr
        if not _inside(target_file, target_rank):
            continue
        target = target_rank * 8 + target_file
        flags = _target_move_flags(board.squares[target], color)
        if flags is not None:
            moves.append(MovePacket(square, target, Promo.NONE, flags))
    return moves


def _slide_moves(board: BoardState, square: int, piece: str, dirs: tuple[tuple[int, int], ...]) -> list[MovePacket]:
    color = color_of(piece)
    file_index = square % 8
    rank = square // 8
    moves: list[MovePacket] = []
    for df, dr in dirs:
        target_file = file_index + df
        target_rank = rank + dr
        while _inside(target_file, target_rank):
            target = target_rank * 8 + target_file
            target_piece = board.squares[target]
            flags = _target_move_flags(target_piece, color)
            if flags is not None:
                moves.append(MovePacket(square, target, Promo.NONE, flags))
            if target_piece is not None:
                break
            target_file += df
            target_rank += dr
    return moves


def _king_moves(board: BoardState, square: int, piece: str) -> list[MovePacket]:
    moves = _jump_moves(board, square, piece, KING_DELTAS)
    color = color_of(piece)
    if color == WHITE and square == 4:
        if "K" in board.castling and board.squares[5] is None and board.squares[6] is None and board.squares[7] == "R":
            moves.append(MovePacket(4, 6, Promo.NONE, MoveFlag.CASTLE))
        if "Q" in board.castling and board.squares[3] is None and board.squares[2] is None and board.squares[1] is None and board.squares[0] == "R":
            moves.append(MovePacket(4, 2, Promo.NONE, MoveFlag.CASTLE))
    if color == BLACK and square == 60:
        if "k" in board.castling and board.squares[61] is None and board.squares[62] is None and board.squares[63] == "r":
            moves.append(MovePacket(60, 62, Promo.NONE, MoveFlag.CASTLE))
        if "q" in board.castling and board.squares[59] is None and board.squares[58] is None and board.squares[57] is None and board.squares[56] == "r":
            moves.append(MovePacket(60, 58, Promo.NONE, MoveFlag.CASTLE))
    return moves


def _target_move_flags(target_piece: str | None, own_color: str) -> MoveFlag | None:
    if target_piece is None:
        return MoveFlag.QUIET
    if color_of(target_piece) == own_color or target_piece.lower() == "k":
        return None
    return MoveFlag.CAPTURE


def _castle_path_is_safe(board: BoardState, move: MovePacket) -> bool:
    enemy = opposite(board.side_to_move)
    start_square = move.from_sq
    transit_square = 5 if move.to_sq == 6 else 3 if move.to_sq == 2 else 61 if move.to_sq == 62 else 59
    if is_square_attacked(board.squares, start_square, enemy):
        return False
    transit_squares = list(board.squares)
    king = transit_squares[start_square]
    transit_squares[start_square] = None
    transit_squares[transit_square] = king
    if is_square_attacked(transit_squares, transit_square, enemy):
        return False
    final_squares = apply_move_squares(board, move)
    return not is_square_attacked(final_squares, move.to_sq, enemy)


def _apply_castle_rook_move(squares: list[str | None], move: MovePacket) -> None:
    if move.from_sq == 4 and move.to_sq == 6:
        squares[7] = None
        squares[5] = "R"
    elif move.from_sq == 4 and move.to_sq == 2:
        squares[0] = None
        squares[3] = "R"
    elif move.from_sq == 60 and move.to_sq == 62:
        squares[63] = None
        squares[61] = "r"
    elif move.from_sq == 60 and move.to_sq == 58:
        squares[56] = None
        squares[59] = "r"


def promotion_piece(piece: str, promo: Promo) -> str:
    if promo is Promo.NONE:
        return piece
    promoted = {
        Promo.KNIGHT: "N",
        Promo.BISHOP: "B",
        Promo.ROOK: "R",
        Promo.QUEEN: "Q",
    }[promo]
    return promoted if color_of(piece) == WHITE else promoted.lower()


def _piece_attacks_square(squares: tuple[str | None, ...] | list[str | None], square: int, piece: str, target: int) -> bool:
    color = color_of(piece)
    kind = piece.lower()
    file_index = square % 8
    rank = square // 8
    target_file = target % 8
    target_rank = target // 8
    if kind == "p":
        direction = 1 if color == WHITE else -1
        return target_rank == rank + direction and abs(target_file - file_index) == 1
    if kind == "n":
        return (target_file - file_index, target_rank - rank) in KNIGHT_DELTAS
    if kind == "k":
        return max(abs(target_file - file_index), abs(target_rank - rank)) == 1
    dirs = BISHOP_DIRS if kind == "b" else ROOK_DIRS if kind == "r" else QUEEN_DIRS
    for df, dr in dirs:
        current_file = file_index + df
        current_rank = rank + dr
        while _inside(current_file, current_rank):
            current = current_rank * 8 + current_file
            if current == target:
                return True
            if squares[current] is not None:
                break
            current_file += df
            current_rank += dr
    return False


def _inside(file_index: int, rank: int) -> bool:
    return 0 <= file_index < 8 and 0 <= rank < 8
