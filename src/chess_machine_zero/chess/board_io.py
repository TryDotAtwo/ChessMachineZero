"""FEN parsing and board encoding without strategy logic."""

from __future__ import annotations

from dataclasses import dataclass

from chess_machine_zero.chess.move_packet import square_index, square_name


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
PIECES = "PNBRQKpnbrqk"
EMPTY_TOKEN = 0
PIECE_TO_TOKEN = {piece: idx + 1 for idx, piece in enumerate(PIECES)}
TOKEN_TO_PIECE = {idx + 1: piece for idx, piece in enumerate(PIECES)}
TOKEN_TO_PIECE[EMPTY_TOKEN] = None
CASTLING_BITS = {"K": 1, "Q": 2, "k": 4, "q": 8}
CASTLING_ORDER = "KQkq"
NO_EP = 64


@dataclass(frozen=True, slots=True)
class BoardState:
    squares: tuple[str | None, ...]
    side_to_move: str
    castling: str
    ep_square: int | None
    halfmove_clock: int
    fullmove_number: int

    def __post_init__(self) -> None:
        if len(self.squares) != 64:
            raise ValueError(f"board requires 64 squares, got {len(self.squares)}")
        if self.side_to_move not in ("w", "b"):
            raise ValueError(f"invalid side to move: {self.side_to_move!r}")

    def piece_at(self, square: int) -> str | None:
        if not 0 <= square < 64:
            raise ValueError(f"square out of range: {square}")
        return self.squares[square]

    def to_fen(self) -> str:
        ep = "-" if self.ep_square is None else square_name(self.ep_square)
        castling = self.castling if self.castling else "-"
        return (
            f"{piece_placement_from_squares(self.squares)} {self.side_to_move} "
            f"{castling} {ep} {self.halfmove_clock} {self.fullmove_number}"
        )


def parse_fen(fen: str) -> BoardState:
    fields = fen.strip().split()
    if len(fields) != 6:
        raise ValueError(f"FEN requires 6 fields, got {len(fields)}: {fen!r}")
    placement, side, castling, ep, halfmove, fullmove = fields
    squares: list[str | None] = [None] * 64
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError(f"FEN placement requires 8 ranks: {placement!r}")
    for fen_rank_index, rank_text in enumerate(ranks):
        board_rank = 7 - fen_rank_index
        file_index = 0
        for char in rank_text:
            if char.isdigit():
                file_index += int(char)
                continue
            if char not in PIECES:
                raise ValueError(f"invalid FEN piece: {char!r}")
            if file_index >= 8:
                raise ValueError(f"FEN rank overflow: {rank_text!r}")
            squares[board_rank * 8 + file_index] = char
            file_index += 1
        if file_index != 8:
            raise ValueError(f"FEN rank width is {file_index}, expected 8: {rank_text!r}")
    castling_value = "" if castling == "-" else "".join(ch for ch in CASTLING_ORDER if ch in castling)
    ep_square = None if ep == "-" else square_index(ep)
    return BoardState(tuple(squares), side, castling_value, ep_square, int(halfmove), int(fullmove))


def piece_placement_from_squares(squares: tuple[str | None, ...]) -> str:
    ranks: list[str] = []
    for rank in range(7, -1, -1):
        empty = 0
        text = []
        for file_index in range(8):
            piece = squares[rank * 8 + file_index]
            if piece is None:
                empty += 1
                continue
            if empty:
                text.append(str(empty))
                empty = 0
            text.append(piece)
        if empty:
            text.append(str(empty))
        ranks.append("".join(text))
    return "/".join(ranks)


def piece_token(piece: str | None) -> int:
    if piece is None:
        return EMPTY_TOKEN
    return PIECE_TO_TOKEN[piece]


def piece_from_token(token: int) -> str | None:
    if token not in TOKEN_TO_PIECE:
        raise ValueError(f"unknown piece token: {token}")
    return TOKEN_TO_PIECE[token]


def castling_mask(castling: str) -> int:
    mask = 0
    for char in castling:
        mask |= CASTLING_BITS[char]
    return mask
