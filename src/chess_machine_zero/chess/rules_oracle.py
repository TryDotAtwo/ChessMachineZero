"""Development/test-only rules oracle backed by python-chess."""

from __future__ import annotations

import chess

from chess_machine_zero.chess.move_packet import MoveFlag, MovePacket, Promo
from chess_machine_zero.chess.outcome import ResultCode, TerminalReason, TerminalStatus


def legal_uci_set(fen: str) -> set[str]:
    board = chess.Board(fen)
    return {move.uci() for move in board.legal_moves}


def legal_move_packets(fen: str) -> list[MovePacket]:
    board = chess.Board(fen)
    packets = []
    for move in board.legal_moves:
        flags = MoveFlag.QUIET
        if board.is_capture(move):
            flags |= MoveFlag.CAPTURE
        if board.is_en_passant(move):
            flags |= MoveFlag.EP
        if board.is_castling(move):
            flags |= MoveFlag.CASTLE
        promo = Promo.NONE
        if move.promotion:
            flags |= MoveFlag.PROMOTION
            promo = {
                chess.KNIGHT: Promo.KNIGHT,
                chess.BISHOP: Promo.BISHOP,
                chess.ROOK: Promo.ROOK,
                chess.QUEEN: Promo.QUEEN,
            }[move.promotion]
        packets.append(MovePacket(move.from_square, move.to_square, promo, flags))
    return sorted(packets, key=lambda move: move.sort_key())


def board_after_uci(fen: str, uci: str) -> str:
    board = chess.Board(fen)
    board.push(chess.Move.from_uci(uci))
    return board.fen(en_passant="fen")


def terminal_status(fen: str, ply: int = 0) -> TerminalStatus:
    board = chess.Board(fen)
    if board.is_checkmate():
        return TerminalStatus(
            ResultCode.BLACK_WIN if board.turn == chess.WHITE else ResultCode.WHITE_WIN,
            TerminalReason.CHECKMATE,
            ply,
        )
    if board.is_stalemate():
        return TerminalStatus(ResultCode.DRAW, TerminalReason.STALEMATE, ply)
    if board.is_fifty_moves():
        return TerminalStatus(ResultCode.DRAW, TerminalReason.FIFTY_MOVE, ply)
    if board.is_insufficient_material():
        return TerminalStatus(ResultCode.DRAW, TerminalReason.INSUFFICIENT_MATERIAL, ply)
    return TerminalStatus(ResultCode.ONGOING, TerminalReason.NONE, ply)
