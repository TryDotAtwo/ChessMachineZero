"""Development-only python-chess oracle for recurrent tensor transitions."""

from __future__ import annotations

import chess
import numpy

from .protocol import (
    CONTEXT_ROWS,
    HISTORY_ROWS,
    LEGAL_ROWS,
    MOVE_ROWS,
    PIECE_CHANNELS,
    STATUS_CHANNELS,
    VOCAB_SIZE,
)

_PIECE_TO_CHANNEL = {
    (chess.WHITE, chess.PAWN): PIECE_CHANNELS["WHITE_PAWN"],
    (chess.WHITE, chess.KNIGHT): PIECE_CHANNELS["WHITE_KNIGHT"],
    (chess.WHITE, chess.BISHOP): PIECE_CHANNELS["WHITE_BISHOP"],
    (chess.WHITE, chess.ROOK): PIECE_CHANNELS["WHITE_ROOK"],
    (chess.WHITE, chess.QUEEN): PIECE_CHANNELS["WHITE_QUEEN"],
    (chess.WHITE, chess.KING): PIECE_CHANNELS["WHITE_KING"],
    (chess.BLACK, chess.PAWN): PIECE_CHANNELS["BLACK_PAWN"],
    (chess.BLACK, chess.KNIGHT): PIECE_CHANNELS["BLACK_KNIGHT"],
    (chess.BLACK, chess.BISHOP): PIECE_CHANNELS["BLACK_BISHOP"],
    (chess.BLACK, chess.ROOK): PIECE_CHANNELS["BLACK_ROOK"],
    (chess.BLACK, chess.QUEEN): PIECE_CHANNELS["BLACK_QUEEN"],
    (chess.BLACK, chess.KING): PIECE_CHANNELS["BLACK_KING"],
}
_CHANNEL_TO_PIECE = {channel: piece for piece, channel in _PIECE_TO_CHANNEL.items()}


def _square_to_chess(channel: int) -> chess.Square:
    return chess.square(channel // 10 - 1, channel % 10 - 1)


def _square_from_chess(square: chess.Square) -> int:
    return (chess.square_file(square) + 1) * 10 + chess.square_rank(square) + 1


def _decode_rows(rows: numpy.ndarray) -> tuple[int, ...]:
    return tuple(int(channel) for channel in numpy.argmax(rows, axis=1))


def _native_to_move(board: chess.Board, native: tuple[int, int, int]) -> chess.Move:
    source, destination, result_channel = native
    source_square = _square_to_chess(source)
    source_piece = board.piece_at(source_square)
    promotion = None
    if source_piece is not None and source_piece.piece_type == chess.PAWN:
        result = _CHANNEL_TO_PIECE.get(result_channel)
        if result is not None and result[0] == source_piece.color and result[1] != chess.PAWN:
            promotion = result[1]
    return chess.Move(
        source_square,
        _square_to_chess(destination),
        promotion=promotion,
    )


def _move_to_native(board: chess.Board, move: chess.Move) -> tuple[int, int, int]:
    source_piece = board.piece_at(move.from_square)
    if source_piece is None:
        raise ValueError("legal move has no source piece")
    result_type = move.promotion or source_piece.piece_type
    return (
        _square_from_chess(move.from_square),
        _square_from_chess(move.to_square),
        _PIECE_TO_CHANNEL[(source_piece.color, result_type)],
    )


def _declared_piece_matches(board: chess.Board, native: tuple[int, int, int], move: chess.Move) -> bool:
    return native[2] == _move_to_native(board, move)[2]


def _one_hot_rows(channels: list[int]) -> numpy.ndarray:
    rows = numpy.zeros((len(channels), VOCAB_SIZE), dtype=numpy.float32)
    rows[numpy.arange(len(channels)), channels] = 1.0
    return rows


def _replay_history(context: numpy.ndarray) -> tuple[chess.Board, int]:
    board = chess.Board()
    used_rows = 0
    for start in range(0, HISTORY_ROWS, MOVE_ROWS):
        native = _decode_rows(context[start : start + MOVE_ROWS])
        if native == (0, 0, 0):
            break
        move = _native_to_move(board, native)
        if move not in board.legal_moves or not _declared_piece_matches(board, native, move):
            raise ValueError("history contains a non-legal move")
        board.push(move)
        used_rows += MOVE_ROWS
    return board, used_rows


def _status_after(board: chess.Board) -> int:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return STATUS_CHANNELS["OK"]
    if outcome.winner is True:
        return STATUS_CHANNELS["WHITE_WIN"]
    if outcome.winner is False:
        return STATUS_CHANNELS["BLACK_WIN"]
    return STATUS_CHANNELS["DRAW"]


def transition_oracle(context: numpy.ndarray, requested_move: numpy.ndarray) -> numpy.ndarray:
    state = numpy.asarray(context, dtype=numpy.float32)
    request = numpy.asarray(requested_move, dtype=numpy.float32)
    if state.shape != (CONTEXT_ROWS, VOCAB_SIZE):
        raise ValueError("context must have shape [2045,128]")
    if request.shape != (MOVE_ROWS, VOCAB_SIZE):
        raise ValueError("requested move must have shape [3,128]")
    if not numpy.all(state.sum(axis=1) == 1.0) or not numpy.all(request.sum(axis=1) == 1.0):
        raise ValueError("oracle inputs must be hard one-hot rows")

    board, used_rows = _replay_history(state)
    result = state.copy()
    status_row = HISTORY_ROWS + LEGAL_ROWS
    if used_rows == HISTORY_ROWS:
        result[status_row] = _one_hot_rows([STATUS_CHANNELS["HISTORY_OVERFLOW"]])[0]
        return result

    native = _decode_rows(request)
    move = _native_to_move(board, native)
    if move not in board.legal_moves or not _declared_piece_matches(board, native, move):
        result[status_row] = _one_hot_rows([STATUS_CHANNELS["ILLEGAL_MOVE"]])[0]
        return result

    result[used_rows : used_rows + MOVE_ROWS] = request
    board.push(move)
    legal_moves = sorted(_move_to_native(board, candidate) for candidate in board.legal_moves)
    legal_channels = [channel for candidate in legal_moves for channel in candidate]
    legal_channels.extend([0] * (LEGAL_ROWS - len(legal_channels)))
    result[HISTORY_ROWS : status_row] = _one_hot_rows(legal_channels)
    result[status_row] = _one_hot_rows([_status_after(board)])[0]
    return result
