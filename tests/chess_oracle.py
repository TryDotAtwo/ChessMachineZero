"""Development/test-only python-chess fixtures, independent of VM reconstruction."""

import random

import chess
import numpy

from vm_compiler.protocol import encode_move_one_hot


_CHANNEL_BY_SYMBOL = dict(zip("PNBRQKpnbrqk", range(96, 108)))


def _native_rows(board, move):
    def channel(square):
        return 10 * (chess.square_file(square) + 1) + chess.square_rank(square) + 1

    piece = board.piece_at(move.from_square)
    result = chess.Piece(move.promotion or piece.piece_type, piece.color)
    return encode_move_one_hot(
        channel(move.from_square), channel(move.to_square), _CHANNEL_BY_SYMBOL[result.symbol()]
    )


def _expected_board(board):
    channels = numpy.full(64, 95)
    for square, piece in board.piece_map().items():
        compact = 8 * chess.square_file(square) + chess.square_rank(square)
        channels[compact] = _CHANNEL_BY_SYMBOL[piece.symbol()]
    return numpy.eye(128, dtype=numpy.float32)[channels]


def legal_history_snapshots(*, seed=None, plies=400, uci_moves=()):
    """Return all exact prefix inputs/boards and independently verified special plies."""
    board = chess.Board()
    source = numpy.zeros((2048, 128), dtype=numpy.float32)
    source[:, 0] = 1
    snapshots = {0: (source.copy(), _expected_board(board))}
    special = {"promotion": [], "castling": [], "en_passant": []}
    rng = random.Random(seed)
    for index in range(plies if seed is not None else len(uci_moves)):
        if seed is not None:
            candidates = list(board.legal_moves)
            rng.shuffle(candidates)
            for move in candidates:
                board.push(move)
                ongoing = not board.is_game_over(claim_draw=True)
                board.pop()
                if ongoing:
                    break
            else:
                raise AssertionError("no continuing move")
        else:
            move = chess.Move.from_uci(uci_moves[index])
        assert move in board.legal_moves
        if move.promotion:
            special["promotion"].append(index + 1)
        if board.is_castling(move):
            special["castling"].append(index + 1)
        if board.is_en_passant(move):
            special["en_passant"].append(index + 1)
        source[3 + 3 * index:6 + 3 * index] = _native_rows(board, move)
        board.push(move)
        assert not board.is_game_over(claim_draw=True)
        snapshots[index + 1] = (source.copy(), _expected_board(board))
    return snapshots, special
