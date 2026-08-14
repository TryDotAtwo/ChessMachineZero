from __future__ import annotations

import os
import subprocess
from pathlib import Path

import chess
import pytest


ROOT = Path(__file__).resolve().parents[2]
PIECE_CODE = {
    None: 0,
    (chess.WHITE, chess.PAWN): 1,
    (chess.WHITE, chess.KNIGHT): 2,
    (chess.WHITE, chess.BISHOP): 3,
    (chess.WHITE, chess.ROOK): 4,
    (chess.WHITE, chess.QUEEN): 5,
    (chess.WHITE, chess.KING): 6,
    (chess.BLACK, chess.PAWN): 7,
    (chess.BLACK, chess.KNIGHT): 8,
    (chess.BLACK, chess.BISHOP): 9,
    (chess.BLACK, chess.ROOK): 10,
    (chess.BLACK, chess.QUEEN): 11,
    (chess.BLACK, chess.KING): 12,
}


def oracle_binary() -> Path:
    configured = os.environ.get("CMZ_VM3_STATE_ORACLE")
    candidates = [
        Path(configured) if configured else None,
        ROOT / "build/vm3/native/vm3/cmz_vm3_state_oracle_cli",
        ROOT / "build/vm3-red/native/vm3/cmz_vm3_state_oracle_cli",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    pytest.skip("native VM3 state oracle CLI has not been built")


def run_native(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    binary = oracle_binary()
    if os.name != "nt":
        command = [str(binary), *arguments]
    else:
        try:
            relative = binary.relative_to(ROOT).as_posix()
        except ValueError:
            pytest.skip("Windows oracle fallback requires a workspace-local binary")
        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{ROOT}:/work",
            "-w",
            "/work",
            "cmz-native-dev:2026-05-26",
            f"/work/{relative}",
            *arguments,
        ]
    return subprocess.run(command, check=True, capture_output=True, text=True)


def encode_board(board: chess.Board) -> list[int]:
    result: list[int] = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        result.append(PIECE_CODE[None if piece is None else (piece.color, piece.piece_type)])
    return result


def castling_code(board: chess.Board) -> int:
    return (
        int(board.has_kingside_castling_rights(chess.WHITE))
        | (int(board.has_queenside_castling_rights(chess.WHITE)) << 1)
        | (int(board.has_kingside_castling_rights(chess.BLACK)) << 2)
        | (int(board.has_queenside_castling_rights(chess.BLACK)) << 3)
    )


def bind(board: chess.Board, claim_mode: int = 0) -> tuple[list[int], list[int]]:
    pieces = encode_board(board)
    result = run_native(
        [
            ",".join(map(str, pieces)),
            str(0 if board.turn == chess.WHITE else 1),
            str(castling_code(board)),
            str(-1 if board.ep_square is None else board.ep_square),
            str(board.halfmove_clock),
            str(board.fullmove_number),
            str(claim_mode),
        ]
    )
    lines = result.stdout.strip().splitlines()
    return list(map(int, lines[-2].split(","))), list(map(int, lines[-1].split(",")))


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "4k3/1qrbnp2/8/8/8/8/2PNBRQ1/4K3 w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 150 9526",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 42",
    ],
)
@pytest.mark.parametrize("claim_mode", [0, 1])
def test_native_state_binding_matches_python_chess_fen(fen: str, claim_mode: int) -> None:
    board = chess.Board(fen)
    observed_board, metadata = bind(board, claim_mode)
    assert observed_board == encode_board(board)
    assert metadata == [
        0 if board.turn == chess.WHITE else 1,
        castling_code(board),
        0 if board.ep_square is None else board.ep_square + 1,
        board.halfmove_clock,
        board.fullmove_number % 128,
        board.fullmove_number // 128,
        claim_mode,
        0,
        0,
    ]
