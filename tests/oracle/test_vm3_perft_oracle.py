from __future__ import annotations

import os
import subprocess
from pathlib import Path

import chess
import pytest

from test_vm3_state_oracle import ROOT, castling_code, encode_board

KIWIPETE_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


def perft_binary() -> Path:
    configured = os.environ.get("CMZ_VM3_PERFT_ORACLE")
    candidates = [
        Path(configured) if configured else None,
        ROOT / "native/vm3/build/cmz_vm3_perft_oracle_cli",
        ROOT / "build/vm3/native/vm3/cmz_vm3_perft_oracle_cli",
        ROOT / "build/vm3-red/native/vm3/cmz_vm3_perft_oracle_cli",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    pytest.skip("native VM3 perft oracle CLI has not been built")


def payload(board: chess.Board, depth: int) -> str:
    return (
        f"{','.join(map(str, encode_board(board)))}|"
        f"{0 if board.turn else 1}|"
        f"{castling_code(board)}|"
        f"{-1 if board.ep_square is None else board.ep_square}|"
        f"{board.halfmove_clock}|{board.fullmove_number}|{depth}|\n"
    )


@pytest.mark.parametrize(
    ("fen", "depth", "expected"),
    [
        (chess.STARTING_FEN, 1, 20),
        (chess.STARTING_FEN, 2, 400),
        (chess.STARTING_FEN, 3, 8902),
        (KIWIPETE_FEN, 1, 48),
        (KIWIPETE_FEN, 2, 2039),
        ("8/P6k/8/8/8/8/7p/K7 w - - 0 1", 1, 7),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", 1, 7),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", 1, 26),
    ],
)
def test_native_perft_counts_match_known_positions(
    fen: str, depth: int, expected: int
) -> None:
    board = chess.Board(fen)
    binary = perft_binary()
    if os.name == "nt":
        relative = binary.relative_to(ROOT).as_posix()
        command = [
            "docker", "run", "--rm", "-i", "--gpus", "all", "-v",
            f"{ROOT}:/work", "-w", "/work", "cmz-native-dev:2026-05-26",
            f"/work/{relative}", "--cuda",
        ]
    else:
        command = [str(binary)]
    result = subprocess.run(
        command,
        input=payload(board, depth),
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    lines = [line for line in result.stdout.splitlines()
             if line.startswith("CMZ_PERFT|")]
    assert lines == [f"CMZ_PERFT|{expected}"]
