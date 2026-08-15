from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

import chess
import pytest

from test_vm3_state_oracle import ROOT, castling_code, encode_board


def oracle_binary() -> Path:
    configured = os.environ.get("CMZ_VM3_PSEUDO_LEGAL_ORACLE")
    candidates = [
        Path(configured) if configured else None,
        ROOT / "build/vm3/native/vm3/cmz_vm3_pseudo_legal_oracle_cli",
        ROOT / "build/vm3-red/native/vm3/cmz_vm3_pseudo_legal_oracle_cli",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    pytest.skip("native VM3 pseudo-legal oracle CLI has not been built")


def expected_moves(board: chess.Board) -> set[str]:
    return {move.uci() for move in board.generate_pseudo_legal_moves()}


def positions() -> list[chess.Board]:
    result = [
        chess.Board(),
        chess.Board("8/P6k/8/8/8/8/7p/K7 w - - 0 1"),
        chess.Board("8/P6k/8/8/8/8/7p/K7 b - - 0 1"),
        chess.Board("4k3/8/3p4/3R1n2/3B4/8/8/4K3 w - - 0 1"),
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
        chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
        chess.Board("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1"),
    ]
    target_count = len(result) + 512
    rng = random.Random(0xC0DE4272)
    board = chess.Board()
    while len(result) < target_count:
        board.castling_rights = chess.BB_EMPTY
        board.ep_square = None
        result.append(board.copy(stack=False))
        legal = list(board.legal_moves)
        if not legal or board.ply() >= 180:
            board = chess.Board()
        else:
            board.push(rng.choice(legal))
    return result


def test_native_pseudo_legal_set_matches_python_chess() -> None:
    boards = positions()
    payload = "".join(
        f"{','.join(map(str, encode_board(board)))}|{0 if board.turn else 1}|"
        f"{castling_code(board)}|{-1 if board.ep_square is None else board.ep_square}|\n"
        for board in boards
    )
    binary = oracle_binary()
    if os.name == "nt":
        relative = binary.relative_to(ROOT).as_posix()
        command = [
            "docker", "run", "--rm", "-i", "--gpus", "all", "-v",
            f"{ROOT}:/work", "-w", "/work", "cmz-native-dev:2026-05-26",
            f"/work/{relative}", "--cuda",
        ]
    else:
        command = [str(binary)]
    result = subprocess.run(command, input=payload, check=True,
                            capture_output=True, text=True, timeout=600)
    lines = [line.removeprefix("CMZ|") for line in result.stdout.splitlines()
             if line.startswith("CMZ|")]
    assert len(lines) == len(boards)
    for index, (board, line) in enumerate(zip(boards, lines, strict=True)):
        observed = set(filter(None, line.split(',')))
        assert observed == expected_moves(board), f"position {index}: {board.fen()}"
