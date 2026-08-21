from __future__ import annotations

import os
import random
import subprocess

import chess

from test_vm3_pseudo_legal_oracle import oracle_binary
from test_vm3_state_oracle import ROOT, castling_code, encode_board


def positions() -> list[chess.Board]:
    result = [
        chess.Board(),
        chess.Board("4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1"),
        chess.Board("k7/8/8/r4ppK/8/8/8/8 w - f6 0 1"),
        chess.Board("k3r3/8/8/8/8/8/8/4K2R w K - 0 1"),
        chess.Board("k4r2/8/8/8/8/8/8/4K2R w K - 0 1"),
        chess.Board("k5r1/8/8/8/8/8/8/4K2R w K - 0 1"),
        chess.Board("k7/8/8/8/8/8/8/4K2R w K - 0 1"),
        chess.Board("8/P6k/8/8/8/8/7p/K7 w - - 0 1"),
    ]
    rng = random.Random(0x1E6A14272)
    board = chess.Board()
    while len(result) < 520:
        result.append(board.copy(stack=False))
        legal = list(board.legal_moves)
        if not legal or board.ply() >= 180:
            board = chess.Board()
        else:
            board.push(rng.choice(legal))
    return result


def test_native_final_legal_set_matches_python_chess() -> None:
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
            f"/work/{relative}", "--cuda", "--legal",
        ]
    else:
        command = [str(binary), "--legal"]
    result = subprocess.run(command, input=payload, check=True,
                            capture_output=True, text=True, timeout=900)
    lines = [line.removeprefix("CMZ|") for line in result.stdout.splitlines()
             if line.startswith("CMZ|")]
    assert len(lines) == len(boards)
    for index, (board, line) in enumerate(zip(boards, lines, strict=True)):
        observed = set(filter(None, line.split(',')))
        expected = {move.uci() for move in board.legal_moves}
        assert observed == expected, f"position {index}: {board.fen()}"


def test_sparse_hard_forward_matches_dense_on_full_oracle_corpus() -> None:
    boards = positions()
    payload = "".join(
        f"{','.join(map(str, encode_board(board)))}|{0 if board.turn else 1}|"
        f"{castling_code(board)}|{-1 if board.ep_square is None else board.ep_square}|\n"
        for board in boards
    )
    binary = oracle_binary()
    relative = binary.relative_to(ROOT).as_posix()
    base = [
        "docker", "run", "--rm", "-i", "--gpus", "all", "-v",
        f"{ROOT}:/work", "-w", "/work", "cmz-native-dev:2026-05-26",
        f"/work/{relative}", "--cuda", "--legal",
    ] if os.name == "nt" else [str(binary), "--legal"]
    dense = subprocess.run(base, input=payload, check=True, capture_output=True,
                           text=True, timeout=900)
    sparse = subprocess.run([*base, "--sparse-hard"], input=payload, check=True,
                            capture_output=True, text=True, timeout=900)
    dense_lines = [line for line in dense.stdout.splitlines()
                   if line.startswith("CMZ|")]
    sparse_lines = [line for line in sparse.stdout.splitlines()
                    if line.startswith("CMZ|")]
    assert sparse_lines == dense_lines
