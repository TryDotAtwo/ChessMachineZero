"""Export independent full-context cases for the generic native tensor runner."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import chess
import numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_oracle import _native_rows
from vm_compiler.context import build_context_0
from vm_compiler.oracle import transition_oracle
from vm_compiler.protocol import PIECE_CHANNELS, encode_move_one_hot
from vm_compiler.recurrent_circuit import build_recurrent_artifact


SEQUENCES = {
    "ordinary": "e2e4 d7d5 e4d5",
    "fools-mate": "f2f3 e7e5 g2g4 d8h4",
    "next-move-threefold": "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1",
    "effective-ep-key": (
        "g1f3 d7d5 f3g1 d5d4 e2e4 g8f6 g1f3 f6g8 "
        "f3g1 g8f6 g1f3 f6g8"
    ),
    "stalemate": (
        "e2e3 a7a5 d1h5 a8a6 h5a5 h7h5 a5c7 a6h6 h2h4 f7f6 "
        "c7d7 e8f7 d7b7 d8d3 b7b8 d3h7 b8c8 f7g6 c8e6"
    ),
}


def _sequence_cases(name, text):
    board = chess.Board()
    context = build_context_0()
    cases = []
    for ply, uci in enumerate(text.split(), 1):
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        request = _native_rows(board, move)
        source = numpy.concatenate((request, context)).astype(numpy.float32, copy=False)
        expected = transition_oracle(context, request)
        cases.append((f"{name}-ply{ply}", source, expected))
        board.push(move)
        context = expected
    return cases


def export(destination):
    destination.mkdir(parents=True, exist_ok=False)
    circuit, output = build_recurrent_artifact()
    artifact = circuit.artifact(output).to_bytes()
    (destination / "recurrent.cmz").write_bytes(artifact)

    cases = []
    for name, moves in SEQUENCES.items():
        cases.extend(_sequence_cases(name, moves))
    initial = build_context_0()
    for name, request in (
        ("illegal-e2e5", encode_move_one_hot(52, 55, PIECE_CHANNELS["WHITE_PAWN"])),
        ("wrong-result-piece", encode_move_one_hot(52, 54, PIECE_CHANNELS["WHITE_KNIGHT"])),
    ):
        cases.append((
            name,
            numpy.concatenate((request, initial)).astype(numpy.float32, copy=False),
            transition_oracle(initial, request),
        ))

    with (destination / "recurrent.bin").open("wb") as stream:
        stream.write(struct.pack("<8sIII", b"CMZARR01", len(cases), 2045, 128))
        for label, source, expected in cases:
            encoded = label.encode("ascii")
            stream.write(struct.pack("<I", len(encoded)))
            stream.write(encoded)
            stream.write(source.astype("<f4", copy=False).tobytes())
            stream.write(expected.astype("<f4", copy=False).tobytes())

    manifest = {
        "oracle": "vm_compiler.oracle:transition_oracle",
        "scope": "full recurrent one-step contexts for the generic native tensor runner",
        "known_pending": [],
        "operations": len(circuit.operations),
        "tensors": len(circuit.tensors),
        "cases": len(cases),
        "labels": [label for label, _, _ in cases],
        "memory_payloads": circuit.memory_estimate(),
        "sha256": {
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in ("recurrent.cmz", "recurrent.bin")
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cases": len(cases), **manifest["memory_payloads"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New directory; never overwrite fixtures")
    export(parser.parse_args().destination)
