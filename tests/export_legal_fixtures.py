"""Development-only independent legal sets for the generic native tensor runner."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_oracle import legal_set_snapshots
from vm_compiler.legal_circuit import build_legal_artifact


CASES = (
    "", "e2e4 d7d5 e4d5", "f2f3 e7e5 g2g4 d8h4",
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1",
    "d2d4 d7d5 b1c3 b8c6 c1f4 c8f5 d1d2 d8d7 e1c1 e8c8",
    "e2e4 a7a6 e4e5 d7d5 e5d6",
    "a2a4 h7h5 a4a5 h5h4 a5a6 h4h3 a6b7 h3g2 b7a8q g2h1q",
    "h2h4 a7a6 h1h3 a6a5 h3h1 a5a4 g2g3 d7d5 f1g2 d5d4 g1f3 b7b6",
    "g2g3 b7b6 f1g2 c8a6 g1f3 e7e6 e2e4 h7h6",
    "e2e4 e7e5 f1c4 b8c6 g1h3 d7d6 f2f4 d8h4",
)


def export(destination):
    destination.mkdir(parents=True, exist_ok=False)
    circuit, output = build_legal_artifact()
    artifact = circuit.artifact(output).to_bytes()
    (destination / "legal.cmz").write_bytes(artifact)
    labels = []
    with (destination / "legal.bin").open("w+b") as stream:
        stream.write(struct.pack("<8sIII", b"CMZARR01", 0, 768, 128))
        for case, moves in enumerate(CASES):
            for ply, (source, expected) in enumerate(legal_set_snapshots(moves.split())):
                label = f"case{case}-ply{ply}".encode("ascii")
                stream.write(struct.pack("<I", len(label)))
                stream.write(label)
                stream.write(source.astype("<f4").tobytes())
                stream.write(expected.astype("<f4").tobytes())
                labels.append(label.decode("ascii"))
        stream.seek(8)
        stream.write(struct.pack("<I", len(labels)))
    manifest = {
        "oracle": "tests/chess_oracle.py:legal_set_snapshots",
        "scope": "legal-set subgraph only; not a full recurrent transition",
        "operations": len(circuit.operations), "tensors": len(circuit.tensors),
        "memory_payloads": circuit.memory_estimate(),
        "cases": labels,
        "sha256": {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                   for name in ("legal.cmz", "legal.bin")},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(labels), **manifest["memory_payloads"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New directory; never overwrite fixtures")
    export(parser.parse_args().destination)
