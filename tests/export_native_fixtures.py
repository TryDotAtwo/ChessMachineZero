"""Development-only export: legal oracle histories + exact full-board native inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_oracle import legal_history_snapshots
from vm_compiler.compiler import build_position_reconstruction_artifact


def export(destination):
    destination.mkdir(parents=True, exist_ok=False)
    artifact = build_position_reconstruction_artifact().to_bytes()
    (destination / "position.cmz").write_bytes(artifact)
    labels = []
    with (destination / "positions.bin").open("w+b") as stream:
        stream.write(struct.pack("<8sI", b"CMZPOS01", 0))

        def write_case(label, pair):
            source, expected = pair
            assert source.shape == (2048, 128) and expected.shape == (64, 128)
            encoded = label.encode("ascii")
            stream.write(struct.pack("<I", len(encoded)))
            stream.write(encoded)
            stream.write(source.astype("<f4").tobytes())
            stream.write(expected.astype("<f4").tobytes())
            labels.append(label)

        snapshots, special = legal_history_snapshots(seed=1)
        assert special["promotion"]
        prefixes = {0, 1, 28, 29, 244, 245, 246, 398, 399, 400, *special["promotion"]}
        for ply in sorted(prefixes):
            write_case(f"seed1-ply{ply}", snapshots[ply])
        del snapshots
        short_cases = (
            ("castle-kingside", "castling", "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1"),
            ("castle-queenside-both", "castling", "d2d4 d7d5 b1c3 b8c6 c1f4 c8f5 d1d2 d8d7 e1c1 e8c8"),
            ("en-passant", "en_passant", "e2e4 a7a6 e4e5 d7d5 e5d6"),
            ("promotion-both", "promotion", "a2a4 h7h5 a4a5 h5h4 a5a6 h4h3 a6b7 h3g2 b7a8q g2h1q"),
        )
        for name, kind, moves in short_cases:
            snapshots, special = legal_history_snapshots(uci_moves=moves.split())
            assert special[kind], f"{name} did not exercise {kind}"
            for ply, pair in snapshots.items():
                write_case(f"{name}-ply{ply}", pair)
        stream.seek(8)
        stream.write(struct.pack("<I", len(labels)))
    manifest = {
        "oracle": "tests/chess_oracle.py (python-chess development/test wrapper)",
        "precision": "little-endian FP32 full one-hot inputs and expected boards",
        "cases": labels,
        "sha256": {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                   for name in ("position.cmz", "positions.bin")},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(labels)} exact full-board cases to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New output directory; never overwrite fixtures")
    export(parser.parse_args().destination)
