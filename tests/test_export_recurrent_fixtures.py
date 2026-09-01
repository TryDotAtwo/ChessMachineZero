import hashlib
import json
import struct

from vm_compiler.artifact import Artifact


def test_recurrent_export_contains_full_context_native_cases(tmp_path):
    from export_recurrent_fixtures import export

    destination = tmp_path / "recurrent"
    export(destination)
    artifact_bytes = (destination / "recurrent.cmz").read_bytes()
    fixtures = (destination / "recurrent.bin").read_bytes()
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))

    artifact = Artifact.from_bytes(artifact_bytes)
    magic, cases, rows, columns = struct.unpack_from("<8sIII", fixtures)
    assert magic == b"CMZARR01"
    assert cases == manifest["cases"] >= 10
    assert (rows, columns) == (2045, 128)
    assert artifact.operations[-1].outputs == (len(artifact.operations),)
    assert manifest["operations"] == len(artifact.operations)
    assert "effective-ep-key-ply12" in manifest["labels"]
    assert manifest["known_pending"] == []
    assert manifest["sha256"]["recurrent.cmz"] == hashlib.sha256(artifact_bytes).hexdigest()
    assert manifest["sha256"]["recurrent.bin"] == hashlib.sha256(fixtures).hexdigest()
