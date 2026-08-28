"""Export the executable artifact graph for the static inspector site."""

from __future__ import annotations

import json
from pathlib import Path

from vm_compiler.compiler import build_position_reconstruction_artifact


def build_site_trace() -> dict[str, object]:
    artifact = build_position_reconstruction_artifact()
    operations = [
        {
            "index": index,
            "opcode": operation.opcode.name,
            "inputs": list(operation.inputs),
            "outputs": list(operation.outputs),
            "attributes": list(operation.attributes),
        }
        for index, operation in enumerate(artifact.operations, start=1)
    ]
    return {
        "artifact": "position_latest_event_v1",
        "operation_count": len(operations),
        "runtime_opcodes": sorted({operation["opcode"] for operation in operations}),
        "final_output": {"value": operations[-1]["outputs"][0], "shape": ["B", 64, 128]},
        "operations": operations,
    }


def main() -> None:
    destination = Path(__file__).parents[1] / "site" / "artifact_trace.json"
    destination.write_text(
        json.dumps(build_site_trace(), indent=2, separators=(",", ": ")) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
