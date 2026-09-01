"""Export the executable artifact graph for the static inspector site."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy
import torch

from vm_compiler.compiler import build_position_reconstruction_artifact
from vm_compiler.graph import OpCode
from vm_compiler.protocol import INPUT_ROWS, PIECE_CHANNELS, VOCAB_SIZE, encode_move_one_hot
from vm_compiler.reference_executor import execute_artifact_reference_values
from vm_compiler.site_semantics import build_site_semantics


SITE_VERSIONED_ASSETS = (
    "styles.css",
    "i18n.js",
    "trace_model.js",
    "recurrent_inspector.js",
    "app.js",
    "matrix_inspector.js",
    "recurrent_trace.json",
    "numeric_trace.json",
)


def asset_version(path: Path) -> str:
    """Return a platform-stable content version for a static UTF-8 asset."""
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def refresh_site_asset_versions(site: Path) -> None:
    """Fingerprint every static dependency referenced by the published HTML."""
    index = site / "index.html"
    html = index.read_text(encoding="utf-8")
    names = "|".join(re.escape(name) for name in SITE_VERSIONED_ASSETS)
    pattern = re.compile(rf'(?P<prefix>(?:src|href)="\./(?P<name>{names}))(?:\?v=[0-9a-f]{{16}})?(?P<suffix>")')
    seen: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        seen.add(name)
        return f"{match.group('prefix')}?v={asset_version(site / name)}{match.group('suffix')}"

    updated = pattern.sub(replace, html)
    missing = set(SITE_VERSIONED_ASSETS) - seen
    if missing:
        raise ValueError(f"site index is missing versioned asset references: {sorted(missing)}")
    index.write_text(updated, encoding="utf-8", newline="\n")


def _source_provenance(artifact) -> dict[str, object]:
    source_names = ("compiler.py", "state_circuit.py", "reference_executor.py", "site_trace.py", "site_semantics.py")
    source_root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in source_names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((source_root / name).read_bytes())
        digest.update(b"\0")
    return {
        "executor": "vm_compiler.reference_executor",
        "dtype": "float32",
        "artifact_sha256": hashlib.sha256(artifact.to_bytes()).hexdigest(),
        "source_sha256": digest.hexdigest(),
        "source_identifiers": [f"vm_compiler.{name.removesuffix('.py')}" for name in source_names],
    }


def _attributes_info(operation) -> dict[str, object]:
    attributes = operation.attributes
    if operation.opcode == OpCode.ROW_ROUTE:
        start, end, *stride = attributes
        return {"start": start, "end": end, "stride": stride[0] if stride else 1, "axis": "row"}
    if operation.opcode in (OpCode.TOKEN_PROJECT, OpCode.OUTPUT_PROJECT):
        return {"frozen": f"w{attributes[0]}", "axis": "last_dimension"}
    if operation.opcode == OpCode.POSITION_ADD:
        return {"frozen": f"w{attributes[0]}", "axis": "elementwise"}
    if operation.opcode == OpCode.FROZEN_EXPAND:
        return {"frozen": f"w{attributes[0]}", "axis": "batch"}
    if operation.opcode == OpCode.ROW_CONCAT:
        return {"axis": "row"}
    if operation.opcode == OpCode.HARDMAX_STE:
        return {"temperature": attributes[0] / 1000.0, "axis": "last_dimension", "forward": "lowest-index argmax tie"}
    if operation.opcode == OpCode.HULL_ATTN_2D:
        return {
            "top_k": attributes[0],
            "temperature": attributes[1] / 1000.0,
            "candidate_count": len(attributes) - 2,
            "candidate_first": attributes[2],
            "candidate_last": attributes[-1],
            "forward": "hard argmax value read",
            "backward": "selected-top-k softmax surrogate for Q, K and V",
        }
    return {"axis": "elementwise"}


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


def _number(value: float) -> int | float:
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def _matrix(tensor: torch.Tensor) -> dict[str, object]:
    array = tensor.detach().cpu().numpy()
    coordinates = numpy.argwhere(array != 0)
    return {
        "shape": list(array.shape),
        "layout": "COO",
        "nnz": int(coordinates.shape[0]),
        "entries": [[*[int(axis) for axis in coordinate], _number(array[tuple(coordinate)])] for coordinate in coordinates],
    }


def _equation(operation) -> str:
    output = f"v{operation.outputs[0]}"
    inputs = [f"v{value}" for value in operation.inputs]
    if operation.opcode == OpCode.ROW_ROUTE:
        start, end, *stride = operation.attributes
        return f"{output} = {inputs[0]}[:, {start}:{end}:{stride[0] if stride else 1}, :]"
    if operation.opcode in (OpCode.TOKEN_PROJECT, OpCode.OUTPUT_PROJECT):
        return f"{output} = {inputs[0]} @ w{operation.attributes[0]}"
    if operation.opcode == OpCode.POSITION_ADD:
        return f"{output} = {inputs[0]} + w{operation.attributes[0]}"
    if operation.opcode == OpCode.RESIDUAL_ADD:
        return f"{output} = {inputs[0]} + {inputs[1]}"
    if operation.opcode == OpCode.FROZEN_EXPAND:
        return f"{output} = expand_batch(w{operation.attributes[0]})"
    if operation.opcode == OpCode.ROW_CONCAT:
        return f"{output} = concat_rows({', '.join(inputs)})"
    if operation.opcode == OpCode.HARDMAX_STE:
        return f"{output} = one_hot(argmax({inputs[0]}))"
    if operation.opcode == OpCode.HULL_ATTN_2D:
        return f"{output} = hardmax({inputs[0]} @ transpose({inputs[1]})) @ {inputs[2]}"
    raise ValueError(f"unsupported trace opcode {operation.opcode.name}")


def _sample(operation, values, tensors) -> dict[str, object]:
    output = values[operation.outputs[0]].detach()
    nonzero = torch.nonzero(output, as_tuple=False)
    index = [int(axis) for axis in (nonzero[0].tolist() if len(nonzero) else [0] * output.ndim)]
    result: dict[str, object] = {"output_index": index, "output_value": _number(output[tuple(index)].item())}
    if operation.opcode == OpCode.TOKEN_PROJECT:
        source = values[operation.inputs[0]].detach()
        weight = tensors[operation.attributes[0]].detach()
        prefix, column = index[:-1], index[-1]
        vector = source[tuple(prefix)]
        products = vector * weight[:, column]
        terms = torch.nonzero(products, as_tuple=False).flatten().tolist()
        result["identity"] = "sum_k X[...,k] * W[k,j]"
        result["terms"] = [[int(k), _number(vector[k].item()), _number(weight[k, column].item()), _number(products[k].item())] for k in terms]
    elif operation.opcode in (OpCode.POSITION_ADD, OpCode.RESIDUAL_ADD):
        left = values[operation.inputs[0]].detach()[tuple(index)].item()
        right = (tensors[operation.attributes[0]].detach()[tuple(index[1:])].item() if operation.opcode == OpCode.POSITION_ADD else values[operation.inputs[1]].detach()[tuple(index)].item())
        result["identity"] = "left + right"
        result["terms"] = [_number(left), _number(right)]
    elif operation.opcode == OpCode.HULL_ATTN_2D:
        query = values[operation.inputs[0]].detach()
        keys = values[operation.inputs[1]].detach()
        batch, row, channel = index
        scores = query[batch, row] @ keys[batch].transpose(0, 1)
        winner = int(torch.argmax(scores).item())
        result["identity"] = "argmax_j(sum_d Q[row,d] * K[j,d]); output = V[winner,channel]"
        result["winner"] = winner
        result["score_terms"] = [[d, _number(query[batch, row, d].item()), _number(keys[batch, winner, d].item())] for d in range(query.shape[-1])]
        result["winner_score"] = _number(scores[winner].item())
    return result


def build_numeric_site_trace() -> dict[str, object]:
    """Run one exact fixture and export every frozen matrix and SSA value."""
    moves = ((52, 54, PIECE_CHANNELS["WHITE_PAWN"]), (47, 45, PIECE_CHANNELS["BLACK_PAWN"]), (54, 45, PIECE_CHANNELS["WHITE_PAWN"]))
    vm_input = numpy.zeros((INPUT_ROWS, VOCAB_SIZE), dtype=numpy.float32)
    vm_input[:, 0] = 1.0
    vm_input[3:12] = numpy.concatenate([encode_move_one_hot(*move) for move in moves])
    artifact = build_position_reconstruction_artifact()
    _, values, tensors = execute_artifact_reference_values(artifact, torch.tensor(vm_input[None]))
    operations = []
    derived: dict[str, dict[str, object]] = {}
    for index, operation in enumerate(artifact.operations, start=1):
        frozen = []
        if operation.opcode in (OpCode.TOKEN_PROJECT, OpCode.OUTPUT_PROJECT, OpCode.POSITION_ADD, OpCode.FROZEN_EXPAND):
            frozen.append(f"w{operation.attributes[0]}")
        derived_names: list[str] = []
        if operation.opcode == OpCode.HULL_ATTN_2D:
            queries = values[operation.inputs[0]].detach()
            keys = values[operation.inputs[1]].detach()
            candidates = torch.tensor(operation.attributes[2:], dtype=torch.long)
            selected_keys = keys.index_select(1, candidates)
            scores = queries @ selected_keys.transpose(1, 2)
            attention = torch.nn.functional.one_hot(
                torch.argmax(scores, dim=-1), scores.shape[-1]
            ).to(scores.dtype)
            score_name = f"op{index}_scores"
            attention_name = f"op{index}_attention"
            derived[score_name] = _matrix(scores)
            derived[attention_name] = _matrix(attention)
            derived_names.extend((score_name, attention_name))
        operations.append({"index": index, "opcode": operation.opcode.name, "equation": _equation(operation), "inputs": [f"v{value}" for value in operation.inputs], "frozen": frozen, "derived": derived_names, "output": f"v{operation.outputs[0]}", "attributes_info": _attributes_info(operation), "sample": _sample(operation, values, tensors)})
    return {
        "artifact": "position_latest_event_v1",
        "operation_count": len(operations),
        "runtime_opcodes": sorted({operation["opcode"] for operation in operations}),
        "final_output": {"value": 45, "shape": ["B", 64, 128]},
        "fixture": {"moves": [list(move) for move in moves], "encoding": "one-hot"},
        "format": "Exact nonzero entries in zero-based COO; omitted entries are exactly zero.",
        "provenance": _source_provenance(artifact),
        "semantics": build_site_semantics(artifact),
        "tensors": {f"w{index}": {"name": record.name, **_matrix(tensors[index])} for index, record in enumerate(artifact.tensors)},
        "values": {f"v{index}": _matrix(value) for index, value in values.items()},
        "derived": derived,
        "operations": operations,
    }


def main() -> None:
    site = Path(__file__).parents[1] / "site"
    (site / "artifact_trace.json").write_text(
        json.dumps(build_site_trace(), indent=2, separators=(",", ": ")) + "\n",
        encoding="utf-8",
    )
    (site / "numeric_trace.json").write_text(json.dumps(build_numeric_site_trace(), separators=(",", ":")) + "\n", encoding="utf-8")
    refresh_site_asset_versions(site)


if __name__ == "__main__":
    main()
