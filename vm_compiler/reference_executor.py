"""Development-only differentiable oracle for artifact graph semantics."""

from __future__ import annotations

import math

import numpy
import torch

from .artifact import Artifact, TensorRecord
from .fp4 import decode_e2m1
from .graph import OpCode
from .ste import masked_hardmax_ste


def _materialize(record: TensorRecord, source: torch.Tensor) -> torch.Tensor:
    count = int(numpy.prod(record.shape))
    decoded = decode_e2m1(numpy.frombuffer(record.packed, dtype=numpy.uint8), count)
    scales = numpy.repeat(numpy.asarray(record.scales, dtype=numpy.float32), record.block_size)
    values = (decoded * scales[:count]).reshape(record.shape)
    return torch.as_tensor(values, dtype=source.dtype, device=source.device)


def attention_2d_ste(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    candidates,
    competitor_count: int,
    temperature: float,
) -> torch.Tensor:
    """Exact hard values with selected-top-k softmax derivatives for Q, K and V.

    This dense development reference ranks ties by original key-row index,
    independently in every batch/query. Only selected competitors contribute to
    the surrogate; the hard winning value contributes no additional derivative.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("attention temperature must be finite and positive")
    candidate_rows = torch.as_tensor(candidates, dtype=torch.long, device=keys.device).sort().values
    if not 1 <= competitor_count <= candidate_rows.numel():
        raise ValueError("attention competitor count exceeds candidate capacity")
    logits = queries @ keys.index_select(1, candidate_rows).transpose(1, 2)
    selected = torch.argsort(logits, dim=-1, descending=True, stable=True)[..., :competitor_count]
    selected_scores = logits.gather(-1, selected)
    selected_rows = candidate_rows[selected]
    batches = torch.arange(queries.shape[0], device=queries.device)[:, None, None]
    selected_values = values[batches, selected_rows]
    probabilities = torch.softmax(selected_scores / temperature, dim=-1)
    soft = (probabilities.unsqueeze(-1) * selected_values).sum(dim=-2)
    hard = selected_values[..., 0, :]
    return hard.detach() + (soft - soft.detach())


def execute_artifact_reference_values(
    artifact: Artifact, source: torch.Tensor
) -> tuple[torch.Tensor, dict[int, torch.Tensor], tuple[torch.Tensor, ...]]:
    """Execute the graph and retain every SSA value for development inspection."""
    tensors = tuple(_materialize(record, source) for record in artifact.tensors)
    values: dict[int, torch.Tensor] = {0: source}
    final = 0
    for operation in artifact.operations:
        inputs = [values[index] for index in operation.inputs]
        if operation.opcode in (OpCode.MATRIX_TRANSPOSE, OpCode.MATRIX_RESHAPE,
                                OpCode.MATRIX_MATMUL, OpCode.GROUPED_MATRIX_MATMUL):
            input_count = 2 if operation.opcode in (
                OpCode.MATRIX_MATMUL, OpCode.GROUPED_MATRIX_MATMUL) else 1
            attribute_count = (2 if operation.opcode == OpCode.MATRIX_RESHAPE else
                               1 if operation.opcode == OpCode.GROUPED_MATRIX_MATMUL else 0)
            if (len(inputs) != input_count or len(operation.outputs) != 1
                    or len(operation.attributes) != attribute_count):
                raise ValueError("artifact matrix operation schema mismatch")
            name = operation.opcode.name.lower()
            if any(value.ndim != 3 for value in inputs):
                raise ValueError(f"artifact {name} rank must be three")
            if operation.opcode == OpCode.MATRIX_TRANSPOSE:
                result = inputs[0].transpose(1, 2)
            elif operation.opcode == OpCode.MATRIX_RESHAPE:
                rows, columns = operation.attributes
                if rows <= 0 or columns <= 0 or rows * columns != inputs[0].shape[1] * inputs[0].shape[2]:
                    raise ValueError("artifact reshape element count mismatch")
                result = inputs[0].reshape(inputs[0].shape[0], rows, columns)
            elif operation.opcode == OpCode.MATRIX_MATMUL:
                if inputs[0].shape[0] != inputs[1].shape[0] or inputs[0].shape[2] != inputs[1].shape[1]:
                    raise ValueError("artifact matmul shape mismatch")
                result = inputs[0] @ inputs[1]
            else:
                groups = operation.attributes[0]
                if (groups <= 0 or inputs[0].shape[0] != inputs[1].shape[0]
                        or inputs[0].shape[1] % groups
                        or inputs[1].shape[1] != groups * inputs[0].shape[2]):
                    raise ValueError("artifact grouped matmul group or shape mismatch")
                batch = inputs[0].shape[0]
                rows = inputs[0].shape[1] // groups
                contraction = inputs[0].shape[2]
                columns = inputs[1].shape[2]
                result = (inputs[0].reshape(batch * groups, rows, contraction)
                          @ inputs[1].reshape(batch * groups, contraction, columns))
                result = result.reshape(batch, groups * rows, columns)
        elif operation.opcode == OpCode.ROW_ROUTE:
            start, end = operation.attributes[:2]
            stride = operation.attributes[2] if len(operation.attributes) == 3 else 1
            result = inputs[0][:, start:end:stride]
        elif operation.opcode in (OpCode.TOKEN_PROJECT, OpCode.OUTPUT_PROJECT):
            result = inputs[0] @ tensors[operation.attributes[0]]
        elif operation.opcode == OpCode.POSITION_ADD:
            result = inputs[0] + tensors[operation.attributes[0]]
        elif operation.opcode == OpCode.RESIDUAL_ADD:
            result = inputs[0] + inputs[1]
        elif operation.opcode == OpCode.FROZEN_EXPAND:
            frozen = tensors[operation.attributes[0]]
            result = frozen.unsqueeze(0).expand(source.shape[0], *frozen.shape)
        elif operation.opcode == OpCode.ROW_CONCAT:
            result = torch.cat(inputs, dim=1)
        elif operation.opcode == OpCode.HARDMAX_STE:
            logits = inputs[0]
            mask = inputs[1].bool() if len(inputs) == 2 else torch.ones_like(logits, dtype=torch.bool)
            result = masked_hardmax_ste(logits, mask, operation.attributes[0] / 1000.0)
        elif operation.opcode == OpCode.HULL_ATTN_2D and len(inputs) == 3:
            competitor_count, temperature = operation.attributes[:2]
            result = attention_2d_ste(
                inputs[0], inputs[1], inputs[2], operation.attributes[2:],
                competitor_count, temperature / 1000.0,
            )
        else:
            raise ValueError(f"reference executor does not support {operation.opcode.name}")
        values[operation.outputs[0]] = result
        final = operation.outputs[0]
    if final == 0:
        raise ValueError("artifact graph is empty")
    return values[final], values, tensors


def execute_artifact_reference(artifact: Artifact, source: torch.Tensor) -> torch.Tensor:
    result, _, _ = execute_artifact_reference_values(artifact, source)
    return result
