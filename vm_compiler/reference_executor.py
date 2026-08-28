"""Development-only differentiable oracle for artifact graph semantics."""

from __future__ import annotations

import numpy
import torch
import torch.nn.functional as functional

from .artifact import Artifact, TensorRecord
from .fp4 import decode_e2m1
from .graph import OpCode


def _materialize(record: TensorRecord, source: torch.Tensor) -> torch.Tensor:
    count = int(numpy.prod(record.shape))
    decoded = decode_e2m1(numpy.frombuffer(record.packed, dtype=numpy.uint8), count)
    scales = numpy.repeat(numpy.asarray(record.scales, dtype=numpy.float32), record.block_size)
    values = (decoded * scales[:count]).reshape(record.shape)
    return torch.as_tensor(values, dtype=source.dtype, device=source.device)


def execute_artifact_reference_values(
    artifact: Artifact, source: torch.Tensor
) -> tuple[torch.Tensor, dict[int, torch.Tensor], tuple[torch.Tensor, ...]]:
    """Execute the graph and retain every SSA value for development inspection."""
    tensors = tuple(_materialize(record, source) for record in artifact.tensors)
    values: dict[int, torch.Tensor] = {0: source}
    final = 0
    for operation in artifact.operations:
        inputs = [values[index] for index in operation.inputs]
        if operation.opcode == OpCode.ROW_ROUTE:
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
            if len(inputs) == 2:
                logits = logits.masked_fill(~inputs[1].bool(), -torch.inf)
            hard_indices = torch.argmax(logits, dim=-1)
            hard = functional.one_hot(hard_indices, logits.shape[-1]).to(logits.dtype)
            soft = torch.softmax(logits / (operation.attributes[0] / 1000.0), dim=-1)
            result = hard + (soft - soft.detach())
        elif operation.opcode == OpCode.HULL_ATTN_2D and len(inputs) == 3:
            competitor_count, temperature = operation.attributes[:2]
            candidates = torch.as_tensor(
                operation.attributes[2:], dtype=torch.long, device=source.device
            )
            keys = inputs[1].index_select(1, candidates)
            selected_values = inputs[2].index_select(1, candidates)
            logits = torch.matmul(inputs[0], keys.transpose(1, 2))
            hard_indices = torch.argmax(logits, dim=-1)
            hard = functional.one_hot(hard_indices, logits.shape[-1]).to(logits.dtype)
            soft = torch.softmax(logits / (temperature / 1000.0), dim=-1)
            weights = hard + (soft - soft.detach())
            result = torch.matmul(weights, selected_values)
            del competitor_count
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
