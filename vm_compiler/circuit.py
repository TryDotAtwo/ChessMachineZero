"""Offline rank-three frozen circuit synthesis; no runtime tensor inspection.

Value IDs name [B,rows,columns] matrices. Shape/weight decisions happen only at
compile time. Boolean helpers require hard 0/1 operands in forward; derivatives
are the declared hardmax softmax surrogate, not Boolean derivatives.
"""

from __future__ import annotations

import math

import numpy as np

from .artifact import Artifact, TensorRecord
from .fp4 import decode_e2m1, encode_e2m1
from .graph import OpCode, Operation


def exact_record(name: str, values) -> TensorRecord:
    """Pack exact FP32 constants; never round an unsupported value to FP4."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or any(size <= 0 for size in array.shape):
        raise ValueError("frozen matrix must have two positive extents")
    if not np.isfinite(array).all():
        raise ValueError("frozen matrix values must be finite")
    flat = array.reshape(-1)
    padded = np.pad(flat, (0, flat.size % 2))
    try:
        packed = encode_e2m1(padded).tobytes()
        block_size = 4096
        scales = (1.,) * math.ceil(flat.size / block_size)
    except ValueError:
        # The wire format has a uint16 scale count. Fail explicitly rather than
        # overflowing that field or silently approximating a large rule matrix.
        if flat.size > 65535:
            raise ValueError("non-lattice matrix exceeds exact per-element scale capacity") from None
        packed = encode_e2m1(np.sign(padded)).tobytes()
        scales = tuple(float(abs(value)) if value != 0 else 1. for value in flat)
        block_size = 1
    if len(scales) > 65535:
        raise ValueError("frozen matrix exceeds wire scale capacity")
    return TensorRecord(name, array.shape, packed, scales, block_size)


def _broadcast(left, right):
    if any(a != b and a != 1 and b != 1 for a, b in zip(left, right)):
        raise ValueError(f"circuit broadcast mismatch: {left} and {right}")
    return tuple(max(a, b) for a, b in zip(left, right))


class Circuit:
    """Small deterministic builder lowering exclusively to generic wire ops."""

    def __init__(self):
        self.input = 0
        self.shapes: dict[int, tuple[int, int]] = {0: (2048, 128)}
        self.semantics = {0: {"name": "request_and_previous_context", "producer": None}}
        self.named_outputs: dict[str, int] = {}
        self.operations: list[Operation] = []
        self.tensors: list[TensorRecord] = []
        self._weights: dict[tuple, int] = {}

    def _weight(self, name, values):
        record = exact_record(name, values)
        key = (record.shape, record.packed, record.scales, record.block_size)
        if key not in self._weights:
            self._weights[key] = len(self.tensors)
            self.tensors.append(record)
        return self._weights[key]

    def _emit(self, opcode, inputs, shape, attributes=(), name=None):
        output = len(self.shapes)
        self.operations.append(Operation(opcode, tuple(inputs), (output,), tuple(attributes)))
        self.shapes[output] = tuple(shape)
        self.semantics[output] = {"name": name or opcode.name.lower(), "producer": len(self.operations) - 1}
        return output

    def route(self, value, start, end, stride=1, name=None):
        rows, columns = self.shapes[value]
        if not all(isinstance(v, int) for v in (start, end, stride)) or not 0 <= start < end <= rows or stride <= 0:
            raise ValueError("circuit route bounds mismatch")
        return self._emit(OpCode.ROW_ROUTE, (value,), ((end - start + stride - 1) // stride, columns),
                          (start, end, stride), name)

    def constant(self, name, array):
        weight = self._weight(name, array)
        return self._emit(OpCode.FROZEN_EXPAND, (self.input,), self.tensors[weight].shape, (weight,), name)

    def project(self, value, weights, name=None):
        matrix = np.asarray(weights, dtype=np.float32)
        rows, columns = self.shapes[value]
        if matrix.ndim != 2 or columns != matrix.shape[0]:
            raise ValueError("circuit projection shape mismatch")
        weight = self._weight(name or "projection", matrix)
        return self._emit(OpCode.TOKEN_PROJECT, (value,), (rows, matrix.shape[1]), (weight,), name)

    def transpose(self, value, name=None):
        return self._emit(OpCode.MATRIX_TRANSPOSE, (value,), self.shapes[value][::-1], name=name)

    def reshape(self, value, rows, columns, name=None):
        if not all(isinstance(v, int) and v > 0 for v in (rows, columns)) or rows * columns != math.prod(self.shapes[value]):
            raise ValueError("circuit reshape element count mismatch")
        return self._emit(OpCode.MATRIX_RESHAPE, (value,), (rows, columns), (rows, columns), name)

    def matmul(self, left, right, name=None):
        if self.shapes[left][1] != self.shapes[right][0]:
            raise ValueError("circuit matmul shape mismatch")
        return self._emit(OpCode.MATRIX_MATMUL, (left, right),
                          (self.shapes[left][0], self.shapes[right][1]), name=name)

    def add(self, left, right, name=None):
        shape = _broadcast(self.shapes[left], self.shapes[right])
        return self._emit(OpCode.RESIDUAL_ADD, (left, right), shape, name=name)

    def bias(self, value, weights, name=None):
        matrix = np.asarray(weights, dtype=np.float32)
        if matrix.ndim == 0:
            matrix = matrix.reshape(1, 1)
        if matrix.ndim != 2:
            raise ValueError("circuit bias must be scalar or matrix")
        shape = _broadcast(self.shapes[value], matrix.shape)
        weight = self._weight(name or "bias", matrix)
        return self._emit(OpCode.POSITION_ADD, (value,), shape, (weight,), name)

    def scale(self, value, multiplier, name=None):
        return self.project(value, np.eye(self.shapes[value][1], dtype=np.float32) * multiplier, name)

    def concat_rows(self, values, name=None):
        values = tuple(values)
        if not values or len(values) > 255 or len({self.shapes[v][1] for v in values}) != 1:
            raise ValueError("circuit concat rows shape or input capacity mismatch")
        return self._emit(OpCode.ROW_CONCAT, values,
                          (sum(self.shapes[v][0] for v in values), self.shapes[values[0]][1]), name=name)

    def concat_columns(self, values, name=None):
        values = tuple(values)
        if not values or len({self.shapes[v][0] for v in values}) != 1:
            raise ValueError("circuit concat columns shape mismatch")
        return self.transpose(self.concat_rows(self.transpose(v) for v in values), name)

    def hardmax(self, value, temperature=.5, name=None):
        encoded = round(temperature * 1000) if math.isfinite(temperature) else 0
        if not 1 <= encoded <= 0xffffffff or encoded / 1000 != temperature:
            raise ValueError("hardmax temperature must be positive exact milliseconds")
        return self._emit(OpCode.HARDMAX_STE, (value,), self.shapes[value], (encoded,), name)

    def threshold(self, value, boundary, temperature=.5, name=None):
        """Elementwise hard (x > boundary), with independent two-logit STE."""
        shape = self.shapes[value]
        flat = self.reshape(value, math.prod(shape), 1)
        logits = self.project(flat, [[0., 1.]])
        logits = self.bias(logits, [[0., -boundary]])
        bits = self.project(self.hardmax(logits, temperature), [[0.], [1.]])
        return self.reshape(bits, *shape, name=name)

    def logical_not(self, value, name=None):
        return self.bias(self.scale(value, -1.), 1., name)

    def logical_and(self, left, right, name=None):
        return self.threshold(self.add(left, right), 1.5, name=name)

    def logical_or(self, left, right, name=None):
        return self.threshold(self.add(left, right), .5, name=name)

    def select(self, condition, when_true, when_false, name=None):
        """Boolean mux; condition and payloads must be hard 0/1 in forward."""
        true_bits = self.logical_and(condition, when_true)
        false_bits = self.logical_and(self.logical_not(condition), when_false)
        return self.add(true_bits, false_bits, name)

    def attention(self, queries, keys, values, competitors, temperature, candidates, name=None):
        q, k, v = (self.shapes[value] for value in (queries, keys, values))
        candidates = tuple(candidates)
        if q[1] != 2 or k[1] != 2 or k[0] != v[0]:
            raise ValueError("circuit attention shape mismatch")
        if not 1 <= competitors <= min(32, len(candidates)) or len(set(candidates)) != len(candidates):
            raise ValueError("circuit attention candidate capacity mismatch")
        if any(index < 0 or index >= k[0] for index in candidates):
            raise ValueError("circuit attention candidate out of range")
        encoded = round(temperature * 1000) if math.isfinite(temperature) else 0
        if not 1 <= encoded <= 0xffffffff or encoded / 1000 != temperature:
            raise ValueError("attention temperature must be positive exact milliseconds")
        return self._emit(OpCode.HULL_ATTN_2D, (queries, keys, values), (q[0], v[1]),
                          (competitors, encoded, *candidates), name)

    def include_graph(self, artifact: Artifact, source=None):
        """Inline a supported generic artifact, remapping IDs at compile time.

        Constants are decoded/repacked exactly offline, not executed. Unsupported
        schemas fail closed; this is not a fallback interpreter in the runtime.
        """
        artifact = Artifact.from_bytes(artifact.to_bytes())
        if not artifact.operations:
            raise ValueError("cannot include an empty graph")
        source = self.input if source is None else source
        values = {0: source}
        constants = []
        for record in artifact.tensors:
            count = math.prod(record.shape)
            codes = decode_e2m1(np.frombuffer(record.packed, dtype=np.uint8), count)
            scales = np.repeat(np.asarray(record.scales, dtype=np.float32), record.block_size)[:count]
            constants.append((codes * scales).reshape(record.shape))
        schemas = {
            OpCode.TOKEN_PROJECT: (1, 1), OpCode.OUTPUT_PROJECT: (1, 1),
            OpCode.POSITION_ADD: (1, 1), OpCode.RESIDUAL_ADD: (2, 0),
            OpCode.FROZEN_EXPAND: (1, 1), OpCode.HARDMAX_STE: (1, 1),
            OpCode.MATRIX_TRANSPOSE: (1, 0), OpCode.MATRIX_RESHAPE: (1, 2),
            OpCode.MATRIX_MATMUL: (2, 0),
        }
        for op in artifact.operations:
            if len(op.outputs) != 1 or op.outputs[0] in values or any(v not in values for v in op.inputs):
                raise ValueError("included graph has invalid SSA definitions")
            args = tuple(values[v] for v in op.inputs)
            attrs = op.attributes
            if op.opcode in schemas and (len(args), len(attrs)) != schemas[op.opcode]:
                raise ValueError("included graph operation schema mismatch")
            if op.opcode == OpCode.ROW_ROUTE and len(args) == 1 and len(attrs) in (2, 3):
                result = self.route(args[0], *attrs)
            elif op.opcode in (OpCode.TOKEN_PROJECT, OpCode.OUTPUT_PROJECT):
                result = self.project(args[0], constants[attrs[0]], artifact.tensors[attrs[0]].name)
            elif op.opcode == OpCode.POSITION_ADD:
                result = self.bias(args[0], constants[attrs[0]], artifact.tensors[attrs[0]].name)
            elif op.opcode == OpCode.RESIDUAL_ADD:
                result = self.add(*args)
            elif op.opcode == OpCode.FROZEN_EXPAND and op.inputs == (0,):
                result = self.constant(artifact.tensors[attrs[0]].name, constants[attrs[0]])
            elif op.opcode == OpCode.ROW_CONCAT and not attrs:
                result = self.concat_rows(args)
            elif op.opcode == OpCode.HARDMAX_STE:
                result = self.hardmax(args[0], attrs[0] / 1000)
            elif op.opcode == OpCode.HULL_ATTN_2D and len(args) == 3 and len(attrs) > 2:
                result = self.attention(*args, attrs[0], attrs[1] / 1000, attrs[2:])
            elif op.opcode == OpCode.MATRIX_TRANSPOSE:
                result = self.transpose(*args)
            elif op.opcode == OpCode.MATRIX_RESHAPE:
                result = self.reshape(args[0], *attrs)
            elif op.opcode == OpCode.MATRIX_MATMUL:
                result = self.matmul(*args)
            else:
                raise ValueError(f"unsupported included operation/schema: {op.opcode.name}")
            values[op.outputs[0]] = result
        return result

    def artifact(self, output) -> Artifact:
        if output not in self.shapes:
            raise ValueError("undefined circuit output")
        if not self.operations or self.operations[-1].outputs[0] != output:
            self.route(output, 0, self.shapes[output][0], name="output")
        return Artifact(tuple(self.tensors), tuple(self.operations))

    def memory_estimate(self, batch_size=1):
        """Conservative logical payload count, NOT peak CUDA allocator usage.

        Includes all retained values, even views; excludes autograd saved tensors,
        attention scores/competitors, temporary workspaces and allocator overhead.
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch size must be a positive integer")
        return {
            "frozen_fp32_bytes": sum(math.prod(t.shape) * 4 for t in self.tensors),
            "retained_values_fp32_bytes": sum(math.prod(s) for s in self.shapes.values()) * batch_size * 4,
            "excludes_autograd_and_workspace": True,
        }
