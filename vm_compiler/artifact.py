"""Strict binary container for immutable FP4 tensors and explicit VM graphs."""

from __future__ import annotations

import hashlib
import math
import operator
import struct
from functools import reduce
from dataclasses import dataclass

from .graph import OpCode, Operation

MAGIC = b"CMZVM001"
VERSION = 1
HEADER = struct.Struct("<8sIIII32s")
TENSOR_PREFIX = struct.Struct("<HBBHHI")
OP_PREFIX = struct.Struct("<HBBHH")


class ArtifactError(ValueError):
    pass


def _validate_tensor_layout(tensor: "TensorRecord") -> None:
    element_count = reduce(operator.mul, tensor.shape, 1)
    expected_packed = (element_count + 1) // 2
    expected_scales = (element_count + tensor.block_size - 1) // tensor.block_size
    if len(tensor.packed) != expected_packed:
        raise ArtifactError("tensor packed FP4 length does not match shape")
    if len(tensor.scales) != expected_scales:
        raise ArtifactError("tensor scale count does not match shape and block size")


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shape: tuple[int, ...]
    packed: bytes
    scales: tuple[float, ...]
    block_size: int


@dataclass(frozen=True)
class Artifact:
    tensors: tuple[TensorRecord, ...]
    operations: tuple[Operation, ...]

    def to_bytes(self) -> bytes:
        body = bytearray()
        for tensor in self.tensors:
            name = tensor.name.encode("utf-8")
            if not name or len(name) > 0xFFFF:
                raise ArtifactError("tensor name length is invalid")
            if not 1 <= len(tensor.shape) <= 8 or any(size <= 0 for size in tensor.shape):
                raise ArtifactError("tensor shape is invalid")
            if not tensor.scales or any(not math.isfinite(scale) for scale in tensor.scales):
                raise ArtifactError("tensor scales are invalid")
            if not 1 <= tensor.block_size <= 0xFFFF:
                raise ArtifactError("tensor block size is invalid")
            _validate_tensor_layout(tensor)
            body.extend(
                TENSOR_PREFIX.pack(
                    len(name),
                    len(tensor.shape),
                    0,
                    tensor.block_size,
                    len(tensor.scales),
                    len(tensor.packed),
                )
            )
            body.extend(struct.pack(f"<{len(tensor.shape)}I", *tensor.shape))
            body.extend(name)
            body.extend(struct.pack(f"<{len(tensor.scales)}f", *tensor.scales))
            body.extend(tensor.packed)
        for operation in self.operations:
            body.extend(
                OP_PREFIX.pack(
                    int(operation.opcode),
                    len(operation.inputs),
                    len(operation.outputs),
                    len(operation.attributes),
                    0,
                )
            )
            words = operation.inputs + operation.outputs + operation.attributes
            body.extend(struct.pack(f"<{len(words)}I", *words))
        digest = hashlib.sha256(body).digest()
        return HEADER.pack(
            MAGIC, VERSION, len(self.tensors), len(self.operations), len(body), digest
        ) + body

    @classmethod
    def from_bytes(cls, encoded: bytes) -> "Artifact":
        try:
            return cls._parse(encoded)
        except (UnicodeDecodeError, struct.error, ValueError) as error:
            if isinstance(error, ArtifactError):
                raise
            raise ArtifactError(str(error)) from error

    @classmethod
    def _parse(cls, encoded: bytes) -> "Artifact":
        if len(encoded) < HEADER.size:
            raise ArtifactError("artifact is shorter than its header")
        magic, version, tensor_count, operation_count, body_size, digest = HEADER.unpack_from(
            encoded
        )
        if magic != MAGIC or version != VERSION:
            raise ArtifactError("artifact magic or version mismatch")
        body = encoded[HEADER.size :]
        if len(body) != body_size:
            raise ArtifactError("artifact body length mismatch")
        if hashlib.sha256(body).digest() != digest:
            raise ArtifactError("artifact digest mismatch")

        cursor = 0
        tensors = []
        for _ in range(tensor_count):
            name_size, rank, reserved, block_size, scale_count, packed_size = TENSOR_PREFIX.unpack_from(
                body, cursor
            )
            cursor += TENSOR_PREFIX.size
            if reserved or not 1 <= rank <= 8 or not scale_count:
                raise ArtifactError("tensor metadata is invalid")
            shape = struct.unpack_from(f"<{rank}I", body, cursor)
            cursor += rank * 4
            name = body[cursor : cursor + name_size].decode("utf-8")
            cursor += name_size
            scales = struct.unpack_from(f"<{scale_count}f", body, cursor)
            cursor += scale_count * 4
            packed = bytes(body[cursor : cursor + packed_size])
            cursor += packed_size
            if not name or any(size <= 0 for size in shape) or any(
                not math.isfinite(scale) for scale in scales
            ):
                raise ArtifactError("tensor record is invalid")
            tensor = TensorRecord(name, tuple(shape), packed, tuple(scales), block_size)
            _validate_tensor_layout(tensor)
            tensors.append(tensor)

        operations = []
        for _ in range(operation_count):
            opcode_raw, input_count, output_count, attribute_count, reserved = OP_PREFIX.unpack_from(
                body, cursor
            )
            cursor += OP_PREFIX.size
            if reserved:
                raise ArtifactError("operation reserved field is nonzero")
            try:
                opcode = OpCode(opcode_raw)
            except ValueError as error:
                raise ArtifactError(f"unknown opcode {opcode_raw}") from error
            word_count = input_count + output_count + attribute_count
            words = struct.unpack_from(f"<{word_count}I", body, cursor)
            cursor += word_count * 4
            inputs = tuple(words[:input_count])
            outputs = tuple(words[input_count : input_count + output_count])
            attributes = tuple(words[input_count + output_count :])
            operations.append(Operation(opcode, inputs, outputs, attributes))
        if cursor != len(body):
            raise ArtifactError("artifact contains trailing bytes")
        return cls(tuple(tensors), tuple(operations))
