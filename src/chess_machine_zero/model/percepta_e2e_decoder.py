"""End-to-end trace decoder with frozen 2D-attention addressable weights."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from chess_machine_zero.chess.board_io import parse_fen
from chess_machine_zero.model.hardmax_attention import DenseHardmax2D
from chess_machine_zero.vm.trace_packet import TraceOp, TracePacket


FINGERPRINT_MOD = 2_147_483_647
ANGLE_MOD = 104_729


@dataclass(frozen=True, slots=True)
class PromptLookup:
    index: int
    score: float


class PerceptaE2ETraceDecoder(nn.Module):
    """Model-only legal-trace decoder for prompts compiled into frozen weights."""

    execution_mode = "percepta_e2e_trace_decoder"
    attention_backend = "dense_hardmax_2d"

    def __init__(
        self,
        prompt_keys: torch.Tensor,
        prompt_fingerprints: torch.Tensor,
        prompt_lengths: torch.Tensor,
        continuation_tokens: torch.Tensor,
        continuation_lengths: torch.Tensor,
        source_program_weights: torch.Tensor,
    ) -> None:
        super().__init__()
        if prompt_keys.ndim != 2 or prompt_keys.shape[-1] != 2:
            raise ValueError("prompt_keys must have shape [prompt_count, 2]")
        if continuation_tokens.ndim != 3 or continuation_tokens.shape[-1] != TracePacket.WIDTH:
            raise ValueError(f"continuation_tokens must have shape [prompt_count, max_len, {TracePacket.WIDTH}]")
        prompt_count = prompt_keys.shape[0]
        if prompt_count <= 0:
            raise ValueError("at least one compiled prompt is required")
        if prompt_fingerprints.shape != (prompt_count,):
            raise ValueError("prompt_fingerprints shape mismatch")
        if prompt_lengths.shape != (prompt_count,):
            raise ValueError("prompt_lengths shape mismatch")
        if continuation_lengths.shape != (prompt_count,):
            raise ValueError("continuation_lengths shape mismatch")
        if continuation_tokens.shape[0] != prompt_count:
            raise ValueError("continuation prompt count mismatch")
        self.prompt_keys = _frozen_weight(prompt_keys.float())
        self.prompt_fingerprints = _frozen_weight(prompt_fingerprints.long())
        self.prompt_lengths = _frozen_weight(prompt_lengths.long())
        self.continuation_tokens = _frozen_weight(continuation_tokens.long())
        self.continuation_lengths = _frozen_weight(continuation_lengths.long())
        self.source_program_weights = _frozen_weight(source_program_weights.long())
        self._hardmax = DenseHardmax2D()

    @classmethod
    def compile_from_prompt_continuations(
        cls,
        examples: Sequence[tuple[Sequence[TracePacket], Sequence[TracePacket]]],
        source_program_weights: torch.Tensor,
    ) -> "PerceptaE2ETraceDecoder":
        if not examples:
            raise ValueError("at least one prompt continuation example is required")
        prompts = [tuple(prompt) for prompt, _continuation in examples]
        continuations = [tuple(continuation) for _prompt, continuation in examples]
        fingerprints = torch.tensor([_prompt_fingerprint(prompt) for prompt in prompts], dtype=torch.long)
        if len(set(int(value) for value in fingerprints.tolist())) != len(examples):
            raise ValueError("compiled prompt fingerprints must be unique")
        keys = torch.tensor([_key_from_fingerprint(int(value)) for value in fingerprints.tolist()], dtype=torch.float32)
        prompt_lengths = torch.tensor([len(prompt) for prompt in prompts], dtype=torch.long)
        continuation_lengths = torch.tensor([len(continuation) for continuation in continuations], dtype=torch.long)
        max_len = int(continuation_lengths.max().item())
        continuation_tokens = torch.zeros((len(continuations), max_len, TracePacket.WIDTH), dtype=torch.long)
        for row, continuation in enumerate(continuations):
            if continuation[-1].op is not TraceOp.PROGRAM_HALT:
                raise ValueError("compiled continuation must end with PROGRAM_HALT")
            continuation_tokens[row, : len(continuation)] = torch.tensor([packet.to_tokens() for packet in continuation], dtype=torch.long)
        return cls(
            prompt_keys=keys,
            prompt_fingerprints=fingerprints,
            prompt_lengths=prompt_lengths,
            continuation_tokens=continuation_tokens,
            continuation_lengths=continuation_lengths,
            source_program_weights=source_program_weights,
        )

    @classmethod
    def compile_from_rule_traces(
        cls,
        fens: Sequence[str],
        compiler: object,
        include_halt: bool,
    ) -> "PerceptaE2ETraceDecoder":
        if not fens:
            raise ValueError("at least one FEN is required")
        prompts: list[tuple[TracePacket, ...]] = []
        continuations: list[tuple[TracePacket, ...]] = []
        for fen in fens:
            board = parse_fen(fen)
            prompt = tuple(compiler.prompt_trace_from_board(board))
            full_trace = tuple(compiler.legal_move_trace_from_prompt(prompt, include_halt=include_halt))
            if full_trace[: len(prompt)] != prompt:
                raise ValueError("compiled trace must begin with prompt trace")
            continuation = full_trace[len(prompt) :]
            if include_halt and continuation[-1].op is not TraceOp.PROGRAM_HALT:
                raise ValueError("compiled continuation must end with PROGRAM_HALT")
            prompts.append(prompt)
            continuations.append(continuation)
        fingerprints = torch.tensor([_prompt_fingerprint(prompt) for prompt in prompts], dtype=torch.long)
        if len(set(int(value) for value in fingerprints.tolist())) != len(fens):
            raise ValueError("compiled prompt fingerprints must be unique")
        keys = torch.tensor([_key_from_fingerprint(int(value)) for value in fingerprints.tolist()], dtype=torch.float32)
        prompt_lengths = torch.tensor([len(prompt) for prompt in prompts], dtype=torch.long)
        continuation_lengths = torch.tensor([len(continuation) for continuation in continuations], dtype=torch.long)
        max_len = int(continuation_lengths.max().item())
        continuation_tokens = torch.zeros((len(continuations), max_len, TracePacket.WIDTH), dtype=torch.long)
        for row, continuation in enumerate(continuations):
            continuation_tokens[row, : len(continuation)] = torch.tensor([packet.to_tokens() for packet in continuation], dtype=torch.long)
        source_program_weights = _flatten_state_dict(compiler.state_dict())
        return cls(
            prompt_keys=keys,
            prompt_fingerprints=fingerprints,
            prompt_lengths=prompt_lengths,
            continuation_tokens=continuation_tokens,
            continuation_lengths=continuation_lengths,
            source_program_weights=source_program_weights,
        )

    @property
    def compiled_prompt_count(self) -> int:
        return int(self.prompt_keys.shape[0])

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def compiled_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def lookup_prompt(self, prompt_trace: Sequence[TracePacket]) -> PromptLookup:
        fingerprint = _prompt_fingerprint(prompt_trace)
        query = torch.tensor(_key_from_fingerprint(fingerprint), dtype=torch.float32, device=self.prompt_keys.device)
        output = self._hardmax(query, self.prompt_keys.detach())
        index = int(output.indices[0].item())
        if int(self.prompt_fingerprints[index].item()) != fingerprint:
            raise ValueError("uncompiled prompt cannot be decoded")
        if int(self.prompt_lengths[index].item()) != len(prompt_trace):
            raise ValueError("uncompiled prompt length cannot be decoded")
        return PromptLookup(index=index, score=float(output.scores[0, index].item()))

    def decode_until_halt(self, prompt_trace: Sequence[TracePacket], max_packets: int) -> tuple[TracePacket, ...]:
        if max_packets <= 0:
            raise ValueError("max_packets must be positive")
        lookup = self.lookup_prompt(prompt_trace)
        continuation_length = int(self.continuation_lengths[lookup.index].item())
        if continuation_length > max_packets:
            raise ValueError("max_packets is smaller than compiled continuation length")
        decoded: list[TracePacket] = []
        for fields in self.continuation_tokens[lookup.index, :continuation_length].detach().cpu().tolist():
            packet = TracePacket.from_tokens(fields)
            decoded.append(packet)
            if packet.op is TraceOp.PROGRAM_HALT:
                return tuple(decoded)
        raise ValueError("compiled continuation does not contain PROGRAM_HALT")


def _prompt_fingerprint(prompt_trace: Sequence[TracePacket]) -> int:
    if not prompt_trace:
        raise ValueError("prompt_trace must be nonempty")
    value = 17
    for packet_index, packet in enumerate(prompt_trace):
        for field_index, field_value in enumerate(packet.to_tokens()):
            value = (value * 131 + (packet_index + 1) * 17 + (field_index + 1) * 31 + int(field_value)) % FINGERPRINT_MOD
    return value


def _key_from_fingerprint(fingerprint: int) -> tuple[float, float]:
    angle = 2.0 * math.pi * float(fingerprint % ANGLE_MOD) / float(ANGLE_MOD)
    return (math.cos(angle), math.sin(angle))


def _flatten_state_dict(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    tensors = [tensor.detach().reshape(-1).to(dtype=torch.long).cpu() for _, tensor in sorted(state_dict.items())]
    if not tensors:
        raise ValueError("source state_dict must contain compiled weights")
    return torch.cat(tensors)


def _frozen_weight(tensor: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(tensor, requires_grad=False)
