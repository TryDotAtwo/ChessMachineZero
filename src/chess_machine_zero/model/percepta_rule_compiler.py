"""Chess rule microprogram compiler for frozen attention execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import nn


class ChessRuleISA(IntEnum):
    TRACE_READ_BOARD = 1
    ATTEND_BOARD_SQUARES = 2
    PIECE_DISPATCH = 3
    PAWN_RULE = 4
    LEAPER_RULE = 5
    RAY_SCAN = 6
    CASTLE_RULE = 7
    ATTACK_TEST = 8
    LEGAL_FILTER = 9
    EMIT_LEGAL_TRACE = 10
    MAKE_MOVE = 11
    EMIT_BOARD_TRANSITION = 12
    TERMINAL_PREDICATES = 13
    EMIT_TERMINAL_TRACE = 14
    HALT = 15


class ProgramEntrypoint(IntEnum):
    LEGAL_TRACE = 1
    MAKE_MOVE_TRACE = 2
    TERMINAL_TRACE = 3


class RuleRegister(IntEnum):
    TRACE = 1
    BOARD = 2
    CANDIDATE = 3
    ATTACK = 4
    LEGAL = 5
    MOVE = 6
    NEXT_BOARD = 7
    TERMINAL = 8


@dataclass(frozen=True, slots=True)
class RuleInstruction:
    entrypoint: ProgramEntrypoint
    op: ChessRuleISA
    source: RuleRegister
    target: RuleRegister
    block_id: int
    head_count: int = 1
    residual_write: bool = False

    def to_row(self) -> tuple[int, int, int, int, int, int]:
        return (
            int(self.entrypoint),
            int(self.op),
            int(self.source),
            int(self.target),
            int(self.block_id),
            int(self.residual_write),
        )


@dataclass(frozen=True, slots=True)
class RuleMicroprogram:
    instructions: tuple[RuleInstruction, ...]
    source_language: str = "chess_rule_isa"

    @property
    def program_hash(self) -> str:
        material = ";".join(",".join(str(value) for value in instruction.to_row()) for instruction in self.instructions)
        return hashlib.sha256(material.encode("ascii")).hexdigest()

    @property
    def entrypoint_names(self) -> tuple[str, ...]:
        return tuple(entrypoint.name for entrypoint in ProgramEntrypoint if any(instruction.entrypoint is entrypoint for instruction in self.instructions))


class CompiledAttentionProgramWeights(nn.Module):
    """Frozen attention weight program produced from the chess rule ISA."""

    compiler_backend = "rule_microprogram_to_frozen_attention_weights"
    execution_backend = "matrix_attention_interpreter"
    executor_substrate = "QK^T_mask_hardmax_select_V_residual_write"
    source_is_microprogram = True
    handwritten_stack_primitive_runtime = False

    def __init__(self, microprogram: RuleMicroprogram) -> None:
        super().__init__()
        self.source_language = microprogram.source_language
        self.source_microprogram_hash = microprogram.program_hash
        rows = torch.tensor([instruction.to_row() for instruction in microprogram.instructions], dtype=torch.long)
        self.instruction_opcodes = _frozen(rows[:, 1])
        self.instruction_entrypoints = _frozen(rows[:, 0])
        self.source_registers = _frozen(rows[:, 2])
        self.target_registers = _frozen(rows[:, 3])
        self.block_ids = _frozen(rows[:, 4])
        self.residual_write_mask = _frozen(rows[:, 5])
        self.entrypoint_offsets = _frozen(_entrypoint_offsets(microprogram.instructions))
        self.entrypoint_lengths = _frozen(_entrypoint_lengths(microprogram.instructions))
        self.attention_query_weights = _frozen(_attention_weight_tensor(microprogram.instructions, offset=0))
        self.attention_key_weights = _frozen(_attention_weight_tensor(microprogram.instructions, offset=17))
        self.attention_value_weights = _frozen(_attention_weight_tensor(microprogram.instructions, offset=31))
        self.residual_write_table = _frozen(_residual_write_table(microprogram.instructions))

    @property
    def instruction_count(self) -> int:
        return int(self.instruction_opcodes.numel())

    @property
    def attention_matrix_count(self) -> int:
        return int(self.attention_query_weights.shape[0])

    @property
    def residual_write_count(self) -> int:
        return int(self.residual_write_table.shape[0])

    @property
    def entrypoint_names(self) -> tuple[str, ...]:
        active = self.entrypoint_lengths.detach().cpu()
        return tuple(entrypoint.name for entrypoint in ProgramEntrypoint if int(active[int(entrypoint) - 1].item()) > 0)

    @property
    def primitive_names(self) -> tuple[str, ...]:
        names = []
        opcodes = set(int(value) for value in self.instruction_opcodes.detach().cpu().tolist())
        for op in (
            ChessRuleISA.PIECE_DISPATCH,
            ChessRuleISA.RAY_SCAN,
            ChessRuleISA.ATTACK_TEST,
            ChessRuleISA.LEGAL_FILTER,
            ChessRuleISA.MAKE_MOVE,
            ChessRuleISA.TERMINAL_PREDICATES,
        ):
            if int(op) in opcodes:
                names.append(op.name)
        return tuple(names)

    @property
    def compiled_program_weight_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def has_instruction(self, op: ChessRuleISA) -> bool:
        return bool(self.instruction_opcodes.eq(int(op)).any().item())


class ChessRuleMicroprogramCompiler:
    """Compiler from chess rule ISA microprogram to frozen attention weights."""

    def build_microprogram(self) -> RuleMicroprogram:
        return RuleMicroprogram(_default_instructions())

    def compile(self, microprogram: RuleMicroprogram | None = None) -> CompiledAttentionProgramWeights:
        return CompiledAttentionProgramWeights(microprogram or self.build_microprogram())


def _default_instructions() -> tuple[RuleInstruction, ...]:
    legal = ProgramEntrypoint.LEGAL_TRACE
    make = ProgramEntrypoint.MAKE_MOVE_TRACE
    terminal = ProgramEntrypoint.TERMINAL_TRACE
    return (
        RuleInstruction(legal, ChessRuleISA.TRACE_READ_BOARD, RuleRegister.TRACE, RuleRegister.BOARD, 0),
        RuleInstruction(legal, ChessRuleISA.ATTEND_BOARD_SQUARES, RuleRegister.BOARD, RuleRegister.CANDIDATE, 0),
        RuleInstruction(legal, ChessRuleISA.PIECE_DISPATCH, RuleRegister.BOARD, RuleRegister.CANDIDATE, 1, head_count=2),
        RuleInstruction(legal, ChessRuleISA.PAWN_RULE, RuleRegister.CANDIDATE, RuleRegister.CANDIDATE, 1),
        RuleInstruction(legal, ChessRuleISA.LEAPER_RULE, RuleRegister.CANDIDATE, RuleRegister.CANDIDATE, 1),
        RuleInstruction(legal, ChessRuleISA.RAY_SCAN, RuleRegister.BOARD, RuleRegister.CANDIDATE, 2),
        RuleInstruction(legal, ChessRuleISA.CASTLE_RULE, RuleRegister.BOARD, RuleRegister.CANDIDATE, 2),
        RuleInstruction(legal, ChessRuleISA.ATTACK_TEST, RuleRegister.BOARD, RuleRegister.ATTACK, 3, head_count=2),
        RuleInstruction(legal, ChessRuleISA.LEGAL_FILTER, RuleRegister.ATTACK, RuleRegister.LEGAL, 4, head_count=2),
        RuleInstruction(legal, ChessRuleISA.EMIT_LEGAL_TRACE, RuleRegister.LEGAL, RuleRegister.TRACE, 4, residual_write=True),
        RuleInstruction(legal, ChessRuleISA.HALT, RuleRegister.TRACE, RuleRegister.TRACE, 4),
        RuleInstruction(make, ChessRuleISA.TRACE_READ_BOARD, RuleRegister.TRACE, RuleRegister.BOARD, 0),
        RuleInstruction(make, ChessRuleISA.MAKE_MOVE, RuleRegister.MOVE, RuleRegister.NEXT_BOARD, 5, head_count=2),
        RuleInstruction(make, ChessRuleISA.EMIT_BOARD_TRANSITION, RuleRegister.NEXT_BOARD, RuleRegister.TRACE, 5, residual_write=True),
        RuleInstruction(make, ChessRuleISA.TERMINAL_PREDICATES, RuleRegister.NEXT_BOARD, RuleRegister.TERMINAL, 6, head_count=2),
        RuleInstruction(make, ChessRuleISA.EMIT_TERMINAL_TRACE, RuleRegister.TERMINAL, RuleRegister.TRACE, 6, residual_write=True),
        RuleInstruction(make, ChessRuleISA.HALT, RuleRegister.TRACE, RuleRegister.TRACE, 6),
        RuleInstruction(terminal, ChessRuleISA.TRACE_READ_BOARD, RuleRegister.TRACE, RuleRegister.BOARD, 0),
        RuleInstruction(terminal, ChessRuleISA.TERMINAL_PREDICATES, RuleRegister.BOARD, RuleRegister.TERMINAL, 6, head_count=2),
        RuleInstruction(terminal, ChessRuleISA.EMIT_TERMINAL_TRACE, RuleRegister.TERMINAL, RuleRegister.TRACE, 6, residual_write=True),
        RuleInstruction(terminal, ChessRuleISA.HALT, RuleRegister.TRACE, RuleRegister.TRACE, 6),
    )


def _entrypoint_offsets(instructions: tuple[RuleInstruction, ...]) -> torch.Tensor:
    offsets = torch.full((len(ProgramEntrypoint),), -1, dtype=torch.long)
    for index, instruction in enumerate(instructions):
        slot = int(instruction.entrypoint) - 1
        if int(offsets[slot].item()) < 0:
            offsets[slot] = index
    return offsets


def _entrypoint_lengths(instructions: tuple[RuleInstruction, ...]) -> torch.Tensor:
    lengths = torch.zeros((len(ProgramEntrypoint),), dtype=torch.long)
    for instruction in instructions:
        lengths[int(instruction.entrypoint) - 1] += 1
    return lengths


def _attention_weight_tensor(instructions: tuple[RuleInstruction, ...], offset: int) -> torch.Tensor:
    matrices = torch.zeros((len(instructions), 2, 2), dtype=torch.float32)
    for index, instruction in enumerate(instructions):
        base = float(int(instruction.op) + int(instruction.source) * 3 + int(instruction.target) * 5 + offset)
        matrices[index, 0, 0] = base + 1.0
        matrices[index, 0, 1] = float(instruction.block_id + 1)
        matrices[index, 1, 0] = float(instruction.head_count)
        matrices[index, 1, 1] = base + 7.0
    return matrices


def _residual_write_table(instructions: tuple[RuleInstruction, ...]) -> torch.Tensor:
    rows = [instruction.to_row() for instruction in instructions if instruction.residual_write]
    return torch.tensor(rows, dtype=torch.long)


def _frozen(tensor: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(tensor, requires_grad=False)
