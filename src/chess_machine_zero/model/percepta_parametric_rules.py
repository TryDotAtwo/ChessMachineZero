"""Percepta-style parametric chess rules stored in frozen model weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chess_machine_zero.model.hardmax_attention import DenseHardmax2D
from chess_machine_zero.model.weight_compiled_rules import WeightCompiledRulesTransformer


class PerceptaParametricRulesTransformer(WeightCompiledRulesTransformer):
    """Arbitrary-position chess rule circuit backed by reusable frozen tensors."""

    rule_execution_mode = "percepta_parametric_rule_weights"
    attention_backend = "dense_hardmax_2d"
    parametric_rule_weights = True
    position_lookup = False
    finite_prompt_lookup = False

    def __init__(self) -> None:
        super().__init__()
        self.square_attention_keys = _frozen_weight(_square_attention_keys())
        self._hardmax = DenseHardmax2D()

    @property
    def compiled_prompt_count(self) -> int:
        return 0

    def attention_select_square(self, square: int) -> int:
        """Address a board square through a deterministic 2D hardmax head."""

        if not 0 <= square < 64:
            raise ValueError(f"square out of range: {square}")
        query = self.square_attention_keys[square].detach()
        output = self._hardmax(query, self.square_attention_keys.detach())
        return int(output.indices[0].item())

    def compiled_rule_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(frozen=True, slots=True)
class PerceptaParametricRuleCompiler:
    """Build a reusable parametric rule circuit without board-position examples."""

    def compile_rule_circuit(self) -> PerceptaParametricRulesTransformer:
        return PerceptaParametricRulesTransformer()


def _square_attention_keys() -> torch.Tensor:
    keys = torch.empty((64, 2), dtype=torch.float32)
    for square in range(64):
        file_index = float(square % 8)
        rank_index = float(square // 8)
        keys[square, 0] = file_index * 17.0 + 1.0
        keys[square, 1] = rank_index * 19.0 + 3.0
    return keys


def _frozen_weight(tensor: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(tensor, requires_grad=False)
