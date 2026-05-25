"""Model-side executor components."""

from chess_machine_zero.model.hardmax_attention import DenseHardmax2D
from chess_machine_zero.model.hosted_vm import TransformerHostedVM
from chess_machine_zero.model.machine_transformer import CMZMachineTransformer
from chess_machine_zero.model.analytic_machine import CMZAnalyticMachine
from chess_machine_zero.model.analytic_rules import AnalyticRuleCompiler, AnalyticRulesTransformer
from chess_machine_zero.model.baseline import CMZOutcomeBaseline
from chess_machine_zero.model.ranker import CMZMoveRanker
from chess_machine_zero.model.trace_executor import CMZTraceExecutor
from chess_machine_zero.model.percepta_e2e_decoder import PerceptaE2ETraceDecoder
from chess_machine_zero.model.percepta_selfplay import PerceptaSelfPlaySession
from chess_machine_zero.model.percepta_parametric_rules import PerceptaParametricRuleCompiler, PerceptaParametricRulesTransformer
from chess_machine_zero.model.percepta_parametric_selfplay import PerceptaParametricSelfPlaySession
from chess_machine_zero.model.percepta_attention_rule_kernels import FrozenAttentionTensorRuleKernels
from chess_machine_zero.model.percepta_rule_compiler import (
    ChessRuleISA,
    ChessRuleMicroprogramCompiler,
    CompiledAttentionProgramWeights,
    ProgramEntrypoint,
    RuleInstruction,
    RuleMicroprogram,
    RuleRegister,
)
from chess_machine_zero.model.percepta_matrix_attention_runtime import FrozenMatrixAttentionInterpreter
from chess_machine_zero.model.percepta_attention_block_stack import (
    FrozenAttentionHead,
    FrozenTransformerAttentionBlock,
    FrozenTransformerAttentionBlockStack,
    ResidualTraceWrite,
)
from chess_machine_zero.model.percepta_tensor_trace_runtime import FrozenAttentionTensorTraceRuntime
from chess_machine_zero.model.percepta_frozen_attention_vm import PerceptaFrozenAttentionRuleCompiler, PerceptaFrozenAttentionRuleComputer
from chess_machine_zero.model.weight_compiled_machine import CMZWeightCompiledMachine
from chess_machine_zero.model.weight_compiled_rules import WeightCompiledRuleCompiler, WeightCompiledRulesTransformer

__all__ = [
    "CMZMachineTransformer",
    "CMZMoveRanker",
    "CMZOutcomeBaseline",
    "CMZAnalyticMachine",
    "CMZTraceExecutor",
    "DenseHardmax2D",
    "AnalyticRuleCompiler",
    "AnalyticRulesTransformer",
    "TransformerHostedVM",
    "WeightCompiledRuleCompiler",
    "WeightCompiledRulesTransformer",
    "CMZWeightCompiledMachine",
    "PerceptaE2ETraceDecoder",
    "PerceptaSelfPlaySession",
    "PerceptaParametricRuleCompiler",
    "PerceptaParametricRulesTransformer",
    "PerceptaParametricSelfPlaySession",
    "FrozenAttentionTensorRuleKernels",
    "ChessRuleISA",
    "ChessRuleMicroprogramCompiler",
    "CompiledAttentionProgramWeights",
    "ProgramEntrypoint",
    "RuleInstruction",
    "RuleMicroprogram",
    "RuleRegister",
    "FrozenMatrixAttentionInterpreter",
    "FrozenAttentionHead",
    "FrozenTransformerAttentionBlock",
    "FrozenTransformerAttentionBlockStack",
    "ResidualTraceWrite",
    "FrozenAttentionTensorTraceRuntime",
    "PerceptaFrozenAttentionRuleCompiler",
    "PerceptaFrozenAttentionRuleComputer",
]
