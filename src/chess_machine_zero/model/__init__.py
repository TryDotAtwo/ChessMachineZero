"""Model-side executor components."""

from chess_machine_zero.model.hardmax_attention import DenseHardmax2D
from chess_machine_zero.model.hosted_vm import TransformerHostedVM
from chess_machine_zero.model.machine_transformer import CMZMachineTransformer
from chess_machine_zero.model.analytic_rules import AnalyticRuleCompiler, AnalyticRulesTransformer
from chess_machine_zero.model.trace_executor import CMZTraceExecutor
from chess_machine_zero.model.percepta_e2e_decoder import PerceptaE2ETraceDecoder
from chess_machine_zero.model.percepta_selfplay import PerceptaSelfPlaySession
from chess_machine_zero.model.percepta_parametric_rules import PerceptaParametricRuleCompiler, PerceptaParametricRulesTransformer
from chess_machine_zero.model.percepta_rule_compiler import (
    ChessRuleISA,
    ChessRuleMicroprogramCompiler,
    CompiledAttentionProgramWeights,
    ProgramEntrypoint,
    RuleInstruction,
    RuleMicroprogram,
    RuleRegister,
)
from chess_machine_zero.model.weight_compiled_rules import WeightCompiledRuleCompiler, WeightCompiledRulesTransformer

__all__ = [
    "CMZMachineTransformer",
    "CMZTraceExecutor",
    "DenseHardmax2D",
    "AnalyticRuleCompiler",
    "AnalyticRulesTransformer",
    "TransformerHostedVM",
    "WeightCompiledRuleCompiler",
    "WeightCompiledRulesTransformer",
    "PerceptaE2ETraceDecoder",
    "PerceptaSelfPlaySession",
    "PerceptaParametricRuleCompiler",
    "PerceptaParametricRulesTransformer",
    "ChessRuleISA",
    "ChessRuleMicroprogramCompiler",
    "CompiledAttentionProgramWeights",
    "ProgramEntrypoint",
    "RuleInstruction",
    "RuleMicroprogram",
    "RuleRegister",
]
