from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_source_has_no_python_attention_runtime_modules() -> None:
    forbidden_paths = (
        "src/chess_machine_zero/model/percepta_attention_rule_kernels.py",
        "src/chess_machine_zero/model/percepta_attention_block_stack.py",
        "src/chess_machine_zero/model/percepta_matrix_attention_runtime.py",
        "src/chess_machine_zero/model/percepta_tensor_trace_runtime.py",
        "src/chess_machine_zero/model/percepta_rule_layer_graph.py",
        "src/chess_machine_zero/model/percepta_frozen_attention_vm.py",
        "src/chess_machine_zero/model/percepta_parametric_selfplay.py",
        "src/chess_machine_zero/dashboard/server.py",
        "src/chess_machine_zero/dashboard/state.py",
    )

    for path in forbidden_paths:
        assert not (ROOT / path).exists(), path


def test_public_python_import_surface_has_no_attention_runtime_exports() -> None:
    model_init = (ROOT / "src/chess_machine_zero/model/__init__.py").read_text(encoding="utf-8")
    dashboard_init = (ROOT / "src/chess_machine_zero/dashboard/__init__.py").read_text(encoding="utf-8")
    combined = "\n".join((model_init, dashboard_init))

    forbidden_tokens = (
        "PerceptaFrozenAttentionRuleComputer",
        "PerceptaFrozenAttentionRuleCompiler",
        "FrozenAttentionTensorRuleKernels",
        "FrozenMatrixAttentionInterpreter",
        "FrozenAttentionTensorTraceRuntime",
        "FrozenTransformerAttentionBlockStack",
        "PerceptaParametricSelfPlaySession",
        "CMZDashboardSession",
        "DashboardApp",
    )
    for token in forbidden_tokens:
        assert token not in combined
