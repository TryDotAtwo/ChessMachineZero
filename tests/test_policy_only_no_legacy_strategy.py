from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_policy_only_runtime_has_no_legacy_ranker_baseline_search_modules() -> None:
    forbidden_paths = (
        "src/chess_machine_zero/model/ranker.py",
        "src/chess_machine_zero/model/baseline.py",
        "src/chess_machine_zero/model/analytic_machine.py",
        "src/chess_machine_zero/model/weight_compiled_machine.py",
        "src/chess_machine_zero/selfplay/actor.py",
        "src/chess_machine_zero/train/losses.py",
        "src/chess_machine_zero/vm/lookahead.py",
        "src/chess_machine_zero/vm/decision_program.py",
    )

    for path in forbidden_paths:
        assert not (ROOT / path).exists(), path


def test_python_public_imports_do_not_expose_legacy_strategy_api() -> None:
    model_init = (ROOT / "src/chess_machine_zero/model/__init__.py").read_text(encoding="utf-8")
    selfplay_init = (ROOT / "src/chess_machine_zero/selfplay/__init__.py").read_text(encoding="utf-8")
    bytecode = (ROOT / "src/chess_machine_zero/vm/bytecode.py").read_text(encoding="utf-8")
    combined = "\n".join((model_init, selfplay_init, bytecode))

    forbidden_tokens = (
        "CMZMoveRanker",
        "CMZOutcomeBaseline",
        "CMZAnalyticMachine",
        "CMZWeightCompiledMachine",
        "SelfPlayActor",
        "SelfPlayConfig",
        "CALL_MOVE_RANKER",
        "SAMPLE_BY_SCORE",
        "COMMIT_SELECTED_MOVE",
        "BEGIN_CANDIDATE_SCORING",
    )
    for token in forbidden_tokens:
        assert token not in combined
