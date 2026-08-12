from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def constant(source: str, name: str) -> int:
    match = re.search(rf"{name}\s*=\s*(\d+);", source)
    assert match, f"missing integer constant {name}"
    return int(match.group(1))


def test_site_claims_match_compiled_schema() -> None:
    chess_header = (ROOT / "native/vm2/include/cmz_vm2/chess1_compiler.h").read_text(encoding="utf-8")
    vm_header = (ROOT / "native/vm2/include/cmz_vm2/schema.h").read_text(encoding="utf-8")
    app = (ROOT / "site/src/App.tsx").read_text(encoding="utf-8")
    trace = (ROOT / "site/src/trace.ts").read_text(encoding="utf-8")

    candidates = constant(chess_header, "kCandidateTokenCount")
    stages = constant(vm_header, "kStageCount")
    assert f"{candidates} кандидатов" in app
    assert f"{stages} стадий" in app
    assert f"{candidates} candidate-токенов" in trace
    assert "Полные шахматы" in app and "pending" in app


def test_pages_workflow_gates_runtime_and_public_claims() -> None:
    workflow = (ROOT / ".github/workflows/pages.yml").read_text(encoding="utf-8")
    for watched_path in (
        "site/**",
        "native/vm2/**",
        "tests/test_vm2_source_purity.py",
        "tests/test_public_evidence_gate.py",
        "README.md",
    ):
        assert watched_path in workflow
    assert "pytest tests/test_vm2_source_purity.py tests/test_public_evidence_gate.py" in workflow
    assert "needs: [evidence, build]" in workflow
