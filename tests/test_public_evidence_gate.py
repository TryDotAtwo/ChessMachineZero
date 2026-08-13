from __future__ import annotations

import re
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def constant(source: str, name: str) -> int:
    match = re.search(rf"{name}\s*=\s*(\d+);", source)
    assert match, f"missing integer constant {name}"
    return int(match.group(1))


def test_site_claims_match_compiled_schema() -> None:
    chess_header = (ROOT / "native/vm2/include/cmz_vm2/chess1_compiler.h").read_text(encoding="utf-8")
    vm_header = (ROOT / "native/vm2/include/cmz_vm2/schema.h").read_text(encoding="utf-8")
    app = (ROOT / "site/src/App.tsx").read_text(encoding="utf-8")
    laboratory = (ROOT / "site/src/TraceDebugger.tsx").read_text(encoding="utf-8")
    artifact = json.loads((ROOT / "site/public/traces/pawn-e2-e3.json").read_text(encoding="utf-8"))

    candidates = constant(chess_header, "kCandidateTokenCount")
    stages = constant(vm_header, "kStageCount")
    assert len(artifact["stages"]) == stages
    assert artifact["stages"][0]["matrices"]["S"]["rows"] >= candidates
    public_ui = app + laboratory
    assert "полные шахматы ещё не реализованы" in public_ui
    assert "NativeTraceLaboratory" in public_ui and "Арифметика ячейки" in public_ui
    assert "без логики правил во frontend" in public_ui
    assert "ответ" not in public_ui.lower()


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
