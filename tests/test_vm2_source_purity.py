from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "native/vm2/tools/audit_runtime.py"


def run_audit(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AUDITOR), "--root", str(root)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_vm2_runtime_is_attention_only(tmp_path: Path) -> None:
    clean = run_audit(ROOT)
    assert clean.returncode == 0, clean.stderr
    report = json.loads(clean.stdout)
    assert report == {
        "allowed_tensor_graph": True,
        "attention_matmul": True,
        "chess_runtime": False,
        "compiler_linked": False,
        "forbidden_routing": False,
        "legacy_link": False,
        "opcode_runtime": False,
        "semantic_branching": False,
        "semantic_scalar_read": False,
        "runtime_sources": 4,
    }


def test_knight_runtime_is_only_fixed_attention_and_matrix_write() -> None:
    source = (ROOT / "native/vm2/src/knight_runtime.cpp").read_text(encoding="utf-8")
    assert "self_attention(" in source
    assert source.count("torch::matmul(") >= 3
    for forbidden in (
        "source % 8", "target % 8", "std::abs", "index_put_", ".item<",
        "if (", "switch (", ".sum(", "legal",
    ):
        assert forbidden not in source.lower()


@pytest.mark.parametrize(
    ("injected", "message"),
    [
        ("void forbidden() { switch (Opcode::Add) {} }", "opcode-specific runtime semantics"),
        (
            "void forbidden(torch::Tensor state) { "
            "if (state[kControlToken][kHaltOffset].item<double>() == 1.0) {} }",
            "semantic host branching",
        ),
        (
            "void forbidden(torch::Tensor state) { auto x = state[0][kHaltOffset].item<double>(); }",
            "semantic scalar extraction",
        ),
        (
            "void forbidden(torch::Tensor state) { state.index_put_({0, 0}, 1.0); }",
            "forbidden runtime tensor routing",
        ),
        (
            "void forbidden(torch::Tensor state) { auto x = torch::where(state > 0, state, state); }",
            "forbidden runtime tensor routing",
        ),
        (
            "void forbidden(torch::Tensor state) { auto x = torch::nonzero(state); }",
            "forbidden runtime tensor routing",
        ),
        (
            "void forbidden(torch::Tensor state) { auto x = state.index_select(0, torch::tensor({0})); }",
            "forbidden runtime tensor routing",
        ),
        (
            "void forbidden(torch::Tensor state) { auto x = state * state; }",
            "forbidden runtime tensor routing",
        ),
        ("void forbidden(bool predicate) { if (predicate) {} }", "semantic host branching"),
        ("void forbidden(int program_counter) { switch (program_counter) {} }", "semantic host branching"),
        (
            "void forbidden(torch::Tensor state) { auto x = torch::relu(state); }",
            "unapproved runtime tensor operation",
        ),
        (
            "void forbidden(torch::Tensor state) { auto x = state.relu(); }",
            "unapproved runtime tensor operation",
        ),
        (
            "void forbidden(torch::Tensor board) { auto square = board; }",
            "chess-specific runtime semantics",
        ),
    ],
)
def test_audit_rejects_semantic_runtime_mutations(
    tmp_path: Path, injected: str, message: str
) -> None:

    broken_root = tmp_path / "repo"
    shutil.copytree(ROOT / "native/vm2", broken_root / "native/vm2")
    machine = broken_root / "native/vm2/src/machine.cpp"
    machine.write_text(
        machine.read_text(encoding="utf-8") + f"\n{injected}\n",
        encoding="utf-8",
    )
    broken = run_audit(broken_root)
    assert broken.returncode == 1
    assert message in broken.stderr


def test_audit_rejects_runtime_link_to_offline_compiler(tmp_path: Path) -> None:
    broken_root = tmp_path / "repo"
    shutil.copytree(ROOT / "native/vm2", broken_root / "native/vm2")
    cmake = broken_root / "native/vm2/CMakeLists.txt"
    cmake.write_text(
        cmake.read_text(encoding="utf-8").replace(
            "target_link_libraries(cmz_vm2 PUBLIC cmz_vm2_attention)",
            "target_link_libraries(cmz_vm2 PUBLIC cmz_vm2_attention cmz_vm2_compiler)",
        ),
        encoding="utf-8",
    )
    broken = run_audit(broken_root)
    assert broken.returncode == 1
    assert "offline compiler linked into runtime" in broken.stderr


def test_audit_rejects_runtime_link_to_chess_compiler(tmp_path: Path) -> None:
    broken_root = tmp_path / "repo"
    shutil.copytree(ROOT / "native/vm2", broken_root / "native/vm2")
    cmake = broken_root / "native/vm2/CMakeLists.txt"
    cmake.write_text(
        cmake.read_text(encoding="utf-8").replace(
            "target_link_libraries(cmz_vm2 PUBLIC cmz_vm2_attention)",
            "target_link_libraries(cmz_vm2 PUBLIC cmz_vm2_attention cmz_vm2_chess1_compiler)",
        ),
        encoding="utf-8",
    )
    broken = run_audit(broken_root)
    assert broken.returncode == 1
    assert "offline compiler linked into runtime" in broken.stderr
