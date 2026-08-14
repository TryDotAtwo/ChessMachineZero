from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "native/vm3/tools/audit_runtime.py"


def run_audit(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AUDITOR), "--root", str(root)],
        check=False,
        capture_output=True,
        text=True,
    )


def clone_vm3(tmp_path: Path) -> Path:
    if not (ROOT / "native/vm3/CMakeLists.txt").is_file():
        pytest.skip("VM3 target graph is not implemented yet")
    broken_root = tmp_path / "repo"
    (broken_root / "native").mkdir(parents=True)
    shutil.copy2(ROOT / "CMakeLists.txt", broken_root / "CMakeLists.txt")
    shutil.copytree(ROOT / "native/vm3", broken_root / "native/vm3")
    return broken_root


def add_executor_source(root: Path, source: str) -> None:
    path = root / "native/vm3/src/mutated_executor.cpp"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    cmake = root / "native/vm3/CMakeLists.txt"
    text = cmake.read_text(encoding="utf-8")
    marker = "set(CMZ_VM3_EXECUTOR_SOURCES\n"
    assert marker in text
    cmake.write_text(
        text.replace(marker, marker + "  src/mutated_executor.cpp\n", 1),
        encoding="utf-8",
    )


def test_vm3_auditor_accepts_the_declared_target_closure() -> None:
    result = run_audit(ROOT)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["schema"] == "cmz-vm3-purity-v1"
    assert report["targets"] == {
        "compiler": "cmz_vm3_compiler",
        "executor": "cmz_vm3_executor",
        "loader": "cmz_vm3_program_loader",
        "policy": "cmz_vm3_policy",
    }
    assert report["contract"] == {
        "candidate_routes": 4273,
        "control_routes": 4,
        "lookup_head_dim": 2,
        "move_candidates": 4272,
        "piece_states": 13,
        "policy_controls": 3,
    }
    assert report["compiler_linked"] is False
    assert report["forbidden_findings"] == []
    assert report["unclassified_runtime_sources"] == []


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("void f(Tensor state) { state.item<double>(); }", "semantic scalar read"),
        ("void f(Tensor x) { x.to(torch::kCPU); }", "CPU tensor transfer"),
        ("void f(Tensor x) { x.detach(); }", "autograd detach"),
        ("void f(Tensor x) { torch::where(x > 0, x, x); }", "dynamic tensor routing"),
        ("void f(Tensor x) { torch::nonzero(x); }", "dynamic tensor routing"),
        ("void f(Tensor x) { x.index_put_({0}, 1); }", "in-place semantic write"),
        ("void f(Tensor x) { x.gather(0, x); }", "dynamic tensor routing"),
        ("void f(Tensor x) { x.gt(0); }", "state-dependent mask"),
        ("void f() { torch::NoGradGuard guard; }", "host or autograd escape"),
        ("void f(Tensor x) { torch::relu(x); }", "unregistered tensor operation"),
        ("void f(Tensor x) { torch::sort(x); }", "operation is not allowlisted"),
        (
            "class Sneaky : public torch::autograd::Function<Sneaky> {};",
            "unregistered custom autograd operation",
        ),
        ("void f(Tensor board) { if (board.defined()) {} }", "chess-aware runtime"),
        ("void f(int opcode) { switch (opcode) {} }", "semantic runtime branch"),
    ],
)
def test_vm3_auditor_rejects_runtime_mutations(
    tmp_path: Path, source: str, message: str
) -> None:
    broken_root = clone_vm3(tmp_path)
    add_executor_source(broken_root, source)
    result = run_audit(broken_root)
    assert result.returncode == 1
    assert message in result.stderr


def test_vm3_auditor_rejects_compiler_in_executor_link_closure(
    tmp_path: Path,
) -> None:
    broken_root = clone_vm3(tmp_path)
    cmake = broken_root / "native/vm3/CMakeLists.txt"
    text = cmake.read_text(encoding="utf-8")
    cmake.write_text(
        text.replace(
            "target_link_libraries(cmz_vm3_executor PUBLIC cmz_vm3_policy cmz_vm3_program_loader ${TORCH_LIBRARIES})",
            "target_link_libraries(cmz_vm3_executor PUBLIC cmz_vm3_policy cmz_vm3_program_loader cmz_vm3_compiler ${TORCH_LIBRARIES})",
            1,
        ),
        encoding="utf-8",
    )
    result = run_audit(broken_root)
    assert result.returncode == 1
    assert "compiler target linked into executor closure" in result.stderr


def test_vm3_auditor_rejects_unclassified_runtime_source(tmp_path: Path) -> None:
    broken_root = clone_vm3(tmp_path)
    (broken_root / "native/vm3/src").mkdir(parents=True, exist_ok=True)
    (broken_root / "native/vm3/src/orphan.cpp").write_text(
        "void orphan() {}\n", encoding="utf-8"
    )
    result = run_audit(broken_root)
    assert result.returncode == 1
    assert "unclassified runtime source" in result.stderr
