from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "native/vm2/tools/audit_runtime.py"


def run_audit(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AUDITOR), "--root", str(root)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_vm2_runtime_is_attention_only_and_audit_rejects_opcode_execution(tmp_path: Path) -> None:
    clean = run_audit(ROOT)
    assert clean.returncode == 0, clean.stderr
    report = json.loads(clean.stdout)
    assert report == {
        "attention_matmul": True,
        "legacy_link": False,
        "opcode_runtime": False,
        "runtime_sources": 4,
    }

    broken_root = tmp_path / "repo"
    shutil.copytree(ROOT / "native/vm2", broken_root / "native/vm2")
    machine = broken_root / "native/vm2/src/machine.cpp"
    machine.write_text(
        machine.read_text(encoding="utf-8") + "\nvoid forbidden() { switch (Opcode::Add) {} }\n",
        encoding="utf-8",
    )
    broken = run_audit(broken_root)
    assert broken.returncode == 1
    assert "opcode-specific runtime semantics" in broken.stderr
