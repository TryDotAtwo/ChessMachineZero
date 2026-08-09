from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


RUNTIME_FILES = (
    "native/vm2/include/cmz_vm2/attention.h",
    "native/vm2/include/cmz_vm2/machine.h",
    "native/vm2/src/attention.cpp",
    "native/vm2/src/machine.cpp",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    sources = [(args.root / path).read_text(encoding="utf-8") for path in RUNTIME_FILES]
    runtime = "\n".join(sources)
    cmake = (args.root / "native/vm2/CMakeLists.txt").read_text(encoding="utf-8")

    opcode_runtime = bool(
        re.search(r"Opcode::|switch\s*\([^)]*opcode|if\s*\([^)]*opcode", runtime, re.IGNORECASE)
    )
    legacy_link = "cmz_engine" in cmake or "native/cpp" in cmake
    attention_matmul = all(
        expression in runtime
        for expression in (
            "torch::matmul(x, wq)",
            "torch::matmul(x, wk)",
            "torch::matmul(x, wv)",
            "torch::matmul(q, k.t())",
            "torch::matmul(one_hot, v)",
        )
    )

    if opcode_runtime:
        print("opcode-specific runtime semantics detected", file=sys.stderr)
        return 1
    if legacy_link:
        print("legacy engine linkage detected", file=sys.stderr)
        return 1
    if not attention_matmul:
        print("classical self-attention matrix path is incomplete", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "attention_matmul": attention_matmul,
                "legacy_link": legacy_link,
                "opcode_runtime": opcode_runtime,
                "runtime_sources": len(RUNTIME_FILES),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
