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
    compiler_linked = bool(
        re.search(
            r"target_link_libraries\s*\(\s*cmz_vm2\b[^)]*cmz_vm2(?:_[a-z0-9]+)*_compiler",
            cmake,
            re.IGNORECASE | re.DOTALL,
        )
    )
    chess_runtime = bool(
        re.search(
            r"\b(?:chess|board|square|occupancy|legal_?move|white_?pawn)\b",
            runtime,
            re.IGNORECASE,
        )
    )
    semantic_scalar_read = bool(re.search(r"\.item\s*<", runtime))
    semantic_branching = bool(
        re.search(
            r"\b(?:if|switch)\s*\([^)]*(?:state|halt|predicate|program_?counter|kprogramcounter|\.item)",
            runtime,
            re.IGNORECASE | re.DOTALL,
        )
    )
    forbidden_routing = bool(
        re.search(
            r"index_put_|\bindex_select\s*\(|\bgather\s*\(|\bscatter\w*\s*\(|"
            r"(?:torch|at)::nonzero\s*\(|torch::where\s*\(|"
            r"\bstate\s*\*|\bprojected\s*\*",
            runtime,
            re.IGNORECASE,
        )
    )
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
    tensor_calls = set(re.findall(r"\b(?:torch|at)::([A-Za-z_]\w*)\s*\(", runtime))
    member_calls = set(re.findall(r"\.\s*([A-Za-z_]\w*)\s*\(", runtime))
    allowed_tensor_graph = tensor_calls <= {"matmul", "one_hot", "stack"} and member_calls <= {
        "clone",
        "max",
        "push_back",
        "scalar_type",
        "size",
        "squeeze",
        "t",
        "to",
        "unsqueeze",
    }

    if opcode_runtime:
        print("opcode-specific runtime semantics detected", file=sys.stderr)
        return 1
    if chess_runtime:
        print("chess-specific runtime semantics detected", file=sys.stderr)
        return 1
    if semantic_branching:
        print("semantic host branching detected", file=sys.stderr)
        return 1
    if semantic_scalar_read:
        print("semantic scalar extraction detected", file=sys.stderr)
        return 1
    if forbidden_routing:
        print("forbidden runtime tensor routing detected", file=sys.stderr)
        return 1
    if not allowed_tensor_graph:
        print("unapproved runtime tensor operation detected", file=sys.stderr)
        return 1
    if compiler_linked:
        print("offline compiler linked into runtime", file=sys.stderr)
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
                "allowed_tensor_graph": allowed_tensor_graph,
                "attention_matmul": attention_matmul,
                "compiler_linked": compiler_linked,
                "chess_runtime": chess_runtime,
                "forbidden_routing": forbidden_routing,
                "legacy_link": legacy_link,
                "opcode_runtime": opcode_runtime,
                "semantic_branching": semantic_branching,
                "semantic_scalar_read": semantic_scalar_read,
                "runtime_sources": len(RUNTIME_FILES),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
