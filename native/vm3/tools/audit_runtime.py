from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


TARGETS = {
    "compiler": "cmz_vm3_compiler",
    "executor": "cmz_vm3_executor",
    "loader": "cmz_vm3_program_loader",
    "policy": "cmz_vm3_policy",
}

SOURCE_VARIABLES = {
    "CMZ_VM3_EXECUTOR_SOURCES": "executor",
    "CMZ_VM3_POLICY_SOURCES": "policy",
    "CMZ_VM3_LOADER_SOURCES": "loader",
    "CMZ_VM3_COMPILER_SOURCES": "compiler",
}

CONTRACT_NAMES = {
    "candidate_routes": "kCandidateSelectorRouteCount",
    "control_routes": "kControlRouteCount",
    "lookup_head_dim": "kLookupHeadDim",
    "move_candidates": "kMoveCandidateCount",
    "piece_states": "kPieceStateCount",
    "policy_controls": "kPolicyControlCount",
}

FORBIDDEN_PATTERNS = (
    (re.compile(r"\.item\s*(?:<|\()"), "semantic scalar read"),
    (re.compile(r"\.to\s*\(\s*torch::kCPU"), "CPU tensor transfer"),
    (re.compile(r"\.detach\s*\("), "autograd detach"),
    (re.compile(r"torch::(?:where|nonzero)\s*\("), "dynamic tensor routing"),
    (re.compile(r"\.index_(?:put|select)_?\s*\("), "in-place semantic write"),
    (re.compile(r"\.copy_\s*\("), "full-state or history copy"),
    (re.compile(r"\.(?:gather|nonzero|sort|topk)\s*\("), "dynamic tensor routing"),
    (re.compile(r"\.(?:gt|ge|lt|le|eq)\s*\("), "state-dependent mask"),
    (re.compile(r"torch::relu\s*\(|\.relu\s*\("), "unregistered tensor operation"),
    (re.compile(r"\b(?:NoGradGuard|cudaStreamSynchronize|synchronize|host_callback)\b"),
     "host or autograd escape"),
)

CHESS_WORD = re.compile(
    r"\b(?:board|pawn|knight|bishop|rook|queen|king|fen|uci)\b",
    re.IGNORECASE,
)
BRANCH = re.compile(r"\b(?:if|switch)\s*\(")
TORCH_OPERATION = re.compile(r"\b(torch|at)::([A-Za-z_]\w*)\s*\(")
AUTOGRAD_CLASS = re.compile(
    r"class\s+(\w+).*?torch::autograd::Function\s*<\s*\1\s*>", re.DOTALL
)
REGISTERED_AUTOGRAD = re.compile(r"CMZ_VM3_REGISTERED_AUTOGRAD:\s*(\w+)")
ALLOWED_TORCH_OPERATIONS = {
    "at::one_hot",
    "torch::Tensor",
    "torch::exp",
    "torch::index_select",
    "torch::matmul",
    "torch::ones_like",
    "torch::stack",
}


@dataclass(frozen=True)
class Finding:
    path: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def parse_source_manifest(cmake_text: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for variable, role in SOURCE_VARIABLES.items():
        match = re.search(rf"set\(\s*{variable}\s*(.*?)\)", cmake_text, re.DOTALL)
        if match is None:
            raise ValueError(f"missing source manifest {variable}")
        result[role] = [
            token.replace("\\", "/")
            for token in re.findall(r"(?:src|compiler)/[^\s\)]+\.cpp", match.group(1))
        ]
    return result


def eval_expression(node: ast.AST, values: dict[str, int]) -> int:
    if isinstance(node, ast.Expression):
        return eval_expression(node.body, values)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return eval_expression(node.left, values) + eval_expression(node.right, values)
    raise ValueError("unsupported tensor-contract expression")


def parse_contract(header: str) -> dict[str, int]:
    declarations = re.findall(
        r"inline\s+constexpr\s+std::int64_t\s+(\w+)\s*=\s*(.*?);",
        header,
        re.DOTALL,
    )
    values: dict[str, int] = {}
    for name, expression in declarations:
        values[name] = eval_expression(ast.parse(expression.strip(), mode="eval"), values)
    missing = sorted(set(CONTRACT_NAMES.values()) - values.keys())
    if missing:
        raise ValueError(f"missing tensor-contract constants: {', '.join(missing)}")
    return {key: values[name] for key, name in CONTRACT_NAMES.items()}


def target_is_declared(cmake_text: str, target: str) -> bool:
    return re.search(rf"add_library\(\s*{re.escape(target)}\b", cmake_text) is not None


def compiler_is_linked(cmake_text: str) -> bool:
    match = re.search(
        r"target_link_libraries\(\s*cmz_vm3_executor\s+(?:INTERFACE|PUBLIC|PRIVATE)\s+(.*?)\)",
        cmake_text,
        re.DOTALL,
    )
    if match is None:
        raise ValueError("missing executor link closure")
    return "cmz_vm3_compiler" in match.group(1).split()


def scan_runtime(root: Path, manifests: dict[str, list[str]]) -> tuple[list[Finding], list[str], list[str]]:
    findings: list[Finding] = []
    classified = {
        path
        for role in ("executor", "policy", "loader")
        for path in manifests[role]
    }
    src_root = root / "native/vm3/src"
    existing = {
        path.relative_to(root / "native/vm3").as_posix()
        for path in src_root.glob("*.cpp")
    } if src_root.is_dir() else set()
    unclassified = sorted(existing - classified)
    operations: set[str] = set()

    for relative in sorted(set(manifests["executor"] + manifests["policy"])):
        path = root / "native/vm3" / relative
        if not path.is_file():
            findings.append(Finding(relative, "classified runtime source is missing"))
            continue
        text = path.read_text(encoding="utf-8")
        for pattern, message in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                findings.append(Finding(relative, message))
        if BRANCH.search(text):
            message = "chess-aware runtime" if CHESS_WORD.search(text) else "semantic runtime branch"
            findings.append(Finding(relative, message))
        for namespace, operation in TORCH_OPERATION.findall(text):
            qualified = f"{namespace}::{operation}"
            operations.add(qualified)
            if qualified not in ALLOWED_TORCH_OPERATIONS:
                findings.append(Finding(relative, f"operation is not allowlisted: {qualified}"))
        registered = set(REGISTERED_AUTOGRAD.findall(text))
        for operation in AUTOGRAD_CLASS.findall(text):
            if operation not in registered:
                findings.append(Finding(relative, "unregistered custom autograd operation"))

    for relative in unclassified:
        findings.append(Finding(relative, "unclassified runtime source"))
    return findings, unclassified, sorted(operations)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    cmake_path = root / "native/vm3/CMakeLists.txt"
    header_path = root / "native/vm3/include/cmz_vm3/tensor_contract.h"
    try:
        cmake_text = cmake_path.read_text(encoding="utf-8")
        undeclared = [target for target in TARGETS.values() if not target_is_declared(cmake_text, target)]
        if undeclared:
            raise ValueError(f"missing targets: {', '.join(undeclared)}")
        manifests = parse_source_manifest(cmake_text)
        contract = parse_contract(header_path.read_text(encoding="utf-8"))
        linked = compiler_is_linked(cmake_text)
        findings, unclassified, operations = scan_runtime(root, manifests)
        if linked:
            findings.append(Finding("native/vm3/CMakeLists.txt", "compiler target linked into executor closure"))
    except (OSError, SyntaxError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    report = {
        "schema": "cmz-vm3-purity-v1",
        "targets": TARGETS,
        "contract": contract,
        "compiler_linked": linked,
        "runtime_sources": sorted(manifests["executor"] + manifests["policy"]),
        "operation_manifest": operations,
        "forbidden_findings": [finding.__dict__ for finding in findings],
        "unclassified_runtime_sources": unclassified,
    }
    if findings:
        for finding in findings:
            print(f"{finding.path}: {finding.message}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
