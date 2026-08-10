# Predicate control flow — 2026-08-10

## Implemented contract

- `CMP_EQ p, left, right` produces a predicate token through equality relation
  attention.
- `JUMP_IF p, target` selects target or fallthrough through branch-candidate
  tokens and hardmax attention.
- Literal predicate source `TRUE` implements unconditional back-edges without a
  separate native jump path.
- Runtime remains the same fixed-unroll attention/projection graph and never
  decodes predicates or PC on the host.

## Exact acceptance

- all `16 x 16` equality pairs;
- both predicate values across all sixteen branch targets;
- counter loop reaches `r0=3`, `p0=TRUE`, and absorbing `HALT=TRUE`;
- legacy `LOAD_CONST`, `MOVE`, `ADD`, and `HALT` behavior remains exact.

## Verification

```text
python -m pytest tests/test_vm2_source_purity.py -q
```

Result: `12 passed`.

The audit reported:

```json
{"allowed_tensor_graph":true,"attention_matmul":true,"compiler_linked":false,"forbidden_routing":false,"legacy_link":false,"opcode_runtime":false,"runtime_sources":4,"semantic_branching":false,"semantic_scalar_read":false}
```

Clean Docker/LibTorch build and CTest:

```text
cmake -S /work -B /tmp/vm-build -G Ninja -DCMAKE_PREFIX_PATH=/opt/pytorch/share/cmake
cmake --build /tmp/vm-build
ctest --test-dir /tmp/vm-build --output-on-failure
```

Result: build succeeded; `3/3` tests passed. The exhaustive machine test took
`137.71 sec` on CPU LibTorch. The container had no NVIDIA driver; this is
semantic evidence, not a GPU-performance result.
