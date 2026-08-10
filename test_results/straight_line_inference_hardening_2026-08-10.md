# Straight-line inference hardening — 2026-08-10

## Contract

- fixed inference unroll; no early exit on `HALT`;
- post-HALT state is absorbing and byte-identical;
- no runtime tensor-to-host scalar extraction;
- no opcode, PC, register, predicate, or HALT host dispatch;
- runtime writes use frozen batched matrix projections;
- offline compiler is not linked into the runtime library.

## TDD evidence

Before the fix, the new purity gate failed on the existing runtime with:

```text
semantic host branching detected
```

The fixed-unroll C++ test also failed to compile before implementation because
`run_fixed` did not exist.

## Final verification

Command:

```text
python -m pytest tests/test_vm2_source_purity.py -q
```

Result: `12 passed`, including both namespace-style and tensor-member operation
mutations.

Audit report:

```json
{"allowed_tensor_graph":true,"attention_matmul":true,"compiler_linked":false,"forbidden_routing":false,"legacy_link":false,"opcode_runtime":false,"runtime_sources":4,"semantic_branching":false,"semantic_scalar_read":false}
```

Clean Docker/LibTorch command:

```text
cmake -S /work -B /tmp/vm-build -G Ninja -DCMAKE_PREFIX_PATH=/opt/pytorch/share/cmake
cmake --build /tmp/vm-build
ctest --test-dir /tmp/vm-build --output-on-failure
```

Result: build completed; `3/3` CTest tests passed. The container had no NVIDIA
driver and tests ran on CPU LibTorch; this is semantic validation, not a GPU
performance claim.
