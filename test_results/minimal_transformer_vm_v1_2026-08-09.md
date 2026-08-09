# Minimal Transformer VM V1 — 2026-08-09

## Result

- status=implemented
- runtime=native/vm2 independent CPU LibTorch target
- semantic_path=6 fixed classical self-attention stages
- stages=decode,left-read,right-read,alu-lookup,register-write,pc-halt-write
- attention=Q=XWq,K=XWk,V=XWv,scores=QK^T+mask,lowest-index-hardmax,output=AV
- runtime_opcode_control_flow=false
- runtime_chess_semantics=false
- runtime_legacy_engine_link=false
- compiler_linked_into_runtime=false

## Exact acceptance

Program:

```text
LOAD_CONST r0,2
LOAD_CONST r1,3
ADD r2,r0,r1
HALT
```

Result:

```text
r2=5
pc_trace=0,1,2,3,3
steps=4
halt=1
```

The native machine test also checks all 136 pairs `(left,right)` with
`left+right<16`, all 16 MOVE values, deterministic repeated traces, and an
explicit pre-HALT step-limit failure.

## Verification

```text
docker run ... cmake -S . -B build ... && cmake --build build && ctest --test-dir build --output-on-failure
3/3 passed; 0 failed; 5.76 s

python -m pytest tests/test_vm2_source_purity.py -q -W error
1 passed
```

The clean orphan branch was configured and built from its root `CMakeLists.txt`.
LibTorch CMake emitted container warnings about NVTX3, NVRTC shorthash,
and a missing optional static Kineto library. Compilation and all tests passed;
VM2 executed on CPU in this slice.

## Purity audit

`native/vm2/tools/audit_runtime.py` checks the four production runtime files and
the CMake link boundary. Its mutation fixture appends an opcode-specific switch
to a copied runtime and verifies that the audit exits with failure.

The offline compiler may use ordinary C++ control flow to validate bytecode and
construct frozen tensors. Production `cmz_vm2` links only the generic attention
primitive and performs no opcode decoding.
