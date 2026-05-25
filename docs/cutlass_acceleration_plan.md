# CUTLASS Acceleration Plan

## Current Baseline

- date=2026-05-25
- device=NVIDIA GeForce RTX 3070 Laptop GPU
- torch=2.8.0+cu128
- torch_cuda_available=true
- cuda_driver=12.8
- nvcc_available=true
- nvcc_paths=[CUDA v12.5, CUDA v12.2, CUDA v10.2]
- ninja_available=true
- msvc_cl_available=false
- local_cutlass_source=C:\tmp\cmz_ascii\external\cutlass

## Measured Slow Path

- benchmark=one dashboard transformer self-play step from starting position
- init_seconds=0.959327
- step_seconds=6.793559
- emitted_trace_packets=117
- full_legal_trace_once_seconds=0.113989
- host_append_legal_trace_seconds=4.554635
- ratio=host_append_legal_trace_seconds/full_legal_trace_once_seconds ~= 39.96

## Root Cause

- root_cause_1=current token streaming loop recomputes the whole legal trace continuation for every emitted token.
- root_cause_1_location=src/chess_machine_zero/model/percepta_frozen_attention_vm.py::decode_legal_tensor_trace_host_append_only
- root_cause_1_effect=41 legal tokens trigger 41 full legal-trace executions.
- root_cause_2=current matrix attention interpreter runs many tiny CPU/PyTorch ops.
- root_cause_2_hotspots=FrozenMatrixAttentionInterpreter._candidate_table, _lookup_rows, _attention_select, _board_read.
- root_cause_3=current tensors are CPU tensors in rule runtime; GPU exists but the rule path is not GPU-resident.

## cProfile Hotspots

```text
step_seconds=6.793559
execute_trace: 50 calls, 6.509s cumulative
_candidate_table: 52 calls, 6.445s cumulative
decode_legal_tensor_trace_host_append_only: 5.396s cumulative
_emit_legal_trace: 41 calls, 5.375s cumulative
_lookup_rows: 1126 calls, 5.218s cumulative
_attention_select: 1444 calls, 3.917s cumulative
torch.argmax: 1505 calls, 1.623s cumulative
torch.one_hot: 2676 calls, 1.389s cumulative
Tensor.to: 12436 calls, 0.984s cumulative
```

## Required Order

1. Remove algorithmic re-execution before CUDA work.
   - Implement a decode-session/KV-cache equivalent inside the frozen-attention runtime.
   - The runtime must compute a continuation once per prompt/entrypoint and then stream one token at a time from model-owned state.
   - Prefix corruption checks must remain exact.
   - No host-side legality computation may be introduced.

2. Move matrix-interpreter tensors to a device-stable layout.
   - Eliminate repeated `.to(torch.long)` calls.
   - Store rule tables and prompt trace tensors on one selected device.
   - Keep CPU display conversion only at dashboard/API boundary.

3. Replace tiny PyTorch attention calls with a fused CUDA extension.
   - Target kernels: hardmax/select over small row sets, lookup rows, board latest-write read, candidate table generation.
   - Use CUTLASS/CuTe for GEMM-like QK/V patterns and custom CUDA for bit/predicate-heavy kernels.
   - Keep Python reference path as test oracle only, not as a silent runtime fallback.

4. Add parity tests.
   - CPU reference tensor output must equal CUDA/CUTLASS output for arbitrary deterministic FENs.
   - If CUDA extension is requested and unavailable, tests must fail explicitly.
   - No smoke tests; parity and invariant tests only.

## Build Blockers

- blocker=MSVC `cl.exe` is not available on PATH and was not found under `C:\Program Files\Microsoft Visual Studio`.
- implication=Windows CUDA extension build will fail until Visual Studio Build Tools or a configured Developer Command Prompt is available.
- available=CUTLASS source exists at `C:\tmp\cmz_ascii\external\cutlass`; nvcc and ninja exist.

## Immediate Next Implementation Target

- target=decode-session cache in `PerceptaFrozenAttentionRuleComputer`
- expected_gain=legal trace streaming from ~4.55s toward ~0.11s before CUTLASS
- correctness_contract=token-by-token API unchanged; emitted tokens unchanged; corruption rejection unchanged; dashboard trace log unchanged
