# Self-Play Step Performance Baseline

## Environment

- date=2026-05-25
- torch=2.8.0+cu128
- torch_cuda_available=true
- torch_cuda_version=12.8
- gpu=NVIDIA GeForce RTX 3070 Laptop GPU
- nvidia_driver=572.70
- nvidia_cuda_driver=12.8
- nvcc_available=true
- ninja_available=true
- msvc_cl_available=false
- cutlass_source_found=C:\tmp\cmz_ascii\external\cutlass

## One Step Profile

```text
init_seconds=0.959327
step_seconds=6.793559
move=a2a3
trace_packets=117
```

```text
98492 function calls (94831 primitive calls) in 6.791 seconds

execute_trace: 50 calls, 6.509s cumulative
_candidate_table: 52 calls, 6.445s cumulative
decode_legal_tensor_trace_host_append_only: 5.396s cumulative
decode_next_legal_tensor_packet: 41 calls, 5.392s cumulative
legal_tensor_trace_from_prompt_tensor: 41 calls, 5.377s cumulative
_emit_legal_trace: 41 calls, 5.375s cumulative
_lookup_rows: 1126 calls, 5.218s cumulative
_attention_select: 1444 calls, 3.917s cumulative
_board_read: 394 calls, 3.564s cumulative
torch.argmax: 1505 calls, 1.623s cumulative
torch.one_hot: 2676 calls, 1.389s cumulative
Tensor.to: 12436 calls, 0.984s cumulative
```

## Continuation Recompute Check

```text
full_legal_once_seconds=0.113989; rows=109
host_append_legal_seconds=4.554635; rows=109
```

## Finding

- primary_bottleneck=current host-append decode recomputes the full frozen-attention legal trace for every emitted token.
- measured_overhead=host_append_legal_seconds/full_legal_once_seconds ~= 39.96x.
- second_bottleneck=matrix attention interpreter is implemented as many small CPU/PyTorch tensor ops.
- cutlass_prerequisite=fix algorithmic re-execution first, then fuse and move kernels to CUDA/CUTLASS.
