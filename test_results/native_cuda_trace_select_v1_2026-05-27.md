# Native CUDA Trace-Select v1 - 2026-05-27

## Scope

- change_id=native_cuda_trace_select_v1
- goal=move one-packet legal trace decode selection onto CUDA
- runtime_mode=native_cuda_trace_select_decoder
- c_api=cmz_engine_cuda_trace_select_count
- cuda_kernel=cmz_cuda_select_trace_packet
- graph_layer=trace_select_backend=cuda_trace_select_packet
- route=cmz_engine_legal_trace_next uses CUDA packet select per decoded packet
- no_fallback=CUDA trace-select failure raises explicit native error; CPU fallback forbidden
- equivalence=streamed legal trace equals full legal_trace_packets on start position
- counter_contract=cmz_engine_cuda_trace_select_count equals decoded packet count

## Verification

- tdd_red=test_results/native_container_logs/cargo_test_cuda_trace_select_expected_fail_2026-05-27.txt
- tdd_red_result=expected failure on missing cmz_engine_cuda_trace_select_count symbol
- cargo_fmt=test_results/native_container_logs/cargo_fmt_cuda_trace_select_2026-05-27.txt
- cargo_fmt_result=passed
- cargo_test_targeted=test_results/native_container_logs/cargo_test_cuda_trace_select_2026-05-27.txt
- cargo_test_targeted_result=passed; tests=1
- cargo_clippy=test_results/native_container_logs/cargo_clippy_cuda_trace_select_2026-05-27.txt
- cargo_clippy_result=passed
- cargo_test_workspace=test_results/native_container_logs/cargo_test_cuda_trace_select_workspace_2026-05-27.txt
- cargo_test_workspace_result=passed; cmz-engine-sys=39 passed; cmz-dashboard=1 passed; native_total=40 passed
- pytest_full=test_results/cuda_trace_select_pytest_2026-05-27.txt
- pytest_full_result=passed; tests=146
- pytest_werror=test_results/cuda_trace_select_pytest_werror_2026-05-27.txt
- pytest_werror_result=passed; tests=146

## Files

- native/cpp/include/cmz_engine.h
- native/cpp/src/cmz_engine.cpp
- native/cpp/src/cmz_cuda_kernels.cu
- native/crates/cmz-engine-sys/src/lib.rs
- docs/native_rust_cpp_cuda_architecture.md
- docs/project_memory.md
- docs/change_history.md
- docs/prompt_history.md
