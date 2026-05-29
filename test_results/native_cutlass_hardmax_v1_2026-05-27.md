# Native CUTLASS Hardmax v1 - 2026-05-27

## Scope

- change_id=native_cutlass_hardmax_v1
- goal=move HullHardmax2D frozen-attention score computation onto CUTLASS GEMM
- cutlass_include=/opt/cutlass/include
- build_contract=missing CUTLASS headers cause CMake fatal error; fallback forbidden
- c_api=cmz_engine_cutlass_hardmax2d_count
- cuda_symbol=cmz_cutlass_hardmax2d_values
- contract=hull_score_backend=cutlass_gemm_2d
- graph=hull_lookup_backend=cutlass_gemm_2d
- attention_mapping=query[1,2] * keys[2,N] -> scores[1,N]
- route=HullHardmax2D uses CUTLASS GEMM scoring over convex-hull vertices
- equivalence=CUTLASS-backed HullHardmax2D result equals dense hardmax argmax on deterministic test vectors
- no_fallback=non-CUTLASS score fallback forbidden; CUTLASS/CUDA failure raises explicit native error
- counter_contract=cmz_engine_cutlass_hardmax2d_count increments per HullHardmax2D call

## Verification

- tdd_red=test_results/native_container_logs/cargo_test_cutlass_hardmax_expected_fail_2026-05-27.txt
- tdd_red_result=expected failure on missing cmz_engine_cutlass_hardmax2d_count symbol
- cargo_fmt=test_results/native_container_logs/cargo_fmt_cutlass_hardmax_2026-05-27.txt
- cargo_fmt_result=passed
- cargo_test_targeted=test_results/native_container_logs/cargo_test_cutlass_hardmax_2026-05-27.txt
- cargo_test_targeted_result=passed; tests=1
- cargo_clippy=test_results/native_container_logs/cargo_clippy_cutlass_hardmax_2026-05-27.txt
- cargo_clippy_result=passed
- cargo_test_workspace=test_results/native_container_logs/cargo_test_cutlass_hardmax_workspace_2026-05-27.txt
- cargo_test_workspace_result=passed; cmz-engine-sys=40 passed; cmz-dashboard=1 passed; native_total=41 passed
- pytest_full=test_results/cutlass_hardmax_pytest_2026-05-27.txt
- pytest_full_result=passed; tests=146
- pytest_werror=test_results/cutlass_hardmax_pytest_werror_2026-05-27.txt
- pytest_werror_result=passed; tests=146

## Files

- native/cpp/CMakeLists.txt
- native/cpp/include/cmz_engine.h
- native/cpp/src/cmz_engine.cpp
- native/cpp/src/cmz_cuda_kernels.cu
- native/crates/cmz-engine-sys/src/lib.rs
- docs/native_rust_cpp_cuda_architecture.md
- docs/project_memory.md
- docs/change_history.md
- docs/prompt_history.md
