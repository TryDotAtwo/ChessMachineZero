# Native CUDA Board Projection v1 - 2026-05-27

## Scope

- change_id=native_cuda_board_projection_v1
- goal=move board trace reconstruction/latest-write projection onto CUDA
- c_api=cmz_engine_cuda_board_projection_count
- cuda_symbol=cmz_cuda_project_board_latest_writes
- graph=board_projection_backend=cuda_latest_write_projection
- attention_mapping=latest-write frozen attention over TracePacket rows writes square_piece_tokens[64] and side_to_move
- route=cmz_engine_project_board_trace uses CUDA latest-write projection
- equivalence=e2e4 make-move trace projects e2=empty, e4=white_pawn, side_to_move=black
- no_fallback=CUDA board projection failure raises explicit native error; CPU fallback forbidden
- counter_contract=cmz_engine_cuda_board_projection_count increments per project_board_trace call

## Verification

- tdd_red=test_results/native_container_logs/cargo_test_cuda_board_projection_expected_fail_2026-05-27.txt
- tdd_red_result=expected failure on missing cmz_engine_cuda_board_projection_count symbol after fixing test index types
- cargo_fmt=test_results/native_container_logs/cargo_fmt_cuda_board_projection_2026-05-27.txt
- cargo_fmt_result=passed
- cargo_test_targeted=test_results/native_container_logs/cargo_test_cuda_board_projection_2026-05-27.txt
- cargo_test_targeted_result=passed; tests=1
- cargo_clippy=test_results/native_container_logs/cargo_clippy_cuda_board_projection_2026-05-27.txt
- cargo_clippy_result=passed
- cargo_test_workspace=test_results/native_container_logs/cargo_test_cuda_board_projection_workspace_2026-05-27.txt
- cargo_test_workspace_result=passed; cmz-engine-sys=41 passed; cmz-dashboard=1 passed; native_total=42 passed
- pytest_full=test_results/cuda_board_projection_pytest_2026-05-27.txt
- pytest_full_result=passed; tests=146
- pytest_werror=test_results/cuda_board_projection_pytest_werror_2026-05-27.txt
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
