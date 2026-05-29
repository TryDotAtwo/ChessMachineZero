# Native CUDA Castle Attention V1

- date=2026-05-27
- scope=castling target masks routed through CUDA castle-path attention
- semantic_contract=frozen_attention_castle_path_filter
- technical_substrate=CUDA kernel; CUTLASS remains used for HullHardmax2D score path
- no_fallback=true

## Attention Mapping

- query=(board_state,castling_rights,side)
- path_empty_attention=checks king-side/queen-side transit squares
- attack_attention=rejects current, transit, and destination king squares when attacked
- output=castle_target_mask

## TDD Red

- command=`cd /work/native && cargo test -p cmz-engine-sys frozen_castle_targets_use_cuda_castle_path_attention_without_cpu_fallback -- --nocapture`
- log=test_results/native_container_logs/cargo_test_cuda_castle_expected_fail_2026-05-27.txt
- expected_failure=linker error; undefined symbols `cmz_engine_frozen_castle_target_mask` and `cmz_engine_cuda_castle_attention_count`

## Implementation

- c_api=cmz_engine_frozen_castle_target_mask; cmz_engine_cuda_castle_attention_count
- cuda_symbol=cmz_cuda_castle_target_attention
- graph_fields=castling_targets_backend=cuda_castle_path_attention
- route=cmz_engine_frozen_castle_target_mask -> frozen_castle_target_mask_attention -> cmz_cuda_castle_target_attention
- internal_route=frozen_castle_target_mask_layer -> cmz_cuda_castle_target_attention
- failure_mode=CUDA error becomes explicit native error; CPU castle-target fallback forbidden

## Verification

- command=`cd /work/native && cargo test -p cmz-engine-sys frozen_castle_targets_use_cuda_castle_path_attention_without_cpu_fallback -- --nocapture`
- log=test_results/native_container_logs/cargo_test_cuda_castle_2026-05-27.txt
- result=1 passed

- command=`cd /work/native && cargo fmt --all -- --check`
- log=test_results/native_container_logs/cargo_fmt_cuda_castle_2026-05-27.txt
- result=passed

- command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`
- log=test_results/native_container_logs/cargo_clippy_cuda_castle_2026-05-27.txt
- result=passed

- command=`cd /work/native && cargo test --workspace`
- log=test_results/native_container_logs/cargo_test_cuda_castle_workspace_2026-05-27.txt
- result=46 native tests passed

- command=`py -m pytest -p no:cacheprovider -q`
- log=test_results/cuda_castle_pytest_2026-05-27.txt
- result=passed; 146 tests reached 100%

- command=`py -m pytest -p no:cacheprovider -W error -q`
- log=test_results/cuda_castle_pytest_werror_2026-05-27.txt
- result=passed; 146 tests reached 100%
