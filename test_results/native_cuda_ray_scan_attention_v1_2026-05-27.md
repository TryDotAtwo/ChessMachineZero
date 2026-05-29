# Native CUDA Ray Scan Attention V1

- date=2026-05-27
- scope=blocker-aware ray scan routed through CUDA nearest-blocker frozen attention
- semantic_contract=2D/frozen_attention_QK_hardmax_V_nearest_blocker
- technical_substrate=CUDA kernel; CUTLASS remains used for HullHardmax2D score path
- no_fallback=true

## Attention Mapping

- query=(from_square,direction)
- keys=ray_squares_along_direction
- hardmax_select=nearest_occupied_square_or_edge
- values=square_masks
- output=blocker_inclusive_ray_mask

## TDD Red

- command=`cd /work/native && cargo test -p cmz-engine-sys frozen_ray_scan_uses_cuda_nearest_blocker_attention_without_cpu_fallback -- --nocapture`
- log=test_results/native_container_logs/cargo_test_cuda_ray_scan_expected_fail_2026-05-27.txt
- expected_failure=linker error; undefined symbol `cmz_engine_cuda_ray_scan_attention_count`

## Implementation

- c_api=cmz_engine_cuda_ray_scan_attention_count
- cuda_symbol=cmz_cuda_ray_scan_attention
- graph_fields=ray_scan_backend=cuda_nearest_blocker_attention; ray_scan_semantics=qk_hardmax_v_nearest_blocker
- route=cmz_engine_frozen_ray_scan_mask -> frozen_ray_scan_mask_attention -> cmz_cuda_ray_scan_attention
- internal_route=frozen_ray_scan_mask_layer -> cmz_cuda_ray_scan_attention
- failure_mode=CUDA error becomes explicit native error; CPU ray-scan fallback forbidden

## Verification

- command=`cd /work/native && cargo test -p cmz-engine-sys frozen_ray_scan_uses_cuda_nearest_blocker_attention_without_cpu_fallback -- --nocapture`
- log=test_results/native_container_logs/cargo_test_cuda_ray_scan_2026-05-27.txt
- result=1 passed

- command=`cd /work/native && cargo fmt --all -- --check`
- log=test_results/native_container_logs/cargo_fmt_cuda_ray_scan_2026-05-27.txt
- result=passed

- command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`
- log=test_results/native_container_logs/cargo_clippy_cuda_ray_scan_2026-05-27.txt
- result=passed

- command=`cd /work/native && cargo test --workspace`
- log=test_results/native_container_logs/cargo_test_cuda_ray_scan_workspace_2026-05-27.txt
- result=44 native tests passed

- command=`py -m pytest -p no:cacheprovider -q`
- log=test_results/cuda_ray_scan_pytest_2026-05-27.txt
- result=passed; 146 tests reached 100%

- command=`py -m pytest -p no:cacheprovider -W error -q`
- log=test_results/cuda_ray_scan_pytest_werror_2026-05-27.txt
- result=passed; 146 tests reached 100%
