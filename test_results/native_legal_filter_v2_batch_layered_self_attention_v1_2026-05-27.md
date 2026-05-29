# Native legal-filter v2 batched layered self-attention v1

- date=2026-05-27
- change_id=native_legal_filter_v2_batch_layered_self_attention_v1
- scope=replace remaining legal trace batch v1 monolithic legal-filter route with batched legal_filter_v2 layered CUDA self-attention stack
- semantic_target=frozen_2d_self_attention; low_level_substrate=CUDA/CUTLASS-compatible QK -> hardmax/select -> V/write kernels
- c_api=cmz_engine_cuda_legal_filter_v2_batch_attention_count, cmz_engine_cuda_legal_filter_v2_batch_layer_count
- cuda_symbol=cmz_cuda_legal_filter_v2_batch_attention
- layers=batch_board_write_select_attention, batch_king_square_select_attention, batch_short_attack_select_attention, batch_ray_blocker_select_attention, batch_final_legal_select_attention
- route=frozen_legal_trace_attention_packets+legal_moves_uci+resolve_legal_move -> frozen_legal_filter_batch_attention -> cmz_cuda_legal_filter_v2_batch_attention
- old_route_status=cmz_engine_cuda_legal_filter_batch_attention_count remains 0 on targeted legal trace generation
- graph=legal_filter_batch_backend=cuda_legal_filter_v2_batched_layered_self_attention; legal_filter_batch_v1_kernel_remaining=false
- truth_state=current_full_frozen_2d_self_attention_only=false until remaining non-legal-filter rule paths and legacy monolithic CUDA rule symbols are fully audited or removed

## TDD evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_legal_filter_v2_batch_expected_fail_2026-05-27.txt
- expected_fail_reason=missing cmz_engine_cuda_legal_filter_v2_batch_attention_count and cmz_engine_cuda_legal_filter_v2_batch_layer_count before implementation

## Verification

- log=test_results/native_container_logs/cargo_test_legal_filter_v2_batch_targeted_2026-05-27.txt; command=`cd /work/native && cargo test -p cmz-engine-sys frozen_legal_trace_uses_batched_v2_layered_self_attention_without_v1_batch_kernel -- --nocapture`; result=passed
- log=test_results/native_container_logs/cargo_fmt_legal_filter_v2_batch_2026-05-27.txt; command=`cd /work/native && cargo fmt --all -- --check`; result=passed
- log=test_results/native_container_logs/cargo_clippy_legal_filter_v2_batch_2026-05-27.txt; command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`; result=passed
- log=test_results/native_container_logs/cargo_test_legal_filter_v2_batch_2026-05-27.txt; command=`cd /work/native && cargo test --workspace`; result=passed; tests=47 native tests
- log=test_results/legal_filter_v2_batch_pytest_2026-05-27.txt; command=`python -m pytest -p no:cacheprovider -q`; result=passed; tests=146
- log=test_results/legal_filter_v2_batch_pytest_werror_2026-05-27.txt; command=`python -m pytest -p no:cacheprovider -W error -q`; result=passed; tests=146
