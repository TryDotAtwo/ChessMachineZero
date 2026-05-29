# Native legal-filter v2 complete required layers v1

- date=2026-05-27
- change_id=native_legal_filter_v2_complete_layers_v1
- scope=split legal_filter_v2 into all required frozen self-attention layers for single-move and batched routes
- semantic_target=frozen_2d_self_attention; low_level_substrate=CUDA/CUTLASS-compatible QK -> hardmax/select -> V/write kernels
- required_layers=move_type_select, board_write_select, en_passant_capture_select, castle_rook_write_select, promotion_select, king_square_select, attack_source_select, ray_blocker_select, final_legal_select
- single_route=cmz_engine_frozen_move_legal -> cmz_cuda_legal_filter_v2_attention; layer_count_per_call>=9; v1_single_counter=0
- batch_route=frozen_legal_trace_attention_packets+legal_moves_uci+resolve_legal_move -> cmz_cuda_legal_filter_v2_batch_attention; layer_count_per_batch>=9; v1_batch_counter=0
- graph=legal_filter_v2_layers_complete lists all required layers; small_launch_fusion=legal_filter_batch_v2
- truth_state=current_full_frozen_2d_self_attention_only=false until remaining non-legal-filter rule paths and legacy monolithic CUDA rule symbols are fully audited or removed

## TDD evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_legal_filter_v2_complete_layers_expected_fail_2026-05-27.txt
- expected_fail_reason=previous legal_filter_v2 layer counters were 5 instead of required 9; graph lacked legal_filter_v2_layers_complete

## Verification

- log=test_results/native_container_logs/cargo_test_legal_filter_v2_complete_layers_targeted_2026-05-27.txt; command=`cd /work/native && cargo test -p cmz-engine-sys v2 -- --nocapture`; result=passed; tests=2
- log=test_results/native_container_logs/cargo_fmt_legal_filter_v2_complete_layers_2026-05-27.txt; command=`cd /work/native && cargo fmt --all -- --check`; result=passed
- log=test_results/native_container_logs/cargo_clippy_legal_filter_v2_complete_layers_2026-05-27.txt; command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`; result=passed
- log=test_results/native_container_logs/cargo_test_legal_filter_v2_complete_layers_2026-05-27.txt; command=`cd /work/native && cargo test --workspace`; result=passed; tests=47 native tests
- log=test_results/legal_filter_v2_complete_layers_pytest_2026-05-27.txt; command=`python -m pytest -p no:cacheprovider -q`; result=passed; tests=146
- log=test_results/legal_filter_v2_complete_layers_pytest_werror_2026-05-27.txt; command=`python -m pytest -p no:cacheprovider -W error -q`; result=passed; tests=146
