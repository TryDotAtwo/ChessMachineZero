# Native Legal-Filter V2 Layered Self-Attention V1

- date=2026-05-27
- change_id=native_legal_filter_v2_layered_self_attention_v1
- user_target=translate legal_filter into layered QK -> hardmax/select -> V/write graph
- scope=single-move `cmz_engine_frozen_move_legal` route

## Implementation

- c_api_added=cmz_engine_cuda_legal_filter_v2_attention_count
- c_api_added=cmz_engine_cuda_legal_filter_v2_layer_count
- cuda_symbol_added=cmz_cuda_legal_filter_v2_attention
- route_changed=cmz_engine_frozen_move_legal now calls legal_filter_v2 layered self-attention path
- old_single_kernel_route=cmz_cuda_legal_filter_attention no longer used by cmz_engine_frozen_move_legal
- remaining_monolithic_route=cmz_cuda_legal_filter_batch_attention still used by legal trace batch path

## V2 Layers

- layer=board_write_select_attention; semantics=QK hardmax selects old-piece/from-clear/to-write/en-passant-clear/castle-rook-write value per square
- layer=king_square_select_attention; semantics=QK hardmax selects own king square from next board
- layer=short_attack_select_attention; semantics=QK hardmax selects pawn/knight/king attacker status
- layer=ray_blocker_select_attention; semantics=QK hardmax selects nearest blocker per king ray and checks slider attacker
- layer=final_legal_select_attention; semantics=QK select combines king_found and attack bits into legal bit

## Graph Contract

- legal_filter_backend=cuda_legal_filter_v2_layered_self_attention
- legal_filter_v2_current_backend=cuda_qk_hardmax_v_write_layers
- legal_filter_v2_layers_started=board_write_select,king_square_select,short_attack_select,ray_blocker_select,final_legal_select
- legal_filter_v1_single_kernel_remaining=false
- current_full_frozen_2d_self_attention_only=false
- reason=batch legal trace path still uses monolithic v1 batch kernel

## TDD Evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_legal_filter_v2_expected_fail_2026-05-27.txt
- expected_fail_reason=missing cmz_engine_cuda_legal_filter_v2_attention_count and cmz_engine_cuda_legal_filter_v2_layer_count symbols
- targeted_pass_log=test_results/native_container_logs/cargo_test_legal_filter_v2_targeted_2026-05-27.txt
- targeted_pass_result=1 passed

## Final Verification

- command=`cd /work/native && cargo fmt --all -- --check`
- log=test_results/native_container_logs/cargo_fmt_legal_filter_v2_2026-05-27.txt
- result=passed

- command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`
- log=test_results/native_container_logs/cargo_clippy_legal_filter_v2_2026-05-27.txt
- result=passed

- command=`cd /work/native && cargo test --workspace`
- log=test_results/native_container_logs/cargo_test_legal_filter_v2_2026-05-27.txt
- result=passed; native_tests=47

- command=`py -m pytest -p no:cacheprovider -q`
- log=test_results/legal_filter_v2_pytest_2026-05-27.txt
- result=passed; python_tests=146

- command=`py -m pytest -p no:cacheprovider -W error -q`
- log=test_results/legal_filter_v2_pytest_werror_2026-05-27.txt
- result=passed; python_tests=146

## Next Step

- next=replace cmz_cuda_legal_filter_batch_attention with batched legal_filter_v2 layer stack
- acceptance=legal trace path increments v2 layer counters and does not call monolithic v1 batch kernel
