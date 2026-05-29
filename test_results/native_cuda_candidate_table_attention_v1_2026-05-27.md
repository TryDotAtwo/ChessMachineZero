# Native CUDA Candidate Table Attention V1

- date=2026-05-27
- scope=candidate target masks routed through CUDA frozen attention target lookup
- semantic_contract=2D/frozen_attention_QK_hardmax_V_target_lookup
- technical_substrate=CUDA kernel; CUTLASS remains used for HullHardmax2D score path
- no_fallback=true

## Table Lookup Meaning

- table_lookup=select one or more frozen rule values by key
- attention_form=query→QK_scores→hardmax→V
- chess_rule_example=(piece_token,square,occupancy_state) selects target mask logic/value
- non_goal=not memorized board positions; not human-game data; not engine labels

## TDD Red

- command=`cd /work/native && cargo test -p cmz-engine-sys frozen_candidate_targets_use_cuda_qk_hardmax_v_attention_without_cpu_fallback -- --nocapture`
- log=test_results/native_container_logs/cargo_test_cuda_candidate_table_expected_fail_2026-05-27.txt
- expected_failure=linker error; undefined symbol `cmz_engine_cuda_candidate_table_attention_count`

## Implementation

- c_api=cmz_engine_cuda_candidate_table_attention_count
- cuda_symbol=cmz_cuda_candidate_target_attention
- graph_fields=candidate_targets_backend=cuda_qk_hardmax_v_target_lookup; candidate_filter_backend=cuda_dynamic_mask_attention
- route=cmz_engine_frozen_candidate_target_mask -> frozen_candidate_target_mask_attention -> cmz_cuda_candidate_target_attention
- internal_route=frozen_candidate_target_mask_layer -> cmz_cuda_candidate_target_attention
- attention_mapping=query=(piece_token,square,occupancy_state); values=target_masks; dynamic_filters=friendly/enemy/occupancy/en_passant
- failure_mode=CUDA error becomes explicit native error; CPU candidate-target fallback forbidden

## Verification

- command=`cd /work/native && cargo test -p cmz-engine-sys frozen_candidate_targets_use_cuda_qk_hardmax_v_attention_without_cpu_fallback -- --nocapture`
- log=test_results/native_container_logs/cargo_test_cuda_candidate_table_2026-05-27.txt
- result=1 passed

- command=`cd /work/native && cargo fmt --all -- --check`
- log=test_results/native_container_logs/cargo_fmt_cuda_candidate_table_2026-05-27.txt
- result=passed

- command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`
- log=test_results/native_container_logs/cargo_clippy_cuda_candidate_table_2026-05-27.txt
- result=passed

- command=`cd /work/native && cargo test --workspace`
- log=test_results/native_container_logs/cargo_test_cuda_candidate_table_workspace_2026-05-27.txt
- result=43 native tests passed

- command=`py -m pytest -p no:cacheprovider -q`
- log=test_results/cuda_candidate_table_pytest_2026-05-27.txt
- result=passed; 146 tests reached 100%

- command=`py -m pytest -p no:cacheprovider -W error -q`
- log=test_results/cuda_candidate_table_pytest_werror_2026-05-27.txt
- result=passed; 146 tests reached 100%
