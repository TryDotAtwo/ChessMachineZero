# Native CUDA Attack Table Attention V1

- date=2026-05-27
- scope=static attack-mask table lookup routed through CUDA QK-hardmax-V frozen attention
- semantic_contract=frozen_attention_table_lookup
- technical_substrate=CUDA kernel; CUTLASS remains used for HullHardmax2D score path
- no_fallback=true

## TDD Red

- command=`cd /work/native && cargo test -p cmz-engine-sys frozen_attack_mask_uses_cuda_qk_hardmax_v_table_attention_without_cpu_fallback -- --nocapture`
- log=test_results/native_container_logs/cargo_test_cuda_attack_table_expected_fail_2026-05-27.txt
- expected_failure=linker error; undefined symbol `cmz_engine_cuda_attack_table_attention_count`

## Implementation

- c_api=cmz_engine_cuda_attack_table_attention_count
- cuda_symbol=cmz_cuda_attack_table_lookup_attention
- graph_fields=attack_masks_backend=cuda_qk_hardmax_v_table_lookup; table_lookup_semantics=qk_hardmax_v
- route=cmz_engine_frozen_attack_mask -> frozen_attack_table_attention_lookup -> cmz_cuda_attack_table_lookup_attention
- attention_mapping=query=(piece_token,square); keys=frozen_attack_table_keys; values=attack_masks; hardmax selects one value
- failure_mode=CUDA error becomes explicit native error; CPU lookup fallback forbidden

## Verification

- command=`cd /work/native && cargo test -p cmz-engine-sys frozen_attack_mask_uses_cuda_qk_hardmax_v_table_attention_without_cpu_fallback -- --nocapture`
- log=test_results/native_container_logs/cargo_test_cuda_attack_table_2026-05-27.txt
- result=1 passed

- command=`cd /work/native && cargo fmt --all -- --check`
- log=test_results/native_container_logs/cargo_fmt_cuda_attack_table_2026-05-27.txt
- result=passed

- command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`
- log=test_results/native_container_logs/cargo_clippy_cuda_attack_table_2026-05-27.txt
- result=passed

- command=`cd /work/native && cargo test --workspace`
- log=test_results/native_container_logs/cargo_test_cuda_attack_table_workspace_2026-05-27.txt
- result=42 native tests passed

- command=`py -m pytest -p no:cacheprovider -q`
- log=test_results/cuda_attack_table_pytest_2026-05-27.txt
- result=passed; 146 tests reached 100%

- command=`py -m pytest -p no:cacheprovider -W error -q`
- log=test_results/cuda_attack_table_pytest_werror_2026-05-27.txt
- result=passed; 146 tests reached 100%
