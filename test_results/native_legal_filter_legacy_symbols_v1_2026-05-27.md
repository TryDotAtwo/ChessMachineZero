# Native legal-filter legacy symbol cleanup v1

- date=2026-05-27
- change_id=native_legal_filter_legacy_symbols_v1
- scope=remove unused monolithic v1 legal-filter CUDA symbols after legal_filter_v2 single and batch became the only legal-filter routes
- removed_symbols=cmz_legal_filter_eval, cmz_legal_filter_attention_kernel, cmz_legal_filter_batch_attention_kernel, cmz_cuda_legal_filter_attention, cmz_cuda_legal_filter_batch_attention
- retained_observability=cmz_engine_cuda_legal_filter_attention_count and cmz_engine_cuda_legal_filter_batch_attention_count stay exported as zero-regression counters
- graph=legacy_legal_filter_cuda_symbols_present=false
- rg_check=old v1 legal-filter CUDA symbol names absent from native C++/CUDA/Rust sources

## TDD evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_legal_filter_legacy_symbols_expected_fail_2026-05-27.txt
- expected_fail_reason=graph lacked legacy_legal_filter_cuda_symbols_present=false before source cleanup

## Verification

- log=test_results/native_container_logs/cargo_test_legal_filter_legacy_symbols_targeted_2026-05-27.txt; command=`cd /work/native && cargo test -p cmz-engine-sys v2 -- --nocapture`; result=passed; tests=2
- log=test_results/native_container_logs/cargo_fmt_legal_filter_legacy_symbols_2026-05-27.txt; command=`cd /work/native && cargo fmt --all -- --check`; result=passed
- log=test_results/native_container_logs/cargo_clippy_legal_filter_legacy_symbols_2026-05-27.txt; command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`; result=passed
- log=test_results/native_container_logs/cargo_test_legal_filter_legacy_symbols_2026-05-27.txt; command=`cd /work/native && cargo test --workspace`; result=passed; tests=47 native tests
- log=test_results/legal_filter_legacy_symbols_pytest_2026-05-27.txt; command=`python -m pytest -p no:cacheprovider -q`; result=passed; tests=146
- log=test_results/legal_filter_legacy_symbols_pytest_werror_2026-05-27.txt; command=`python -m pytest -p no:cacheprovider -W error -q`; result=passed; tests=146
