# Native CUDA Legal-Filter Batch Attention V1

- date=2026-05-27
- change_id=native_cuda_legal_filter_batch_attention_v1
- user_target=fuse_many_small_CUDA_launches_into_batched_CUTLASS_attention_kernels
- scope=first launch-fusion slice; batched king-safety legal-filter attention for legal trace generation, legal move listing, and make-move resolution
- semantic_contract=frozen_attention_table_lookup_and_dynamic_attention; CUDA is low-level QK/hardmax/V execution substrate; CPU fallback forbidden

## Implementation

- c_api_added=cmz_engine_cuda_legal_filter_batch_attention_count
- cuda_symbol_added=cmz_cuda_legal_filter_batch_attention
- cuda_kernel_added=cmz_legal_filter_batch_attention_kernel
- graph_metadata_added=legal_filter_batch_backend=cuda_batched_king_safety_attention; small_launch_fusion=legal_filter_batch
- route=frozen_legal_trace_attention_tokens calls frozen_legal_filter_batch_attention once for the pseudo-move vector
- route=legal_moves_uci uses frozen_legal_filter_batch_attention instead of per-move legal_after_king_filter
- route=resolve_legal_move uses frozen_legal_filter_batch_attention instead of per-move legal_after_king_filter
- counter_invariant=start_position_frozen_legal_trace_batch_count=1; start_position_frozen_legal_trace_single_legal_filter_count=0

## TDD Evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_cuda_legal_filter_batch_expected_fail_2026-05-27.txt
- expected_fail_reason=Engine::cuda_legal_filter_batch_attention_count missing
- targeted_pass_log=test_results/native_container_logs/cargo_test_cuda_legal_filter_batch_targeted_2026-05-27.txt
- targeted_pass_result=1 passed

## Final Verification

- command=`cd /work/native && cargo fmt --all -- --check`
- log=test_results/native_container_logs/cargo_fmt_cuda_legal_filter_batch_2026-05-27.txt
- result=passed

- command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`
- log=test_results/native_container_logs/cargo_clippy_cuda_legal_filter_batch_2026-05-27.txt
- result=passed

- command=`cd /work/native && cargo test --workspace`
- log=test_results/native_container_logs/cargo_test_cuda_legal_filter_batch_2026-05-27.txt
- result=passed; native_tests=47

- command=`py -m pytest -p no:cacheprovider -q`
- log=test_results/cuda_legal_filter_batch_pytest_2026-05-27.txt
- result=passed; python_tests=146

- command=`py -m pytest -p no:cacheprovider -W error -q`
- log=test_results/cuda_legal_filter_batch_pytest_werror_2026-05-27.txt
- result=passed; python_tests=146

## Remaining Work

- next_candidate=batch candidate-target generation to reduce one-kernel-per-piece calls
- next_candidate=batch trace-write/packet emission paths where C ABI output layout permits direct GPU writes
- next_candidate=CUTLASS-backed batched top-k where score GEMM dominates lookup cost
