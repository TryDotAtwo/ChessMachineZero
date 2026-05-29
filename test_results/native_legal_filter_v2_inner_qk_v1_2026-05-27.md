# Native legal-filter v2 inner QK hardmax helpers v1

- date=2026-05-27
- change_id=native_legal_filter_v2_inner_qk_v1
- user_question=are internal score/select cycles also implemented as 2D frozen attention
- truthful_start_state=outer legal_filter_v2 layers were frozen self-attention; previous inner score/select loops were partly plain CUDA priority/argmax code
- implemented_state=inner priority/argmax score-select uses explicit 2D QK hardmax helper semantics
- cuda_helpers=cmz_qk2_score_u32, cmz_qk2_hardmax_select_u32
- scope=board_write_select, king_square_select, attack_source_select, ray_blocker_select; single and batch variants
- graph=legal_filter_v2_inner_select=qk_hardmax_2d_helpers; legal_filter_v2_inner_select_plain_cuda_loops_remaining=false
- truth_state=iteration over attention keys still occurs inside CUDA kernels; score/select operation is explicit QK(2D) -> hardmax/select -> value

## TDD evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_legal_filter_v2_inner_qk_expected_fail_2026-05-27.txt
- expected_fail_reason=graph lacked legal_filter_v2_inner_select and legal_filter_v2_inner_select_plain_cuda_loops_remaining fields before implementation

## Verification

- log=test_results/native_container_logs/cargo_test_legal_filter_v2_inner_qk_targeted_2026-05-27.txt; command=`cd /work/native && cargo test -p cmz-engine-sys v2 -- --nocapture`; result=passed; tests=2
- log=test_results/native_container_logs/cargo_fmt_legal_filter_v2_inner_qk_2026-05-27.txt; command=`cd /work/native && cargo fmt --all -- --check`; result=passed
- log=test_results/native_container_logs/cargo_clippy_legal_filter_v2_inner_qk_2026-05-27.txt; command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`; result=passed
- log=test_results/native_container_logs/cargo_test_legal_filter_v2_inner_qk_2026-05-27.txt; command=`cd /work/native && cargo test --workspace`; result=passed; tests=47 native tests
- log=test_results/legal_filter_v2_inner_qk_pytest_2026-05-27.txt; command=`python -m pytest -p no:cacheprovider -q`; result=passed; tests=146
- log=test_results/legal_filter_v2_inner_qk_pytest_werror_2026-05-27.txt; command=`python -m pytest -p no:cacheprovider -W error -q`; result=passed; tests=146
