# Native Full Attention Hot Path Finish V1 - 2026-05-27

- scope=remaining native hot rule-path C++ control-flow categories moved to CUDA attention entry points
- trace_append=cmz_cuda_emit_trace_packet_attention
- candidate_generation=cmz_cuda_candidate_moves_attention
- resolve_move=cmz_cuda_resolve_move_attention
- terminal_status=cmz_cuda_terminal_status_attention
- contract_cpp_control_flow_rule_vm_remaining=false
- contract_full_frozen_attention_only=false
- strict_qk_layer_split_remaining=candidate_moves_attention,terminal_status_attention

## TDD Logs

- trace_append_expected_fail=test_results/native_container_logs/cargo_test_trace_append_cuda_expected_fail_2026-05-27.txt
- trace_append_targeted_pass=test_results/native_container_logs/cargo_test_trace_append_cuda_targeted_2026-05-27.txt
- candidate_moves_expected_fail=test_results/native_container_logs/cargo_test_candidate_moves_cuda_expected_fail_2026-05-27.txt
- candidate_moves_targeted_pass=test_results/native_container_logs/cargo_test_candidate_moves_cuda_targeted_2026-05-27.txt
- terminal_expected_fail=test_results/native_container_logs/cargo_test_terminal_cuda_expected_fail_2026-05-27.txt
- terminal_targeted_pass=test_results/native_container_logs/cargo_test_terminal_cuda_targeted_2026-05-27.txt
- resolve_move_expected_fail=test_results/native_container_logs/cargo_test_resolve_move_cuda_expected_fail_2026-05-27.txt
- resolve_move_targeted_pass=test_results/native_container_logs/cargo_test_resolve_move_cuda_targeted_2026-05-27.txt

## Verification

- cargo_fmt=test_results/native_container_logs/cargo_fmt_full_attention_finish_2026-05-27.txt
- cargo_fmt_result=passed
- cargo_clippy=test_results/native_container_logs/cargo_clippy_full_attention_finish_2026-05-27.txt
- cargo_clippy_result=passed with `-D warnings`
- cargo_test=test_results/native_container_logs/cargo_test_full_attention_finish_2026-05-27.txt
- cargo_test_result=47 native tests passed
- pytest=test_results/full_attention_finish_pytest_2026-05-27.txt
- pytest_result=passed; 146 tests reached 100%
- pytest_werror=test_results/full_attention_finish_pytest_werror_2026-05-27.txt
- pytest_werror_result=passed; 146 tests reached 100%

## Remaining Strict Work

- remaining=candidate_moves_attention fused CUDA kernel must be split into explicit QK/hardmax/V/write frozen 2D attention layers
- remaining=terminal_status_attention fused CUDA kernel must be split into explicit QK/hardmax/V/status-write frozen 2D attention layers
- remaining=trace packet emission should be batched to remove per-packet cudaMalloc/cudaFree overhead
