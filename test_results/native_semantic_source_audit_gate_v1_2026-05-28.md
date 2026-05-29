# Native Semantic Source Audit Gate V1 - 2026-05-28

```text
entity_id=native_semantic_source_audit_gate_v1
status=implemented_verified
target_full_frozen_attention_only=true
full_frozen_attention_only=false
semantic_attention_purity=false
semantic_source_audit=rust_cuda_body_scan_v1
metadata_only_tests_remaining=false
```

## Scope

```text
removed_gap=tests_assert_metadata_not_semantics
new_remaining_non_attention_paths=candidate_target_mask_chess_control_flow,candidate_record_emit_serial_loop,terminal_legal_presence_chess_search,terminal_material_counting_control_flow,terminal_check_state_king_scan,castle_target_chess_control_flow,legal_filter_batch_attack_chess_control_flow,legal_filter_batch_ray_scan_control_flow
contract_truth=full_frozen_attention_only remains false until source-body offenders are lowered to pure frozen 2D attention
```

## TDD Evidence

```text
expected_fail_log=test_results/native_container_logs/cargo_test_semantic_source_audit_expected_fail_2026-05-28.txt
expected_fail_reason=graph missing candidate_target_mask_chess_control_flow while CUDA source-body scanner detected offender body
targeted_pass_log=test_results/native_container_logs/cargo_test_semantic_source_audit_targeted_2026-05-28.txt
targeted_pass_result=cmz-engine-sys package tests passed; 50 passed
```

## Final Verification

```text
cargo_fmt_apply=test_results/native_container_logs/cargo_fmt_apply_semantic_source_audit_2026-05-28.txt; status=passed
cargo_fmt_check=test_results/native_container_logs/cargo_fmt_semantic_source_audit_2026-05-28.txt; status=passed
cargo_clippy=test_results/native_container_logs/cargo_clippy_semantic_source_audit_2026-05-28.txt; status=passed
cargo_test_workspace=test_results/native_container_logs/cargo_test_semantic_source_audit_2026-05-28.txt; status=passed; native_tests=51
pytest=test_results/semantic_source_audit_pytest_2026-05-28.txt; status=passed
pytest_werror=test_results/semantic_source_audit_pytest_werror_2026-05-28.txt; status=passed
```
