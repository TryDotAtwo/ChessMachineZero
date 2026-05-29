# Native Candidate Slider Ray Step Slots V1

```text
date=2026-05-28
slice=native_candidate_slider_ray_step_slots_v1
target_gap=candidate_slider_ray_slot_control_flow
replacement_gap=candidate_slider_ray_step_condition_control_flow
target_backend=qk_explicit_7_step_ray_slot_writes
full_frozen_attention_only=false
semantic_attention_purity=false
```

## Scope

```text
updated=native/cpp/src/cmz_cuda_kernels.cu
updated=native/cpp/src/cmz_engine.cpp
updated=native/crates/cmz-engine-sys/src/lib.rs
updated=docs/project_memory.md
updated=docs/change_history.md
updated=docs/prompt_history.md
updated=docs/native_rust_cpp_cuda_architecture.md
updated=docs/cmz_percepta_policy_only_architecture_review.md
updated=docs/chess_machine_zero_percepta_architecture.md
```

`cmz_candidate_slider_ray_slot_attention_value` now materializes seven explicit distance-step slots and writes selected step values through `cmz_qk2_select_or_write_u64`. The ray-slot body no longer calls `cmz_add_ray_targets` and no longer contains a `while` ray scan. The remaining step-condition helper still contains board-bound and prior-blocker control-flow, so the truthful remaining gap is `candidate_slider_ray_step_condition_control_flow`.

## TDD Logs

```text
expected_fail=test_results/native_container_logs/cargo_test_candidate_slider_ray_step_expected_fail_2026-05-28.txt
targeted_pass=test_results/native_container_logs/cargo_test_candidate_slider_ray_step_targeted_2026-05-28.txt
package_pass=test_results/native_container_logs/cargo_test_candidate_slider_ray_step_package_2026-05-28.txt
```

## Verification

```text
cd /work/native && cargo fmt --all -- --check = passed
cd /work/native && cargo clippy --workspace --all-targets -- -D warnings = passed
cd /work/native && cargo test --workspace = passed, 58 native engine tests plus dashboard test
python -m pytest -p no:cacheprovider -q = passed, 112 tests
python -m pytest -p no:cacheprovider -W error -q = passed, 112 tests
```
