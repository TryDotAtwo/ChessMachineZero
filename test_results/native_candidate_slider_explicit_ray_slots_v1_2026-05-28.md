# Native Candidate Slider Explicit Ray Slots V1

```text
date=2026-05-28
slice=native_candidate_slider_explicit_ray_slots_v1
target_gap=candidate_slider_target_mask_control_flow
replacement_gap=candidate_slider_ray_slot_control_flow
target_backend=qk_explicit_slider_ray_slot_writes
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

`cmz_candidate_slider_target_mask_attention_value` now materializes eight explicit ray slots and writes diagonal/orthogonal selected slots through `cmz_qk2_select_or_write_u64`. The old top-level `if (bishop || queen)` and `if (rook || queen)` branch blocks are removed from the slider target body. The remaining ray-slot helper still calls blocker-aware ray expansion, so the truthful remaining gap is `candidate_slider_ray_slot_control_flow`.

## TDD Logs

```text
expected_fail=test_results/native_container_logs/cargo_test_candidate_slider_explicit_ray_slots_expected_fail_2026-05-28.txt
targeted_pass=test_results/native_container_logs/cargo_test_candidate_slider_explicit_ray_slots_targeted_2026-05-28.txt
package_pass=test_results/native_container_logs/cargo_test_candidate_slider_explicit_ray_slots_package_2026-05-28.txt
```

## Verification

```text
cd /work/native && cargo fmt --all -- --check = passed
cd /work/native && cargo clippy --workspace --all-targets -- -D warnings = passed
cd /work/native && cargo test --workspace = passed, 57 native engine tests plus dashboard test
python -m pytest -p no:cacheprovider -q = passed, 112 tests
python -m pytest -p no:cacheprovider -W error -q = passed, 112 tests
```
