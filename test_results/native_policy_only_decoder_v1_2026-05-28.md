# native_policy_only_decoder_v1_2026-05-28

- status=passed
- goal=remove required actor-critic/value-baseline scaffold from native trainable decoder
- expected_fail_log=test_results/native_container_logs/cargo_test_policy_only_decoder_expected_fail_2026-05-28.txt
- targeted_contract_log=test_results/native_container_logs/cargo_test_policy_only_decoder_targeted_2026-05-28.txt
- targeted_forward_log=test_results/native_container_logs/cargo_test_policy_only_decoder_forward_2026-05-28.txt
- rustfmt_apply_log=test_results/native_container_logs/cargo_fmt_apply_policy_only_decoder_2026-05-28.txt
- rustfmt_check_log=test_results/native_container_logs/cargo_fmt_policy_only_decoder_2026-05-28.txt
- clippy_log=test_results/native_container_logs/cargo_clippy_policy_only_decoder_2026-05-28.txt
- cargo_workspace_log=test_results/native_container_logs/cargo_test_policy_only_decoder_2026-05-28.txt
- pytest_log=test_results/policy_only_decoder_pytest_2026-05-28.txt
- pytest_werror_log=test_results/policy_only_decoder_pytest_werror_2026-05-28.txt

## Implementation

- decoder_backend=libtorch_cuda_policy_only_v1
- learning_method=self_play_policy_gradient
- actor_critic=false
- critic_head_enabled=false
- value_head_enabled=false
- externally_prescribed_critic=false
- c_api_change=cmz_engine_decoder_forward returns command logits only
- removed=decoder_value_weight,decoder_value_bias,value_baseline_output,value_loss
- retained=2D attention hidden computation, generic command logits, TracePacket detached path, no labels, no handcrafted eval

## Verification

- `cd /work/native && cargo fmt --all -- --check` => passed
- `cd /work/native && cargo clippy --workspace --all-targets -- -D warnings` => passed
- `cd /work/native && cargo test --workspace` => passed; workspace_tests=48
- `python -m pytest -p no:cacheprovider -q` => passed; tests=146
- `python -m pytest -p no:cacheprovider -W error -q` => passed; tests=146
