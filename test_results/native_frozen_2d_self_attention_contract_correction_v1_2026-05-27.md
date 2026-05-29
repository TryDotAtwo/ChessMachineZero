# Native Frozen 2D Self-Attention Contract Correction V1

- date=2026-05-27
- change_id=native_frozen_2d_self_attention_contract_correction_v1
- user_directive=Absolutely everything must be implemented through self-attention; the project is one transformer with frozen 2D self-attention rule layers.
- correction=previous graph metadata overclaimed full completion; current contract now separates target state from implementation state.

## Authoritative Target

- all_rules_must_lower_to_frozen_2d_self_attention=true
- tensor_layer_substrate=false
- monolithic_custom_cuda_rule_kernels_allowed=false
- legal_filter_v2_target=stack_of_frozen_2d_self_attention_layers
- legal_filter_v2_required_backend=cutlass_qk_scores_hardmax_v_write
- legal_filter_v2_required_layers=move_type_select,board_write_select,en_passant_capture_select,castle_rook_write_select,promotion_select,king_square_select,attack_source_select,ray_blocker_select,final_legal_select

## Current Truth State

- current_full_frozen_2d_self_attention_only=false
- full_frozen_attention_only=false
- full_rule_lowering_complete=false
- cpp_control_flow_rule_vm_remaining=true
- monolithic_custom_cuda_rule_kernels_remaining=true
- legal_filter_v1_monolithic_cuda_kernel_deprecated=true

## Changed Contract Fields

- graph_field=all_rules_must_lower_to_frozen_2d_self_attention; value=true
- graph_field=current_full_frozen_2d_self_attention_only; value=false
- graph_field=full_frozen_attention_only; value=false
- graph_field=full_rule_lowering_complete; value=false
- graph_field=cpp_control_flow_rule_vm_remaining; value=true
- graph_field=monolithic_custom_cuda_rule_kernels_allowed; value=false
- graph_field=monolithic_custom_cuda_rule_kernels_remaining; value=true
- graph_field=legal_filter_v1_monolithic_cuda_kernel_deprecated; value=true
- graph_field=legal_filter_v2_target; value=stack_of_frozen_2d_self_attention_layers

## Verification

- command=`cd /work/native && cargo test -p cmz-engine-sys native_frozen_rule_graph_declares_full_attention_only_contract -- --nocapture`
- log=test_results/native_container_logs/cargo_test_self_attention_contract_targeted_2026-05-27.txt
- result=passed; targeted_tests=1

- command=`cd /work/native && cargo fmt --all -- --check`
- log=test_results/native_container_logs/cargo_fmt_self_attention_contract_2026-05-27.txt
- result=passed

- command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`
- log=test_results/native_container_logs/cargo_clippy_self_attention_contract_2026-05-27.txt
- result=passed

- command=`cd /work/native && cargo test --workspace`
- log=test_results/native_container_logs/cargo_test_self_attention_contract_2026-05-27.txt
- result=passed; native_tests=47

- command=`py -m pytest -p no:cacheprovider -q`
- log=test_results/self_attention_contract_pytest_2026-05-27.txt
- result=passed; python_tests=146

- command=`py -m pytest -p no:cacheprovider -W error -q`
- log=test_results/self_attention_contract_pytest_werror_2026-05-27.txt
- result=passed; python_tests=146

## Next Implementation Step

- next_change=replace legal_filter v1 monolithic CUDA kernel with legal_filter_v2 layer stack
- first_layer_candidate=move_type_select_attention plus board_write_select_attention
- acceptance=arbitrary legal-filter cases still match current oracle tests while graph reports current_full_frozen_2d_self_attention_only=true only after monolithic rule kernels are removed
