# Percepta Policy-Only Full Code Audit Result Log, 2026-05-28

- result_id=percepta_policy_only_full_code_audit_2026-05-28
- report=docs/percepta_policy_only_full_code_audit_2026-05-28.md
- scope=static audit of source/config/docs/tests for Percepta policy-only architecture conformance
- commands_run=rg file inventory; rg semantic searches for `full_frozen_attention_only`, `HullKVCache`, `NestedHullTopK2D`, `hardmax`, `softmax`, `baseline`, `ranker`, `negamax`, `python-chess`, `fallback_allowed`, `head_dim`
- runtime_tests=full_pytest_timeout_after_visible_100_percent; targeted_audit_relevant_pytest_passed
- preliminary_verdict=not_fully_conformant
- critical_findings=contract_overclaim_full_frozen_attention_only; custom_CUDA_rule_control_flow; legacy_python_ranker_baseline_negamax; dashboard_not_native_policy_decoder; HullKV_NestedHull_scaffold_not_integrated_hot_path

## Evidence Commands

```powershell
rg --files -g '!test_results/**' -g '!*.png' -g '!*.pyc' -g '!*.pyo' -g '!__pycache__/**' -g '!.pytest_cache/**' -g '!.playwright-mcp/**' -g '!.git/**'
rg -n "full_frozen_attention_only|executor_head_dim|rule_attention_backend|HullKVCache|python_hot_path|fallback_allowed|decoder_attention|actor_critic|critic_head_enabled|value_head_enabled|frozen_rule_attention_graph|latest_write_hardmax_2d|cuda_qk|cutlass|NestedHullTopK2D|decoder_hidden_2d|torch::softmax|decoder_policy_logits|policy_gradient|legal_trace_begin|legal_trace_next|sorted_pseudo_legal_moves|resolve_legal_move|frozen_candidate_move_from_request|insufficient_material\(|frozen_legal_trace_attention_tokens|frozen_make_move_trace_attention_tokens|frozen_board_transition_emit_attention_layer|pseudo_legal_moves" native/cpp/src/cmz_engine.cpp
rg -n "cmz_candidate_target_mask_value|cmz_candidate_record_emit_attention_kernel|cmz_candidate_move_is_legal|cmz_any_legal_candidate_move|cmz_insufficient_material_value|cmz_resolve_move_attention_kernel|cmz_qk2_hardmax_select_u32|cmz_cutlass_hardmax2d_values|cmz_hardmax_float_select_kernel|for \(|while \(|if \(" native/cpp/src/cmz_cuda_kernels.cu
rg -n "def step|def play_human_move|def legal_moves|chosen|legal_trace|LEGAL_SET|verification|transformer|last_emitter|snapshot|active_transformer|policy|decoder|rank" src/chess_machine_zero/model/percepta_parametric_selfplay.py src/chess_machine_zero/dashboard/static/dashboard.js src/chess_machine_zero/dashboard/static/index.html src/chess_machine_zero/dashboard/state.py
rg -n "CMZOutcomeBaseline|trace_negamax|depth == 0|_baseline_value|max\(child_scores\)|SCORE_SET|ranker\.score_moves|torch\.softmax|scores\.argmax|CMZMoveRanker|baseline_loss|train_ranker_baseline_step" src/chess_machine_zero/vm/lookahead.py src/chess_machine_zero/selfplay/actor.py src/chess_machine_zero/train/losses.py src/chess_machine_zero/model/baseline.py src/chess_machine_zero/model/ranker.py src/chess_machine_zero/vm/decision_program.py tests/test_trace_lookahead.py tests/test_selfplay_training_step.py tests/test_select_move_trace.py
rg -n "import chess|rules_oracle|python-chess|oracle" src tests docs/cmz_percepta_policy_only_architecture_review.md
```

## Verification To Append

- command=`python -m pytest -p no:cacheprovider -q`
- status=timeout_exit_code_124
- observed_output=progress_dots_reached_100_percent_with_146_test_positions_visible; process_did_not_return_clean_exit_before_tool_timeout
- command=`python -m pytest -p no:cacheprovider tests/test_hullkv_equivalence.py tests/test_dense_hardmax_2d.py tests/test_dashboard.py tests/test_percepta_frozen_attention_vm.py -q`
- status=passed
- observed_output=`31 passed`
