# Percepta Incremental Decode v10 Test Results 2026-05-26

## Scope

- change_id=percepta_incremental_decode_token_ui_v10
- behavior_change=PerceptaFrozenAttentionRuleComputer caches compiled legal/make continuation tensors per active decode context while preserving host-append token streaming and corrupted-prefix rejection.
- dashboard_change=Trace VM panel now exposes `Readable log` and `Token trace` tabs, hexadecimal token rows, decoded packet annotations, and token/packet counters.
- article_notes=docs/percepta_can_llms_be_computers.md

## TDD Evidence

- initial_targeted_tests_before_implementation=3 failed
- failure_1=missing legal_continuation_compute_count
- failure_2=make-move continuation cache assertion not yet implemented
- failure_3=dashboard lacked Readable log/Token trace tabs
- post_implementation_new_tests=`python -m pytest -p no:cacheprovider tests\test_percepta_frozen_attention_vm.py::test_tensor_host_append_legal_decode_reuses_one_compiled_continuation_per_prompt tests\test_percepta_frozen_attention_vm.py::test_tensor_host_append_make_move_decode_reuses_one_compiled_continuation_per_move_context tests\test_dashboard.py::test_dashboard_frontend_exposes_percepta_style_readable_and_token_trace_views -q` => 3 passed

## Regression Tests

- targeted_suite=`python -m pytest -p no:cacheprovider tests\test_percepta_rule_compiler.py tests\test_percepta_frozen_attention_vm.py tests\test_dashboard.py -q` => 27 passed
- full_pytest=`python -m pytest -p no:cacheprovider -q` => 144 passed
- full_warning_check=`python -m pytest -p no:cacheprovider -W error -q` => 144 passed
- collected_tests=144

## Boundary Scans

- python_chess_direct_import_scan=`rg -n "^(import chess|from chess import|import chess\.|from chess\.)" src tests` => only src/chess_machine_zero/chess/rules_oracle.py
- fallback_smoke_scan=`rg -n "fallback|smoke" src tests` => no matches
- runtime_oracle_scan=`rg -n "rules_oracle" src/chess_machine_zero/dashboard src/chess_machine_zero/model/percepta_frozen_attention_vm.py src/chess_machine_zero/model/percepta_parametric_selfplay.py src/chess_machine_zero/model/percepta_matrix_attention_runtime.py src/chess_machine_zero/model/percepta_attention_block_stack.py src/chess_machine_zero/model/percepta_tensor_trace_runtime.py` => no matches

## Local Performance

- environment=Windows host Python/Torch runtime
- full_legal_once_seconds=0.048632
- host_append_legal_seconds=0.043370
- host_append_legal_compute_count=1
- selfplay_step_seconds=0.158801
- move=a2a3
- trace_packets=117
- result_log=test_results/perf_incremental_decode_local_2026-05-26.txt

## Docker Performance

- docker_image=gpu-dev-cutlass-nsight:2026-05-24
- torch=2.6.0+cu124
- cuda_available=true
- device0=NVIDIA GeForce RTX 3070 Laptop GPU
- rule_param_device=cpu
- prompt_device=cpu
- full_legal_once_seconds=0.073621
- host_append_legal_seconds=0.039792
- host_append_legal_compute_count=1
- selfplay_step_seconds=0.143764
- docker_prior_host_append_legal_seconds=1.294018
- docker_prior_selfplay_step_seconds=1.775451
- docker_host_append_speedup=32.52x
- docker_step_speedup=12.35x
- result_log=test_results/docker_perf_incremental_decode_2026-05-26.txt
- cprofile_log=test_results/docker_cprofile_incremental_decode_2026-05-26.txt
- nsight_stats_log=test_results/nsight_incremental_stats_2026-05-26.txt

## Post-Change cProfile

- cprofile_step_seconds=0.136517
- prior_cprofile_step_seconds=1.696030
- cprofile_step_speedup=12.42x
- function_calls_before=98492
- function_calls_after=17512
- execute_trace_calls_before=50
- execute_trace_calls_after=2
- torch_one_hot_calls_before=2676
- torch_one_hot_calls_after=228
- tensor_to_calls_before=12436
- tensor_to_calls_after=1223

## Browser Verification

- browser_url=http://127.0.0.1:8769
- browser_check=64 squares rendered; Readable log tab visible; Token trace tab visible; token tab switch worked; hex token rows visible; decoded annotations visible; token meter visible; console warnings/errors=0
- screenshot=.playwright-mcp/cmz-token-trace-v10.png

## Remaining Performance Limit

- cuda_kernel_runtime=false
- current_rule_param_device=cpu
- current_prompt_device=cpu
- nsight_cuda_kernel_data=false
- nsight_gpu_memory_data=false
- next_required_fix=explicit CUDA device ownership plus fused CUDA/CUTLASS matrix-attention kernels
