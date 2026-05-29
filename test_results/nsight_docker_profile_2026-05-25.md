# Docker Nsight Profile 2026-05-25

## Scope

- task=Investigate slow Percepta frozen-attention dashboard self-play step using local Docker image with NVIDIA Nsight tools.
- docker_image=gpu-dev-cutlass-nsight:2026-05-24
- workspace_mount=/work
- command_family=docker run --rm --gpus all
- runtime_entry=CMZDashboardSession.step_transformer()
- deterministic_seed=20260524
- behavior_change=false

## Docker GPU Environment

- torch=2.6.0+cu124
- torch_cuda_available=true
- cuda_device=NVIDIA GeForce RTX 3070 Laptop GPU
- nsight_systems=/usr/local/bin/nsys
- nsight_systems_version=2024.1.1.0
- nsight_compute=/usr/local/cuda/bin/ncu
- nsight_compute_version=2024.1.1.0
- python_chess_installed=false
- python_chess_required_for_profile=false

## Timing Results

- full_legal_once_seconds=0.079215
- host_append_legal_seconds=1.294018
- selfplay_step_seconds=1.775451
- cprofile_step_seconds=1.696030
- emitted_move=a2a3
- emitted_trace_packets=117
- full_legal_rows=117
- host_legal_rows=117

## Device Placement

- rule_param_device=cpu
- prompt_device=cpu
- nsight_cuda_kernel_data=false
- nsight_gpu_memory_data=false
- conclusion=Current profile does not measure slow CUDA kernels because the current rule path remains on CPU tensors.

## cProfile Hotspots

- execute_trace_calls=50
- execute_trace_cumtime_seconds=1.618
- candidate_table_calls=52
- candidate_table_cumtime_seconds=1.579
- decode_legal_tensor_trace_host_append_only_cumtime_seconds=1.325
- decode_next_legal_tensor_packet_calls=41
- lookup_rows_calls=1126
- lookup_rows_cumtime_seconds=1.238
- board_read_calls=394
- board_read_cumtime_seconds=0.923
- attention_select_calls=1444
- attention_select_cumtime_seconds=0.798
- torch_one_hot_calls=2676
- torch_one_hot_cumtime_seconds=0.523
- tensor_to_calls=12436
- tensor_to_cumtime_seconds=0.256
- torch_argmax_calls=1505
- torch_argmax_cumtime_seconds=0.162

## Diagnosis

- primary_bottleneck=host_append_loop_recomputes_full_legal_trace_for_each_emitted_token
- secondary_bottleneck=matrix_attention_interpreter_uses_many_small_cpu_torch_ops
- cuda_bottleneck=false
- cutlass_kernel_bottleneck=false
- first_required_fix=make_decode_next_incremental_or_cache_compiled_trace_state_so_each_next_token_does_not_recompute_full_legal_continuation
- second_required_fix=move_frozen_attention_weights_and_prompt_trace_to_cuda_with_explicit_device_ownership
- third_required_fix=fuse_QK_mask_hardmax_select_V_residual_write_and_candidate_table_ops_into_batched_cuda_or_cutlass_kernels

## Stored Artifacts

- cprofile_log=test_results/docker_cprofile_selfplay_step_2026-05-25.txt
- nsight_stats_log=test_results/nsight_stats_selfplay_step_2026-05-25.txt
- raw_nsys_report=test_results/nsight_selfplay_step_2026_05_25.nsys-rep
- raw_nsys_sqlite=test_results/nsight_selfplay_step_2026_05_25.sqlite
- raw_artifact_policy=raw Nsight report and SQLite are ignored by git because they are large generated profiler artifacts.
