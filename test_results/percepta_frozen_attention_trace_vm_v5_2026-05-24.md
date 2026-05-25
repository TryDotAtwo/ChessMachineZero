# Percepta Frozen Attention Trace VM V5 Test Results

## Scope

- task_id=percepta_frozen_attention_trace_vm_v5
- runtime_model=PerceptaFrozenAttentionRuleComputer
- runtime_session=PerceptaParametricSelfPlaySession
- core_trace_runtime=tensor_trace_in_frozen_attention_blocks_tensor_trace_out
- python_host_boundary_role=display_only
- tensor_trace_core_runtime=true
- tracepacket_core_runtime=false
- rule_execution_mode=percepta_frozen_attention_trace_vm
- rule_core_execution_mode=executable_frozen_attention_layer_graph
- primitive_kernel_execution_mode=pure_frozen_attention_tensor_layers
- python_rule_primitive_runtime=false
- python_control_flow_rule_primitives=false
- compiled_rule_primitives=PIECE_DISPATCH,RAY_SCAN,ATTACK_TEST,LEGAL_FILTER,MAKE_MOVE,TERMINAL_PREDICATES
- tensor_kernel_count=6
- attention_backend=logarithmic_2d_attention
- lookup_complexity=O(log n)
- host_append_only=true
- token_streaming=true
- uses_mlp=false
- uses_dense_scan=false
- compiled_layer_graph_serialized=true
- position_lookup=false
- finite_prompt_lookup=false
- compiled_prompt_count=0
- compiled_parameter_count=1775338

## TDD Record

- initial_command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider tests\test_percepta_frozen_attention_vm.py tests\test_dashboard.py`
- initial_result=failed as expected
- initial_failures=missing `chess_machine_zero.model.percepta_tensor_trace_runtime`

## Pytest Verification

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider tests\test_percepta_frozen_attention_vm.py tests\test_dashboard.py`
- result=`17 passed in 189.00s (0:03:08)`

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider -x -vv`
- result=`134 passed in 310.28s (0:05:10)`

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider -W error`
- result=`134 passed in 216.21s (0:03:36)`

## Boundary Scans

- command=`rg "^(import chess|from chess import|from chess\.)" src tests -n`
- result=`src\chess_machine_zero\chess\rules_oracle.py:5:import chess`

- command=`rg -i "fallback|smoke" src tests -n`
- result=no matches

- command=`rg "\bBoardState\b|\bTracePacket\b" src\chess_machine_zero\model\percepta_tensor_trace_runtime.py -n`
- result=no matches

## Tensor Trace Verification

- test=`test_tensor_trace_core_runtime_does_not_call_packet_or_boardstate_graph`
- invariant=monkeypatch `FrozenAttentionRuleLayerGraph` packet/BoardState runtime methods to raise `AssertionError`
- result=tensor legal decode and tensor make-move decode passed without graph packet/BoardState runtime calls
- legal_trace_type=torch.Tensor
- legal_trace_shape=[N,7]
- make_trace_type=torch.Tensor
- make_trace_shape=[N,7]
- legal_trace_check=starting position legal UCI set matched oracle exactly after display-boundary conversion
- make_move_check=e2e4 board writes reconstructed oracle board exactly after display-boundary conversion

## Runtime Verification

- server_url=http://127.0.0.1:8768
- server_pid=46428
- API_snapshot_engine=PerceptaFrozenAttentionRuleComputer
- API_snapshot_core_trace_runtime=tensor_trace_in_frozen_attention_blocks_tensor_trace_out
- API_snapshot_python_host_boundary_role=display_only
- API_snapshot_tensor_trace_core_runtime=true
- API_snapshot_tracepacket_core_runtime=false
- API_snapshot_compiled_parameter_count=1775338
- API_snapshot_legal_count=20

## Known Limit

- v5_done=core legal/make trace runtime is tensor trace input to frozen tensor blocks to tensor trace output
- remaining_host_boundary=human UCI parsing, dashboard JSON, and visual board formatting remain Python display/API code
