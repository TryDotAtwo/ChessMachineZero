# Percepta Frozen Attention Trace VM V4 Test Results

## Scope

- task_id=percepta_frozen_attention_trace_vm_v4
- runtime_model=PerceptaFrozenAttentionRuleComputer
- runtime_session=PerceptaParametricSelfPlaySession
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
- compiled_isa_instruction_count=14
- compiled_microprogram_step_count=18
- compiled_attention_layer_count=26
- compiled_parameter_count=1775296
- strategy_training=false
- strategy_module=none

## TDD Record

- initial_command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider tests\test_percepta_frozen_attention_vm.py tests\test_dashboard.py`
- initial_result=failed as expected
- initial_failures=missing `chess_machine_zero.model.percepta_attention_rule_kernels`

## Pytest Verification

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider tests\test_percepta_frozen_attention_vm.py tests\test_dashboard.py`
- result=`16 passed in 126.17s (0:02:06)`

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider`
- result=`133 passed in 230.71s (0:03:50)`

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider -W error`
- result=`133 passed in 281.27s (0:04:41)`

## Boundary Scans

- command=`rg "^(import chess|from chess import|from chess\.)" src tests -n`
- result=`src\chess_machine_zero\chess\rules_oracle.py:5:import chess`

- command=`rg -i "fallback|smoke" src tests -n`
- result=no matches

- command=`rg "DenseHardmax2D|nn\.Linear|PerceptaE2ETraceDecoder|prompt_fingerprints|continuation_tokens|rules_oracle|ChessMachineVM|AnalyticRuleCompiler" src\chess_machine_zero\model\percepta_frozen_attention_vm.py src\chess_machine_zero\model\percepta_rule_layer_graph.py src\chess_machine_zero\model\percepta_attention_rule_kernels.py -n`
- result=no matches

## Tensor-Kernel Verification

- test=`test_frozen_attention_rule_primitives_are_lowered_to_tensor_kernels`
- invariant=`FrozenAttentionTensorRuleKernels` backs `FrozenAttentionRuleLayerGraph`
- checked_methods=piece_dispatch,ray_scan,attack_test,legal_filter,make_move,terminal_predicates,legal_candidate_tensors
- forbidden_ast_nodes=For,AsyncFor,While,If,IfExp,Match,Try
- result=no forbidden AST nodes in checked primitive kernel methods
- legal_trace_check=starting position and deterministic arbitrary oracle-walk positions matched oracle legal UCI sets exactly
- make_move_check=castling and e2e4 board writes reconstructed oracle boards exactly

## Runtime Verification

- server_url=http://127.0.0.1:8768
- server_pid=59772
- API_snapshot_engine=PerceptaFrozenAttentionRuleComputer
- API_snapshot_primitive_kernel_execution_mode=pure_frozen_attention_tensor_layers
- API_snapshot_python_control_flow_rule_primitives=false
- API_snapshot_tensor_kernel_count=6
- API_snapshot_compiled_parameter_count=1775296
- API_snapshot_legal_count=20

## Known Limit

- v4_done=rule primitive decisions run through frozen tensor kernels without Python control-flow inside checked primitive kernel methods
- remaining_host_boundary=trace packet serialization, UCI mapping, and BoardState conversion remain Python I/O formatting around tensor-kernel outputs
