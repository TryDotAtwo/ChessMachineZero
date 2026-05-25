# Percepta Frozen Attention Trace VM V3 Test Results

## Scope

- task_id=percepta_frozen_attention_trace_vm_v3
- runtime_model=PerceptaFrozenAttentionRuleComputer
- runtime_session=PerceptaParametricSelfPlaySession
- rule_execution_mode=percepta_frozen_attention_trace_vm
- rule_core_execution_mode=executable_frozen_attention_layer_graph
- python_rule_primitive_runtime=false
- compiled_rule_primitives=PIECE_DISPATCH,RAY_SCAN,ATTACK_TEST,LEGAL_FILTER,MAKE_MOVE,TERMINAL_PREDICATES
- compiled_rule_primitive_count=6
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
- compiled_parameter_count=13642
- strategy_training=false
- strategy_module=none

## TDD Record

- initial_command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider tests\test_percepta_frozen_attention_vm.py tests\test_dashboard.py`
- initial_result=failed as expected
- initial_failures=missing `rule_core_execution_mode`; runtime still called inherited `legal_move_trace_from_prompt`; dashboard snapshot missing graph primitive metadata

## Pytest Verification

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider tests\test_percepta_frozen_attention_vm.py tests\test_dashboard.py`
- result=`15 passed in 66.88s (0:01:06)`

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider`
- result=`132 passed in 129.45s (0:02:09)`

- command=`$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider -W error`
- result=`132 passed in 123.09s (0:02:03)`

## Boundary Scans

- command=`rg "^(import chess|from chess import|from chess\.)" src tests -n`
- result=`src\chess_machine_zero\chess\rules_oracle.py:5:import chess`

- command=`rg -i "fallback|smoke" src tests -n`
- result=no matches

- command=`rg "DenseHardmax2D|nn\.Linear|PerceptaE2ETraceDecoder|prompt_fingerprints|continuation_tokens|rules_oracle|ChessMachineVM|AnalyticRuleCompiler" src\chess_machine_zero\model\percepta_frozen_attention_vm.py src\chess_machine_zero\model\percepta_rule_layer_graph.py -n`
- result=no matches

## Rule-Graph Verification

- test=`test_frozen_attention_runtime_does_not_call_inherited_python_rule_primitives`
- invariant=monkeypatch inherited `WeightCompiledRulesTransformer` primitive methods to raise `AssertionError`
- result=decode legal trace and make-move trace passed without inherited primitive method calls
- legal_trace_check=starting position legal UCI set matched oracle exactly
- make_move_check=e2e4 board writes reconstructed oracle board exactly
- graph_execution_counts_positive=PIECE_DISPATCH,RAY_SCAN,ATTACK_TEST,LEGAL_FILTER,MAKE_MOVE,TERMINAL_PREDICATES
- dense_attention_instance_check=`hasattr(rules, "_hardmax") == False`

## Runtime Verification

- server_url=http://127.0.0.1:8768
- server_pid=49036
- API_snapshot_engine=PerceptaFrozenAttentionRuleComputer
- API_snapshot_rule_core_execution_mode=executable_frozen_attention_layer_graph
- API_snapshot_python_rule_primitive_runtime=false
- API_snapshot_compiled_rule_primitives=PIECE_DISPATCH,RAY_SCAN,ATTACK_TEST,LEGAL_FILTER,MAKE_MOVE,TERMINAL_PREDICATES
- API_snapshot_compiled_parameter_count=13642
- API_snapshot_legal_count=20

## Known Limit

- v3_done=explicit executable rule layer graph exists, inherited primitive rule runtime calls are blocked by tests, graph metadata is exposed by runtime and dashboard
- next_required=lower each graph primitive kernel from host-language tensor/control operations into pure tensor-attention kernels if stricter Percepta hardware-level execution is required
