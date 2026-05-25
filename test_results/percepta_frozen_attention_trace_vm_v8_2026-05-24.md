# Percepta Frozen Attention Trace VM v8

- date=2026-05-24
- milestone=percepta_frozen_attention_trace_vm_v8
- summary=Replaced low-level executor substrate with `FrozenMatrixAttentionInterpreter`.
- executor_substrate=matrix_attention_interpreter
- attention_step_operator=QK^T_mask_hardmax_select_V_residual_write
- matrix_attention_interpreter_runtime=true
- pytorch_domain_shortcut_runtime=false
- legacy_compiled_executor_runtime=false
- percepta_compiler_pipeline=chess_isa_microprogram_to_frozen_attention_weights
- rule_microprogram_instruction_count=21
- compiled_rule_program_weight_count=408
- compiled_rule_parameter_count=1775746
- dashboard_url=http://127.0.0.1:8768
- dashboard_server_pid=54428

## TDD Check

- initial_targeted_result=expected_failure
- failure_reason=missing `chess_machine_zero.model.percepta_matrix_attention_runtime`
- tests_added=matrix runtime metadata assertions; dashboard matrix runtime assertions; monkeypatch check that forbids legacy `_CompiledAttentionProgramExecutor` calls during legal/make decode

## Verification

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -p no:cacheprovider tests\test_percepta_rule_compiler.py tests\test_percepta_frozen_attention_vm.py tests\test_dashboard.py
```

- result=22 passed in 456.35s

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -p no:cacheprovider -x -vv
```

- result=139 passed in 516.55s

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -p no:cacheprovider -W error
```

- result=139 passed in 523.80s

## Boundary Scans

```powershell
rg "^(import chess|from chess import|from chess\.)" src tests -n
```

- result=`src\chess_machine_zero\chess\rules_oracle.py:5:import chess`

```powershell
rg -i "fallback|smoke" src tests -n
```

- result=no matches

```powershell
rg "rule_kernels\.(piece_dispatch|ray_scan|attack_test|legal_filter|make_move|terminal_predicates|legal_candidate_tensors)" src\chess_machine_zero\model\percepta_tensor_trace_runtime.py src\chess_machine_zero\model\percepta_frozen_attention_vm.py src\chess_machine_zero\model\percepta_attention_block_stack.py src\chess_machine_zero\model\percepta_matrix_attention_runtime.py -n
```

- result=no matches

```powershell
rg "\bBoardState\b|\bTracePacket\b" src\chess_machine_zero\model\percepta_tensor_trace_runtime.py -n
```

- result=no matches

```powershell
rg "DenseHardmax2D|nn\.Linear|PerceptaE2ETraceDecoder|prompt_fingerprints|continuation_tokens|rules_oracle|ChessMachineVM|AnalyticRuleCompiler" src\chess_machine_zero\model\percepta_frozen_attention_vm.py src\chess_machine_zero\model\percepta_attention_block_stack.py src\chess_machine_zero\model\percepta_tensor_trace_runtime.py src\chess_machine_zero\model\percepta_rule_compiler.py src\chess_machine_zero\model\percepta_matrix_attention_runtime.py -n
```

- result=no matches

```powershell
rg "compiled_executor" src\chess_machine_zero\model\percepta_frozen_attention_vm.py src\chess_machine_zero\model\percepta_attention_block_stack.py src\chess_machine_zero\model\percepta_tensor_trace_runtime.py src\chess_machine_zero\model\percepta_matrix_attention_runtime.py -n
```

- result=no matches

## Dashboard Check

```powershell
$snapshot = Invoke-RestMethod http://127.0.0.1:8768/api/snapshot
```

- engine.executor_substrate=matrix_attention_interpreter
- engine.attention_step_operator=QK^T_mask_hardmax_select_V_residual_write
- engine.matrix_attention_interpreter_runtime=true
- engine.matrix_attention_step_count=19346643
- legal_count=20
