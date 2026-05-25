# Percepta Frozen Attention Trace VM v7

- date=2026-05-24
- milestone=percepta_frozen_attention_trace_vm_v7
- summary=Added formal ChessRuleISA microprogram compiler and frozen attention program weights; runtime stack now owns a compiled program plus unified executor.
- percepta_compiler_pipeline=chess_isa_microprogram_to_frozen_attention_weights
- rule_compiler_backend=rule_microprogram_to_frozen_attention_weights
- rule_microprogram_source=chess_rule_isa
- rule_microprogram_instruction_count=21
- compiled_rule_program_weight_count=408
- unified_rule_executor_runtime=true
- handwritten_stack_primitive_runtime=false
- core_rule_compute_backend=frozen_transformer_attention_block_stack
- tensor_kernel_shortcut_runtime=false
- dashboard_url=http://127.0.0.1:8768
- dashboard_server_pid=25716

## TDD Check

- initial_targeted_result=expected_failure
- failure_reason=missing `chess_machine_zero.model.percepta_rule_compiler`
- tests_added=compiler frozen-weight assertions; VM compiler pipeline assertions; dashboard compiler state assertions; monkeypatch check that forbids stack primitive shortcut calls during tensor decode

## Verification

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -p no:cacheprovider tests\test_percepta_rule_compiler.py tests\test_percepta_frozen_attention_vm.py tests\test_dashboard.py
```

- result=21 passed in 212.22s

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -p no:cacheprovider -x -vv
```

- result=138 passed in 219.26s

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -p no:cacheprovider -W error
```

- result=138 passed in 182.30s

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
rg "rule_kernels\.(piece_dispatch|ray_scan|attack_test|legal_filter|make_move|terminal_predicates|legal_candidate_tensors)" src\chess_machine_zero\model\percepta_tensor_trace_runtime.py src\chess_machine_zero\model\percepta_frozen_attention_vm.py -n
```

- result=no matches

```powershell
rg "\bBoardState\b|\bTracePacket\b" src\chess_machine_zero\model\percepta_tensor_trace_runtime.py -n
```

- result=no matches

```powershell
rg "DenseHardmax2D|nn\.Linear|PerceptaE2ETraceDecoder|prompt_fingerprints|continuation_tokens|rules_oracle|ChessMachineVM|AnalyticRuleCompiler" src\chess_machine_zero\model\percepta_frozen_attention_vm.py src\chess_machine_zero\model\percepta_attention_block_stack.py src\chess_machine_zero\model\percepta_tensor_trace_runtime.py src\chess_machine_zero\model\percepta_rule_compiler.py -n
```

- result=no matches

## Dashboard Check

```powershell
$snapshot = Invoke-RestMethod http://127.0.0.1:8768/api/snapshot
```

- engine.percepta_compiler_pipeline=chess_isa_microprogram_to_frozen_attention_weights
- engine.rule_compiler_backend=rule_microprogram_to_frozen_attention_weights
- engine.unified_rule_executor_runtime=true
- engine.rule_microprogram_instruction_count=21
- legal_count=20
