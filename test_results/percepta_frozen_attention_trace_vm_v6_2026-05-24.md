# Percepta Frozen Attention Trace VM v6

- date=2026-05-24
- milestone=percepta_frozen_attention_trace_vm_v6
- summary=Tensor trace execution now routes through `FrozenTransformerAttentionBlockStack`; `FrozenAttentionTensorTraceRuntime` no longer calls `FrozenAttentionTensorRuleKernels` primitive shortcut methods.
- core_rule_compute_backend=frozen_transformer_attention_block_stack
- tensor_kernel_shortcut_runtime=false
- compiled_attention_block_stack=true
- compiled_attention_block_count=6
- compiled_attention_head_count=11
- residual_trace_write_count=3
- core_trace_runtime=tensor_trace_in_frozen_attention_blocks_tensor_trace_out
- dashboard_url=http://127.0.0.1:8768
- dashboard_server_pid=50052

## TDD Check

- initial_targeted_result=expected_failure
- failure_reason=missing `chess_machine_zero.model.percepta_attention_block_stack`
- tests_added=block-stack type assertions; dashboard engine metadata assertions; monkeypatch check that forbids `FrozenAttentionTensorRuleKernels` primitive shortcut calls during tensor decode

## Verification

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -p no:cacheprovider tests\test_percepta_frozen_attention_vm.py tests\test_dashboard.py
```

- result=18 passed in 221.40s

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -p no:cacheprovider -x -vv
```

- result=135 passed in 217.24s

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -p no:cacheprovider -W error
```

- result=135 passed in 406.98s

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
rg "DenseHardmax2D|nn\.Linear|PerceptaE2ETraceDecoder|prompt_fingerprints|continuation_tokens|rules_oracle|ChessMachineVM|AnalyticRuleCompiler" src\chess_machine_zero\model\percepta_frozen_attention_vm.py src\chess_machine_zero\model\percepta_attention_block_stack.py src\chess_machine_zero\model\percepta_tensor_trace_runtime.py -n
```

- result=no matches

## Dashboard Check

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8768/api/snapshot
```

- status_code=200
- engine.core_rule_compute_backend=frozen_transformer_attention_block_stack
- engine.compiled_attention_block_count=6
- legal_count=20
