# Percepta Two-Transformer Dashboard v9 Test Record

## Scope

- task_id=percepta_two_transformer_dashboard_v9
- runtime=two independent `PerceptaFrozenAttentionRuleComputer` instances
- white_actor=`transformer_white`
- black_actor=`transformer_black`
- legal_verification=selected committed move must appear in trace-emitted `LEGAL_SET`
- token_observability=history summaries and `transformer_token_streams` expose emitted trace packet tokens per transformer
- oracle_boundary=python-chess used by tests only through `src/chess_machine_zero/chess/rules_oracle.py`

## Test-First Failure

```text
python -m pytest -p no:cacheprovider tests\test_dashboard.py -q
FFF...F..
4 failed, 5 passed
failure_reasons=missing transformers snapshot fields; actor still "transformer"; emitted token stream fields missing
```

## Passing Verification

```text
python -m pytest -p no:cacheprovider tests\test_dashboard.py -q
9 passed
```

```text
python -m pytest -p no:cacheprovider tests\test_percepta_rule_compiler.py tests\test_percepta_frozen_attention_vm.py tests\test_dashboard.py -q
23 passed
```

```text
python -m pytest -p no:cacheprovider -x -vv
140 passed in 511.78s
```

```text
python -m pytest -p no:cacheprovider -W error
140 passed in 617.98s
```

## API Runtime Check

```json
{"initial_active":"transformer_white","initial_legal":20,"ply_after":2,"last_actor":"transformer_black","illegal_commits":0,"white_tokens":117,"black_tokens":117,"verified":true}
```

## Browser Check

- browser_validation=Playwright MCP
- url=http://127.0.0.1:8768
- server_pid=56360
- desktop=1440x960
- mobile=390x840
- page_title=`ChessMachineZero Dashboard`
- square_count=64
- interaction=Step button advanced dashboard to ply=3
- rendered_token_log_prefix=`transformer_white token[1]=[1,0,4,0,0,1,0]`
- rendered_history=`transformer_white w a2a3 tokens=117 legal=true`; `transformer_black b c7c6 tokens=117 legal=true`; `transformer_white w d2d4 tokens=115 legal=true`
- console_errors=0
- console_warnings=0
- mobile_horizontal_overflow=false

## Boundary Scans

```text
rg "^(import chess|from chess import|from chess\.)" src tests -n
src\chess_machine_zero\chess\rules_oracle.py:5:import chess
```

```text
rg -i "fallback|smoke" src tests -n
no_matches
```

```text
rg -n "rules_oracle|python-chess" src\chess_machine_zero\dashboard src\chess_machine_zero\model\percepta_parametric_selfplay.py src\chess_machine_zero\model\percepta_frozen_attention_vm.py src\chess_machine_zero\model\percepta_attention_block_stack.py src\chess_machine_zero\model\percepta_matrix_attention_runtime.py
no_matches
```

```text
rg -n "compiled_executor|DenseHardmax2D|nn\.Linear|PerceptaE2ETraceDecoder|prompt_fingerprints|continuation_tokens|ChessMachineVM|AnalyticRuleCompiler" src\chess_machine_zero\model\percepta_frozen_attention_vm.py src\chess_machine_zero\model\percepta_attention_block_stack.py src\chess_machine_zero\model\percepta_matrix_attention_runtime.py src\chess_machine_zero\model\percepta_tensor_trace_runtime.py
no_matches
```

## Visibility Fix v9.1

### Test-First Failure

```text
python -m pytest -p no:cacheprovider tests\test_dashboard.py::test_dashboard_frontend_autoplays_sequential_transformer_selfplay -q
1 failed
failure_reason=dashboard.js lacked state.busy/sequential autoplay contract
```

### Passing Verification

```text
python -m pytest -p no:cacheprovider tests\test_dashboard.py::test_dashboard_frontend_autoplays_sequential_transformer_selfplay -q
1 passed
```

```text
python -m pytest -p no:cacheprovider tests\test_dashboard.py -q
10 passed
```

```text
python -m pytest -p no:cacheprovider -W error tests\test_dashboard.py -q
10 passed
```

### Runtime Verification

- url=http://127.0.0.1:8768
- server_pid=52820
- script=/static/dashboard.js?v=9.1
- behavior=page load auto-started selfplay mode
- before={play_button=Pause, ply=0, status=`transformer_white computing move`, emitter=transformer_white}
- after={play_button=Pause, ply=1, status=`transformer turn`, emitter=transformer_white, verified=true}
- token_log_prefix=`transformer_white token[1]=[1,0,4,0,0,1,0]`
- console_errors=0
- console_warnings=0
