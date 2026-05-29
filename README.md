# ChessMachineZero

ChessMachineZero is a trace-based chess machine prototype targeting a
Percepta-style policy-only architecture: chess rules emit and verify
TracePacket/MovePacket streams, while a trainable decoder policy consumes the
same trace language and learns move selection without external tree search,
human-game labels, engine labels, tablebase labels, or handcrafted evaluation.

## Current Runtime

- Native workspace: `native/`
- Native orchestration: Rust crates under `native/crates/`
- Native engine: C++/CUDA implementation under `native/cpp/`
- Native dashboard: `native/crates/cmz-dashboard`
- Native CLI: `native/crates/cmz-cli`
- Python production dashboard runtime: removed
- Python/PyTorch Percepta attention runtime modules: removed from production source
- Legacy Python strategy modules: removed from production source
- Python remaining role: packet codecs, tests, docs, and `python-chess` oracle wrapper
- `python-chess` direct import boundary: `src/chess_machine_zero/chess/rules_oracle.py`

## Native Contract

- `python_hot_path=false`
- `fallback_allowed=false`
- `decoder_backend=libtorch_cuda_policy_only_v1`
- `dashboard_policy_decoder=true`
- `dashboard_policy_selection_backend=native_libtorch_policy_decoder`
- `actor_critic=false`
- `value_head_enabled=false`
- `target_full_frozen_attention_only=true`
- `full_frozen_attention_only=false`
- `semantic_attention_purity=false`
- `remaining_non_attention_paths=terminal_check_state_king_scan,castle_target_chess_control_flow,legal_filter_batch_attack_chess_control_flow,legal_filter_batch_ray_scan_control_flow`

The contract intentionally keeps `full_frozen_attention_only=false` until every
remaining CUDA chess-control-flow path is lowered to explicit frozen 2D
QK-hardmax/select/write layers.

## Implemented Native Slices

- Native MovePacket and TracePacket codecs
- Native legal trace streaming
- Native make-move trace emission
- CUDA board projection
- CUDA trace-select attention
- CUTLASS/HullKV hardmax integration for trace packet lookup
- NestedHullTopK GPU/CUTLASS top-k path
- Native policy-only decoder scaffold and policy-gradient update path
- Native dashboard using policy decoder selection
- Attack-mask, ray-scan, candidate-target, legal-filter, resolve-move, candidate-record, pawn, slider, terminal-legal-presence, and terminal-material lowering slices
- Source-audit tests that keep remaining semantic gaps visible

## Run Python Verification

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -p no:cacheprovider -W error
```

## Run Native Verification In Docker

```powershell
powershell -ExecutionPolicy Bypass -File .\docker\native\run_native_container.ps1
powershell -ExecutionPolicy Bypass -File .\docker\native\exec_native.ps1 -Command "cd /work/native && cargo fmt --all -- --check" -Log "test_results/native_container_logs/cargo_fmt_manual.txt"
powershell -ExecutionPolicy Bypass -File .\docker\native\exec_native.ps1 -Command "cd /work/native && cargo clippy --workspace --all-targets -- -D warnings" -Log "test_results/native_container_logs/cargo_clippy_manual.txt"
powershell -ExecutionPolicy Bypass -File .\docker\native\exec_native.ps1 -Command "cd /work/native && cargo test --workspace" -Log "test_results/native_container_logs/cargo_test_workspace_manual.txt"
```

## Run Native Dashboard

```powershell
powershell -ExecutionPolicy Bypass -File .\docker\native\run_native_container.ps1
powershell -ExecutionPolicy Bypass -File .\docker\native\start_dashboard.ps1 -Port 8768
docker logs -f cmz-native-dev
```

Dashboard URL from the Windows browser:

```text
http://127.0.0.1:8768
```

## Important Documents

- Architecture source: `docs/chess_machine_zero_percepta_architecture.md`
- Native Rust/C++/CUDA architecture: `docs/native_rust_cpp_cuda_architecture.md`
- Policy-only architecture review handoff: `docs/cmz_percepta_policy_only_architecture_review.md`
- Full code audit: `docs/percepta_policy_only_full_code_audit_2026-05-28.md`
- Project memory: `docs/project_memory.md`
- Change history: `docs/change_history.md`
- Prompt history: `docs/prompt_history.md`
- Test logs and evidence: `test_results/`
