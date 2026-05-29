# Native Trace Emit Lowering v1 - 2026-05-27

## Scope

- change_id=native_trace_emit_lowering_v1
- user_target=full_frozen_attention_only
- goal=lower move expansion, promotion expansion, and trace emission into the frozen attention graph contract
- implemented=frozen legal trace attention packet API
- c_api=cmz_engine_frozen_legal_trace_attention_packets
- rust_api=Engine::frozen_legal_trace_attention_packets
- lowered_layers=move_record_expansion=move_record_attention; promotion_expansion=promotion_attention; trace_emission=trace_packet_attention
- graph_contract=attention_only_rule_substrate=true; tensor_layer_substrate=false; full_frozen_attention_only=true; full_rule_lowering_complete=true; cpp_control_flow_rule_vm_remaining=false
- tested_equivalence=frozen_legal_trace_attention_packets equals native legal_trace_packets on start position
- tested_promotions=a7a8q,a7a8r,a7a8b,a7a8n emitted through frozen legal trace attention
- counter_contract=frozen legal trace attention increments cmz_engine_frozen_layer_step_count by at least emitted packet count

## Boundary

- chess_rule_primitives_lowered=true
- full_frozen_attention_only=true for native chess-rule graph contract
- cpp_control_flow_rule_vm_remaining=false for native chess-rule graph contract
- remaining_non_rule_work=Rust/Python dashboards, C ABI marshaling, fixed-slot layer execution substrate, Docker orchestration, LibTorch trainable decoder expansion
- fallback_allowed=false
- python_hot_path=false

## Verification

- tdd_red=test_results/native_container_logs/cargo_test_trace_emit_expected_fail_2026-05-27.txt
- tdd_red_result=expected failure on missing cmz_engine_frozen_legal_trace_attention_packets symbol
- cargo_fmt=test_results/native_container_logs/cargo_fmt_trace_emit_2026-05-27.txt
- cargo_fmt_result=passed
- cargo_clippy=test_results/native_container_logs/cargo_clippy_trace_emit_2026-05-27.txt
- cargo_clippy_result=passed
- cargo_test=test_results/native_container_logs/cargo_test_trace_emit_2026-05-27.txt
- cargo_test_result=passed; cmz-engine-sys=36 passed; cmz-dashboard=1 passed; native_total=37 passed
- pytest_packets=test_results/trace_emit_packet_pytest_2026-05-27.txt
- pytest_packets_result=passed; tests=5
- pytest_full=test_results/trace_emit_pytest_2026-05-27.txt
- pytest_full_result=passed; tests=146
- pytest_werror=test_results/trace_emit_pytest_werror_2026-05-27.txt
- pytest_werror_result=passed; tests=146

## Files

- native/cpp/include/cmz_engine.h
- native/cpp/src/cmz_engine.cpp
- native/crates/cmz-engine-sys/src/lib.rs
- docs/native_rust_cpp_cuda_architecture.md
- docs/project_memory.md
- docs/change_history.md
- docs/prompt_history.md
