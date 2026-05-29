# Native trace runtime v2

## Scope

- milestone_id=native_trace_runtime_v2
- goal=move trace emission for legal enumeration and make-move execution into the native Rust + C++ runtime boundary
- runtime_stack=Rust wrapper -> C ABI -> C++ engine; CUDA probe remains active; Python dashboard/runtime not replaced in this step

## Implemented

- C ABI legal trace packet export: `cmz_engine_legal_trace_packets`
- C ABI legal trace stream: `cmz_engine_legal_trace_begin`, `cmz_engine_legal_trace_next`
- C ABI make-move trace export: `cmz_engine_make_move_trace_packets`
- Rust wrapper methods:
  - `Engine::legal_trace_packets`
  - `Engine::begin_legal_trace_stream`
  - `Engine::decode_next_legal_trace_packet`
  - `Engine::make_move_trace_packets`
- Native legal trace rows:
  - `CANDIDATE`
  - `LEGAL_SET`
  - `PROGRAM_HALT`
- Native make-move trace rows:
  - `COMMIT_MOVE`
  - changed-square `WRITE_SQ`
  - `WRITE_REG`
  - `WRITE_CASTLE`
  - `WRITE_EP`
  - `WRITE_CLOCK`
  - `TERMINAL_SET`
  - `PROGRAM_HALT`
- Native trace stream uses a C++ matrix-attention slice: hardmax-select over packet memory by decode cursor, exposed as `runtime_mode=native_matrix_attention_trace_decoder`.

## TDD Evidence

- first legal trace test failed on missing `Engine::legal_trace_packets`
- first stream trace test failed on missing `Engine::begin_legal_trace_stream` and `Engine::decode_next_legal_trace_packet`
- first make-move trace test failed on missing `Engine::make_move_trace_packets`
- first matrix-attention stream metadata test failed on missing `Engine::runtime_mode` and `Engine::attention_decode_count`

## Verification

- command=`cd /work/native && cargo test -p cmz-engine-sys native_legal_trace -- --nocapture`
- result=passed
- log=test_results/native_container_logs/cargo_test_native_trace_2026-05-26.txt

- command=`cd /work/native && cargo test -p cmz-engine-sys native_legal_trace_stream -- --nocapture`
- result=passed
- log=test_results/native_container_logs/cargo_test_native_trace_stream_2026-05-26.txt

- command=`cd /work/native && cargo test -p cmz-engine-sys native_make_move_trace -- --nocapture`
- result=passed
- log=test_results/native_container_logs/cargo_test_native_make_move_trace_2026-05-26.txt

- command=`cd /work/native && cargo fmt --all -- --check`
- result=passed
- log=test_results/native_container_logs/cargo_fmt_native_trace_v2_2026-05-26.txt

- command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`
- result=passed
- log=test_results/native_container_logs/cargo_clippy_native_trace_v2_2026-05-26.txt

- command=`cd /work/native && cargo test --workspace`
- result=passed
- native_tests=14 passed
- log=test_results/native_container_logs/cargo_test_workspace_native_trace_v2_2026-05-26.txt

- command=`cd /work/native && cargo run -p cmz-cli --quiet -- --trace`
- result=passed
- legal_trace_packet_count=41
- legal_trace_token_count=287
- runtime_mode=native_matrix_attention_trace_decoder
- log=test_results/native_container_logs/cli_trace_start_position_2026-05-26.txt

- command=`python -m pytest -p no:cacheprovider tests\test_trace_packet.py tests\test_move_packet.py -q`
- result=passed
- python_packet_tests=5 passed
- log=test_results/native_trace_v2_python_packet_tests_2026-05-26.txt

## Known Boundary

- native_trace_runtime_v2_status=trace_emission_native_not_full_runtime_replacement
- python_dashboard_runtime=still_present
- python_percepta_frozen_attention_runtime=still_present
- native_frozen_attention_matrix_interpreter=partial_stream_decode_slice_only
- native_cuda_attention_kernels=not_yet_ported
