# Native CUDA Legal Filter Attention V1

- date=2026-05-27
- scope=king-safety legal filtering routed through CUDA board-write plus attack attention
- semantic_contract=frozen_attention_king_safety_filter
- technical_substrate=CUDA kernel; CUTLASS remains used for HullHardmax2D score path
- no_fallback=true

## Attention Mapping

- query=(board_state,move)
- board_write=apply move, promotion, castling rook move, en-passant capture
- king_safety=attack attention over next board
- output=legal bit

## TDD Red

- command=`cd /work/native && cargo test -p cmz-engine-sys frozen_move_legal_uses_cuda_king_safety_attention_without_cpu_fallback -- --nocapture`
- log=test_results/native_container_logs/cargo_test_cuda_legal_filter_expected_fail_2026-05-27.txt
- expected_failure=linker error; undefined symbol `cmz_engine_cuda_legal_filter_attention_count`

## Implementation

- c_api=cmz_engine_cuda_legal_filter_attention_count
- cuda_symbol=cmz_cuda_legal_filter_attention
- graph_fields=legal_filter_backend=cuda_king_safety_attention; make_move_backend=cuda_board_write_attention
- route=cmz_engine_frozen_move_legal -> frozen_move_legal_attention -> cmz_cuda_legal_filter_attention
- internal_route=frozen_legal_filter_layer -> cmz_cuda_legal_filter_attention
- failure_mode=CUDA error becomes explicit native error; CPU legal-filter fallback forbidden

## Verification

- command=`cd /work/native && cargo test -p cmz-engine-sys frozen_move_legal_uses_cuda_king_safety_attention_without_cpu_fallback -- --nocapture`
- log=test_results/native_container_logs/cargo_test_cuda_legal_filter_2026-05-27.txt
- result=1 passed

- command=`cd /work/native && cargo fmt --all -- --check`
- log=test_results/native_container_logs/cargo_fmt_cuda_legal_filter_2026-05-27.txt
- result=passed

- command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`
- log=test_results/native_container_logs/cargo_clippy_cuda_legal_filter_2026-05-27.txt
- result=passed

- command=`cd /work/native && cargo test --workspace`
- log=test_results/native_container_logs/cargo_test_cuda_legal_filter_workspace_2026-05-27.txt
- result=45 native tests passed

- command=`py -m pytest -p no:cacheprovider -q`
- log=test_results/cuda_legal_filter_pytest_2026-05-27.txt
- result=passed; 146 tests reached 100%

- command=`py -m pytest -p no:cacheprovider -W error -q`
- log=test_results/cuda_legal_filter_pytest_werror_2026-05-27.txt
- result=passed; 146 tests reached 100%
