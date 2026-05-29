# Native HullHardmax2D CUDA hardmax select v1

- date=2026-05-27
- change_id=native_hull_cuda_select_v1
- user_request=convert remaining paths to frozen-attention style; first describe exact non-attention paths
- audit_before_edit=remaining non-attention paths include HullHardmax2D host argmax, pseudo_legal_moves C++ loops, frozen_make_move_board_layer C++ transition, frozen_terminal_predicate_layer C++ terminal logic, trace packet append loop, and unused legacy legal-filter CUDA symbols
- previous_state=CUTLASS computed QK score matrix; host copied score array to CPU and selected argmax in a C++ loop
- implemented_state=CUTLASS computes QK score matrix; `cmz_hardmax_float_select_kernel` performs CUDA-side hardmax/select; host copies selected scalar index and selected scalar score only
- scope=cmz_cutlass_hardmax2d_values, cmz_cuda_hardmax2d_values
- contract=hull_select_backend=cuda_hardmax_select; hull_host_argmax=false; hull_hardmax_select_backend=cuda_hardmax_select; hull_hardmax_host_argmax=false
- semantics=frozen_attention_step=QK -> hardmax/select -> V/index

## TDD evidence

- expected_fail_log=test_results/native_container_logs/cargo_test_hull_cuda_select_expected_fail_2026-05-27.txt
- expected_fail_reason=contract lacked hull_select_backend and hull_host_argmax fields before implementation

## Verification

- log=test_results/native_container_logs/cargo_test_hull_cuda_select_targeted_2026-05-27.txt; command=`cd /work/native && cargo test -p cmz-engine-sys native_hull_hardmax_2d_uses_cutlass_gemm_frozen_attention_backend -- --nocapture`; result=passed; tests=1
- log=test_results/native_container_logs/cargo_fmt_hull_cuda_select_2026-05-27.txt; command=`cd /work/native && cargo fmt --all -- --check`; result=passed
- log=test_results/native_container_logs/cargo_clippy_hull_cuda_select_2026-05-27.txt; command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`; result=passed
- log=test_results/native_container_logs/cargo_test_hull_cuda_select_2026-05-27.txt; command=`cd /work/native && cargo test --workspace`; result=passed; tests=47 native tests
- log=test_results/hull_cuda_select_pytest_2026-05-27.txt; command=`python -m pytest -p no:cacheprovider -q`; result=passed; tests=146
- log=test_results/hull_cuda_select_pytest_werror_2026-05-27.txt; command=`python -m pytest -p no:cacheprovider -W error -q`; result=passed; tests=146
