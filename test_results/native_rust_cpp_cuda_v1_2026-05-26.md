# Native Rust + C++ + CUDA v1

## Environment

- base_image=gpu-dev-cutlass-nsight:2026-05-24
- native_image=cmz-native-dev:2026-05-26
- container=cmz-native-dev
- container_status=running
- rustc=1.95.0
- cargo=1.95.0
- cmake=3.22.1
- ninja=1.10.1
- cuda_runtime=12.4.1
- cuda_device=NVIDIA GeForce RTX 3070 Laptop GPU
- nvidia_driver=572.70
- gpu_probe_log=test_results/native_container_logs/nvidia_smi_2026-05-26.txt

## Implemented

- Rust workspace under `native/`.
- C++/CUDA static engine built by CMake/Ninja from Rust `build.rs`.
- Opaque C ABI engine handle.
- Safe Rust wrapper over the C ABI.
- Native Rust `MovePacket` and `TracePacket` codecs.
- Native CLI for legal move enumeration.
- C++ FEN parser and legal UCI move generator.
- CUDA exact probe kernel.
- Persistent Docker development container with log-writing exec helper.

## Verification

- tdd_note=initial native build/test run failed on missing CUDA include path in C++ compilation; CMake was fixed with CUDAToolkit include/link wiring before passing tests.
- quality_note=initial cargo fmt check failed; Rust files were formatted with cargo fmt before final fmt/clippy/test pass.

- command=`cd /work/native && cargo test --workspace`
- result=passed
- native_tests=10 passed
- log=test_results/native_container_logs/cargo_test_workspace_2026-05-26.txt

- command=`cd /work/native && cargo fmt --all -- --check`
- result=passed
- log=test_results/native_container_logs/cargo_fmt_2026-05-26.txt

- command=`cd /work/native && cargo clippy --workspace --all-targets -- -D warnings`
- result=passed
- log=test_results/native_container_logs/cargo_clippy_2026-05-26.txt

- command=`cd /work/native && cargo run -p cmz-cli --quiet -- '<startpos fen>'`
- result=passed
- legal_count=20
- cuda_available=true
- log=test_results/native_container_logs/cli_start_position_2026-05-26.txt

- command=`python -m pytest -p no:cacheprovider -q`
- result=passed
- log=test_results/native_migration_pytest_2026-05-26.txt

- command=`python -m pytest -p no:cacheprovider -W error -q`
- result=passed
- log=test_results/native_migration_pytest_werror_2026-05-26.txt

## Known Boundary

- native_v1_status=foundation_not_full_replacement
- python_dashboard_runtime=still_present
- python_percepta_trace_runtime=still_present
- native_trace_codecs=ported
- native_legal_trace_emission=not_yet_ported
- native_frozen_attention_runtime=not_yet_ported
- native_depth_search=not_implemented_by_request_deferred_to_next_stage
