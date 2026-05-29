# Native Docker Log Stream Fix

## Issue

- container=cmz-native-dev
- symptom=`docker logs cmz-native-dev` showed only CUDA banner and `cmz-native-dev-ready`
- cause=container entrypoint tailed `/dev/null`; `exec_native.ps1` wrote command output only to per-command files under `test_results/native_container_logs/`

## Fix

- `docker/native/Dockerfile` now tails `/work/test_results/native_container_logs/docker-stream.log`
- `docker/native/exec_native.ps1` now wraps every command with start/end markers and tees command output into `docker-stream.log`
- `docker logs -f cmz-native-dev` now shows command output from native runs

## Verification

- command=`powershell -ExecutionPolicy Bypass -File .\docker\native\run_native_container.ps1`
- result=container restarted

- command=`powershell -ExecutionPolicy Bypass -File .\docker\native\exec_native.ps1 -Command "echo CMZ_LOG_STREAM_TEST && cd /work/native && cargo test --workspace" -Log "test_results/native_container_logs/log_stream_test_2026-05-26.txt"`
- result=passed
- native_tests=10 passed

- command=`docker logs --tail 60 cmz-native-dev`
- result=showed `CMZ_LOG_STREAM_TEST`, cargo test output, and `status=0`

## Process Cleanup

- stale_dashboard_port=8768
- stopped_pids=19368,40684,41564
- final_port_8768=free
