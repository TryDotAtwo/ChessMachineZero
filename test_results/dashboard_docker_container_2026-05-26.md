# Dashboard Docker Container Verification

- date=2026-05-26
- container=cmz-native-dev
- image=cmz-native-dev:2026-05-26
- dashboard_bind_container=0.0.0.0:8768
- dashboard_bind_windows=http://127.0.0.1:8768
- port_publish=127.0.0.1:8768->8768/tcp
- launch_script=docker/native/start_dashboard.ps1
- run_script=docker/native/run_native_container.ps1

## Verification

- `python -m pytest -p no:cacheprovider tests\test_dashboard.py -q` => 13 passed
- `powershell -ExecutionPolicy Bypass -File .\docker\native\run_native_container.ps1` => container recreated with published dashboard port
- `docker ps --filter "name=^/cmz-native-dev$" --format "{{.ID}} {{.Names}} {{.Status}} {{.Ports}}"` => `cmz-native-dev ... 127.0.0.1:8768->8768/tcp`
- `powershell -ExecutionPolicy Bypass -File .\docker\native\start_dashboard.ps1 -Port 8768` => dashboard process started inside container
- `Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8768/api/snapshot` => HTTP 200 from Windows host
- Browser plugin loaded `http://127.0.0.1:8768/` => title `ChessMachineZero Dashboard`
- Browser plugin rendered checks => 64 board squares, 2 assistant log cards, Percepta-style white/black token tabs, styled scrollbar color, no console warnings/errors
- Browser token checks => white token trace contains hex tokens and `white.*` packet annotations; black token trace contains hex tokens and `black.*` packet annotations

## Notes

- Dashboard now runs in Docker, not as a Windows Python process.
- Windows browser entrypoint is `http://127.0.0.1:8768`.
- `docker logs -f cmz-native-dev` streams dashboard start markers and dashboard stdout through `/work/test_results/native_container_logs/docker-stream.log`.
