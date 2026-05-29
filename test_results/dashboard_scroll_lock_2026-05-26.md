# Dashboard Trace Scroll Lock Verification

- date=2026-05-26
- target=http://0.0.0.0:8768/
- container=cmz-native-dev
- script_version=/static/dashboard.js?v=9.6

## TDD

- First new contract run: `python -m pytest -p no:cacheprovider tests\test_dashboard.py::test_dashboard_frontend_exposes_percepta_style_readable_and_token_trace_views -q`
- Expected failure: missing `traceScrollState` before implementation.

## Verification

- `python -m pytest -p no:cacheprovider tests\test_dashboard.py -q` => 13 passed
- `curl.exe --max-time 5 http://127.0.0.1:8768/` => HTTP 200 in 0.006209s
- Rendered browser check loaded `http://0.0.0.0:8768/`
- Rendered browser check confirmed `scroll-behavior=auto`
- Rendered browser check confirmed `recordTraceScrollPosition` and `restoreTraceScrollPosition` are active
- Manual-scroll lock check: `lockedTopBefore=0`, `lockedTopAfter=0`, `lockedDelta=0`, `lockedStateBefore=true`, `lockedStateAfter=true`
- Bottom-follow check: `bottomMax=2211`, `bottomAfter=2211.199951171875`, `bottomDelta=0.199951171875`, `bottomStateBefore=false`, `bottomStateAfter=false`
- Console messages: 0 errors, 0 warnings

## Behavior

- If a trace window is at the bottom, new packet rows keep following the newest rows.
- If a user scrolls a trace window upward, new packet rows do not force the viewport downward.
- Scroll position is tracked independently for white/black journals and readable/token modes.
