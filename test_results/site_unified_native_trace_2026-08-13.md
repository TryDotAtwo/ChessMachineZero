# Unified native-trace site verification

- `tsc -b --pretty false`: PASS.
- `pytest tests/test_vm2_source_purity.py tests/test_public_evidence_gate.py -q`: PASS, 18 tests.
- `git diff --check`: PASS (line-ending notices only).
- Vite/Vitest execution in the managed Windows sandbox: BLOCKED by `spawn EPERM`
  during config resolution. GitHub Pages CI remains the bundle/deployment gate.
- Public UI no longer imports or renders the browser-side knight classifier.
- Public claims retain the explicit pawn-rule-slice/full-chess boundary.
