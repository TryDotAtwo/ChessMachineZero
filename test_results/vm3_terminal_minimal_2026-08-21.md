# VM3 minimal terminal slice — 2026-08-21

Scope:

- Implemented first terminal/absorbing slice only:
  - black/white checkmate routing from `LEGAL_SET == empty` and own-king check;
  - stalemate routing from `LEGAL_SET == empty` and no own-king check;
  - automatic seventy-five-move draw at halfmove clock 150;
  - already terminal state remains token-absorbing and appends inactive MOVE padding.
- Not implemented/claimed here:
  - repetition/fivefold;
  - literal claim-draw modes;
  - insufficient-material subsets;
  - GPU terminal evidence in this run.

Architecture notes:

- Terminal/result state rows are compiler-frozen tensors.
- `any_legal` is produced by fixed matmul reducer plus boolean attention.
- Terminal class/result/commit permission is produced by an 8-row frozen lookup selected with deterministic ST attention.
- Runtime executor does not use `.item`, CPU transfer, `where`, `gather`, `cat`, host chess `if/switch`, or Python chess semantics.

Evidence:

```text
docker run ... python3 native/vm3/tools/audit_runtime.py --root .
=> forbidden_findings=[], unclassified_runtime_sources=[], lookup_head_dim=2
```

```text
docker run ... cmake --build native/vm3/build --target cmz_vm3_test_terminal -j1
docker run ... native/vm3/build/cmz_vm3_test_terminal
=> exit code 0
```

```text
py -m pytest -q tests/test_vm3_source_purity.py tests/oracle/test_vm3_perft_oracle.py
=> 19 passed, 8 skipped
```

```text
docker run ... cmake --build native/vm3/build --target cmz_vm3_test_special_moves cmz_vm3_test_king_safety -j1
docker run ... native/vm3/build/cmz_vm3_test_special_moves
docker run ... native/vm3/build/cmz_vm3_test_king_safety
=> exit code 0
```

Environment caveat:

- The Docker runs reported `WARNING: The NVIDIA Driver was not detected`, so the native gates above are CPU-only in this run.
