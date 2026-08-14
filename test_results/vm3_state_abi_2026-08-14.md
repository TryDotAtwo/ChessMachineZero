# VM3 recurrent state ABI evidence — 2026-08-14

- Native `cmz_vm3_state_schema` passes on CPU and RTX 3070 Laptop GPU.
- Exact coverage: 13 square states, 16 castling masks, EP NONE plus every one of
  64 squares, both sides, both adjudication modes, halfmove 0..150 boundary,
  fullmove 1..9526 radix boundary, zero cursor/claims and Running terminal/result.
- Invalid castling, EP, halfmove, fullmove and forged layout tensors fail closed.
- The output tensor shape and non-active identity/workspace columns are exactly
  preserved from `state_template`.
- Test-only python-chess adapter: 8/8 FEN/mode cases match native board and
  metadata rows. Python/FEN code is not linked into executor or compiler.
- This proves initialization and ABI only; move transition recurrence and
  trajectory append are subsequent gates.
