# Project memory

- The active implementation is the orphan branch `codex/percepta-transformer-vm`.
- The inference runtime is `native/vm2/src/attention.cpp` plus
  `native/vm2/src/machine.cpp`.
- Runtime executes a fixed number of identical transitions and never exits
  early on `HALT`.
- `HALT` is an absorbing token state implemented by frozen attention routing.
- State writes are frozen batched matrix projections; elementwise write masks
  are offline compiler intermediates only.
- Opcode-specific compiler control flow is permitted only offline. The runtime
  target must not link `cmz_vm2_compiler`.
- Purity acceptance requires functional CTest plus mutation-tested source and
  graph-operation audits.
- The VM now has sixteen instruction slots, four predicate slots, and eleven
  fixed attention stages.
- `CMP_EQ` emits `FALSE/TRUE` through frozen equality relation tokens.
- `JUMP_IF` builds target/fallthrough scratch tokens and selects one through
  predicate-keyed hardmax attention.
- Runtime write routing uses compact row and feature matrices with
  `X + R @ (Y-X) @ C`; it does not use per-token 3D projection banks.
- The first chess slice compiles 64 white-pawn single-push candidates, fixed
  geometry tokens, and 72 legality relation tokens into a generic ProgramImage.
- Legal board application changes source, target, and side only through
  attention routing; an illegal selected candidate is an inference no-op.
- Full chess is not yet implemented and public claims must remain slice-scoped.
