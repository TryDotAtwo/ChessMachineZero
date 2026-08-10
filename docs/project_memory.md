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
