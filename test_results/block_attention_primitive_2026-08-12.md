# Frozen block-attention primitive — 2026-08-12

## Implemented

`block_self_attention` receives only offline-frozen query/key selection
matrices, local masks and global-key-id value tensors. Runtime performs matrix
selection, local `QK^T`, deterministic hardmax, `AV`, and transpose-matrix
routing back to global rows.

The exact test uses two blocks, non-contiguous global keys and hardmax ties.
Block output and global winner indices equal the dense attention oracle
byte-for-byte.

## Purity

```text
16 passed
```

The audit still reports `allowed_tensor_graph=true`,
`forbidden_routing=false`, `semantic_branching=false`, and
`chess_runtime=false`. Dynamic `nonzero`, `index_select`, `gather` and scatter
remain mutation-tested forbidden operations.

## Claim boundary

The primitive exists, but `transition` still uses dense `self_attention`.
No sparse VM or performance improvement is claimed until an offline
mask-to-block compiler and stage-by-stage dense equivalence gates pass.
