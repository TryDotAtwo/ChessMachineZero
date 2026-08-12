# Offline attention-plan compiler — 2026-08-12

## Implemented

`compile_attention_blocks` reads only the immutable compiled attention mask.
It groups query rows with identical finite key sets and emits frozen selection
matrices, local masks, and global key-id value tensors. This is offline compiler
work; no state-dependent routing was added to inference.

## Exact checks

- Two distinct key-set groups were reconstructed from a 4x4 mask.
- Dense and compiled-block outputs are byte-equal.
- Dense and compiled-block global hardmax winners are byte-equal.
- Every emitted plan tensor has `requires_grad=false`.
- `pytest -q tests/test_vm2_source_purity.py tests/test_public_evidence_gate.py`:
  `18 passed`.

## Claim boundary

The production transition still executes the dense reference attention path.
This checkpoint proves offline plan compilation semantics, not runtime speedup.
