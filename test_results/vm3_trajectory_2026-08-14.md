# VM3 append-only trajectory evidence — 2026-08-14

- `cmz_vm3_trajectory` passes on CPU and RTX 3070 Laptop GPU.
- Every structural ply returns a distinct O(d_model) MoveSlot tensor.
- Committed slots are byte-equal to emitted MOVE; inactive claim/terminal slots
  are byte-equal to the immutable PAD token.
- Appending a later slot preserves the prior data pointer and value.
- Backward reaches both active emitted moves and a differentiable commit
  predicate; inactive emitted moves receive zero gradient.
- Runtime purity rejects cat, copy_, detach and tensor-indexed writes.
- This is the training tape primitive; policy K/V slots and the full recurrent
  unroll are not yet claimed.
