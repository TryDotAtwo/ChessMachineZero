# Change History

## 2026-08-28

- Added latest-event reconstruction for castling rook movement and en-passant pawn removal; compiled all four castling transformations as an exact one-hot FP4 tensor.
- Replaced the promotion/padding third move slot with a mandatory color-specific `RESULT_PIECE` token, removed the recurrent workspace bias, and added one-hard-attention latest-event position reconstruction for ordinary moves and captures.
- Added the first position-reconstruction circuit constants: exact one-hot initial piece state, numeric square decoder, frozen service-row workspace bias, and a branch-free tensor reference transition for ordinary moves and captures.
- Extended 2D HullKV and the artifact opcode to dynamic batched Q/K/V self-attention with exact per-batch routing and floating Q/K/V backward.
- Compiled reusable chess geometry into exact binary FP4 relation tensors and combined them with bootstrap/address records in a strict non-executable rule image; vectorized exact E2M1 packing reduced relation compilation from seconds to milliseconds.
- Added exact FP4-scaled convex-ring 2D addressing for every vocabulary channel and input row, with exhaustive self-address selection tests.
- Added the exact board-free `context_0`, canonical FP4 bootstrap record, CUDA batch loader, and a development-only history-replay oracle for legal, illegal, and terminal transition acceptance.
- Bound `HULL_ATTN_2D` to frozen 2D keys and cached nested-hull indices with batched dynamic values, then verified a complete artifact-declared Transformer block and input gradient on CUDA.
- Replaced the placeholder unconditional recurrent slice with the first strict artifact-declared SSA executor: one-time FP4 CUDA materialization, generic row/GEMM/position/residual/gated-FFN/hardmax dispatch, and load-time graph validation.
- Added exact 2D HullKV CUDA support routing and differentiable hard-attention backward over exact nested-hull top-k competitors.
- Added deterministic dense-equivalence coverage for nested convex layers, including interior points, duplicate quantized keys, stable ties, and zero queries.
- Added fresh hardmax and FP4 custom-autograd STE primitives with exact Python references and real CUDA C++ tests.
- Added a fresh strict FP4 graph artifact writer/parser and independent C++ loader with explicit opcodes and no tensor-count inference.
- Added the fresh tensor-native LibTorch module and a real CUDA autograd test; no legacy executor or loader was copied.
- Created an empty orphan implementation branch after the user rejected migration of the legacy prototype.
- Added only the authoritative recurrent numeric protocol, exact tests, and architecture/project documentation.
