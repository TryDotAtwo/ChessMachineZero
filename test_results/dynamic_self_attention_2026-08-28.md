# Dynamic batched 2D self-attention evidence

- RED artifact graph: `HULL_ATTN_2D` rejected three SSA inputs and the native executable failed before forward.
- GREEN CUDA kernel: queries `[B,Q,2]` search keys `[B,K,2]` only inside their own batch and return stable local candidate indices.
- GREEN custom autograd: dynamic hard routing backward produced nonzero gradients for Q, K, and V.
- GREEN two-batch isolation: identical queries selected value index 1 from the first key geometry and index 3 from its negated second-batch geometry, matching literal expected values.
- GREEN artifact graph: `ROW_ROUTE -> TOKEN_PROJECT(Q/K) -> HULL_ATTN_2D(Q,K,V)` produced exact stable row routing and propagated a nonzero gradient to the original VM input.
- Neither forward nor backward constructs a dense `Q x K^T` tensor; only `[B*Q, competitor_count]` surrogate state is materialized.
