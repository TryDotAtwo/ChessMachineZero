# Artifact-declared Transformer forward evidence

- RED batched primitive: native link failed because `hull_attention_2d_batched_ste` was undefined.
- GREEN batched primitive: hard `[B,Q,D]` routing matched the hand-derived values and backward produced nonzero gradients for dynamic queries and values.
- RED graph binding: an artifact containing `HULL_ATTN_2D` was rejected as unimplemented.
- GREEN graph binding: frozen FP4 keys and artifact-inline nested-hull indices are materialized once on CUDA and route dynamic batched values through the exact hard forward and floating selected-competitor backward.
- RED complete block: one-input all-valid `HARDMAX_STE` failed strict schema validation.
- GREEN complete block: `ROW_ROUTE -> POSITION_ADD -> TOKEN_PROJECT(Q) -> HULL_ATTN_2D -> RESIDUAL_ADD -> GATED_FFN -> RESIDUAL_ADD -> OUTPUT_PROJECT -> HARDMAX_STE` produced the hand-derived hard one-hot recurrent context.
- The complete block backward produced a defined nonzero gradient at the original `[1,2048,128]` input while all artifact weights remained frozen.
- No dense `Q x K^T` is constructed by HullKV; backward materializes only the declared competitor dimension.
