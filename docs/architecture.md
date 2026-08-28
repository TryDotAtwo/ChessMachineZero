# Pure Frozen Transformer Chess VM

This branch is a from-scratch implementation. It does not inherit or adapt the previous engine, dashboard, opaque state codec, move-word ABI, or procedural chess runtime.

## Runtime boundary

One frozen Transformer maps a hard one-hot input tensor `[B,2048,128]` to a hard one-hot recurrent context `[B,2045,128]`. Input is the requested move `[B,3,128]` concatenated with the previous context. The standard initial position is represented by an immutable compiled `context_0` artifact.

The context contains chronological history (`1200` rows, 400 plies), sorted `LEGAL_SET` (`768` rows, 256 move triples), one status row, and 76 reserved service rows. Status vocabulary channels are `89..94`.

Production C++ is restricted to generic tensor execution, recurrent concatenation, player transport, and inference-only status control. Chess semantics live only in immutable frozen weights and masks compiled offline. Python and `python-chess` are development/test tools and are never production dependencies.

All attention heads have `d_head=2`. Exact hard attention uses 2D HullKV with lowest-index tie resolution; dense attention is a development equivalence oracle only. Canonical weights are packed FP4 with block scales and decode exactly to FP16/BF16 when native FP4 is unavailable. Forward values are hard; STE supplies a floating backward path through complete player/VM rollouts while VM weights stay frozen.

## Artifact graph contract

Graph values use SSA identifiers; value `0` is the `[B,2048,128]` input. Every operation reads already-defined values and creates one new value. Tensor attributes index immutable FP4 records materialized once on the selected CUDA device.

`HULL_ATTN_2D` reads dynamic queries `[B,Q,2]` and values `[B,K,D]`. Its frozen key tensor is `[K,2]`; its remaining attributes contain competitor count, STE temperature in milli-units, and the exact nested-hull candidate cache. Hard forward routes the stable maximum value without a dense `Q x K^T`; backward uses only the declared top-k competitors.

`HARDMAX_STE` accepts logits plus an optional computed boolean mask. With no mask input every vocabulary channel is eligible. The final graph value must be `[B,2045,128]`; malformed schemas, undefined/redefined SSA values, invalid tensor references, and invalid hull indices are rejected during VM loading.
