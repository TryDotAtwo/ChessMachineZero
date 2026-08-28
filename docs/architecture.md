# Pure Frozen Transformer Chess VM

This branch is a from-scratch implementation. It does not inherit or adapt the previous engine, dashboard, opaque state codec, move-word ABI, or procedural chess runtime.

## Runtime boundary

One frozen Transformer maps a hard one-hot input tensor `[B,2048,128]` to a hard one-hot recurrent context `[B,2045,128]`. Input is the requested move `[B,3,128]` concatenated with the previous context. The standard initial position is represented by an immutable compiled `context_0` artifact.

The context contains chronological history (`1200` rows, 400 plies), sorted `LEGAL_SET` (`768` rows, 256 move triples), one status row, and 76 reserved service rows. Status vocabulary channels are `89..94`.

Production C++ is restricted to generic tensor execution, recurrent concatenation, player transport, and inference-only status control. Chess semantics live only in immutable frozen weights and masks compiled offline. Python and `python-chess` are development/test tools and are never production dependencies.

All attention heads have `d_head=2`. Exact hard attention uses 2D HullKV with lowest-index tie resolution; dense attention is a development equivalence oracle only. Canonical weights are packed FP4 with block scales and decode exactly to FP16/BF16 when native FP4 is unavailable. Forward values are hard; STE supplies a floating backward path through complete player/VM rollouts while VM weights stay frozen.
