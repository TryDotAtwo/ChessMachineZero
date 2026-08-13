# Attention cost model

For the current 1004-token, 15-stage pawn circuit, dense attention evaluates
15,120,240 score pairs. The immutable masks contain only 203,861 finite pairs,
a theoretical 74.1694x score-pair reduction.

Exact grouping by identical key sets is not an executable optimum: it produces
748–1004 blocks per stage. GPU launch and routing overhead would dominate.
The production compiler must therefore merge immutable route groups into a
small number of padded rectangular GEMMs by minimizing:

`score_flops + projection_flops + padding_flops + launch_cost * blocks + bytes`

All merging remains offline and state-independent. Runtime receives only the
frozen plan and performs matrix kernels plus deterministic hardmax.

The first budgeted optimizer preserves the mask exactly. On stage 2, a
16-block plan evaluates 171,920 padded pairs versus 131,820 useful pairs and
750,992 pairs for a single merged rectangle. With a provisional launch cost of
4,096 pair-equivalents, its estimate is 237,456 versus 755,088.

Dense selection matrices are currently used only as an exactness prototype;
no latency improvement is claimed until a static native selection layout
removes their overhead and a hardware benchmark passes.
