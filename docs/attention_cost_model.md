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
