# Structurally sparse attention without host routing

## Rejected shortcut

Runtime `nonzero`, `gather`, `index_select`, dynamic slicing, token-id reads or
host construction of active key lists is forbidden. Even when followed by a
matrix multiply, it would move semantic routing out of attention and into C++.

## Accepted graph

For each compile-time rectangular attention block `b`, the offline compiler
emits frozen sparse row-selection matrices `Rq_b` and `Rk_b`:

```text
Q  = XWq
K  = XWk
V  = XWv
Qb = Rq_b Q
Kb = Rk_b K
Vb = Rk_b V
Sb = Qb Kb^T + Mb
Ab = one_hot(lowest_index_argmax(Sb))
Ob = Ab Vb
O  = sum_b Rq_b^T Ob
```

Every query row belongs to exactly one block. Key columns in each block retain
global ascending order, preserving the deterministic lowest-global-index tie
break. The block plan depends only on frozen masks, never on token values.

## Required acceptance

1. Dense and sparse outputs are byte-identical for every existing VM stage.
2. Selected global key indices are identical, including ties.
3. Mutation tests reject dynamic index discovery and state-dependent block
   selection.
4. Runtime source contains only sparse/dense matrix multiplication, addition,
   transpose, max, one-hot and fixed plan iteration.
5. Pawn functional tests and recurrent traces remain byte-identical.
6. A measured reduction in multiplied score elements is reported separately
   from wall-clock speed; no speed claim is allowed without a benchmark.

## Compiler factorization

Queries with identical finite-key sets are grouped into one block. This is an
exact biclique cover of the already-frozen attention mask, not an approximate
top-k policy. The dense reference path remains the oracle until all six gates
pass.
