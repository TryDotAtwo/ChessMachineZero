# Parametric knight geometry circuit

## Input and frozen rule structure

- Pair binding changes only source-file, target-file, source-rank and
  target-rank one-hot coordinates on the query token.
- There is no 64x64 pair table and no legal bit in the input.
- Frozen rule banks contain 64 generic one-dimensional coordinate relations
  for file, the same shape for rank, and three final predicate rows.

## Runtime graph

For each of three fixed stages:

```text
Q = XWq; K = XWk; V = XWv
A = one_hot(lowest_index_argmax(QK^T + M))
X' = X + R(AV - X)C
```

The result is read with two fixed matrix selectors. `knight_runtime.cpp`
contains no coordinate arithmetic, `abs`, host semantic branch, scalar read,
indexed write, elementwise routing or precomputed legality field.

## Verification

- Clean LibTorch Docker build: PASS.
- Exhaustive native source/target pairs: 4096/4096 PASS.
- Runtime/source purity plus public evidence: 20/20 PASS.
- CPU semantic test only; no performance claim.
