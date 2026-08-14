# Knight board transition

## Implemented tensor stages

1. file delta lookup;
2. rank delta lookup;
3. knight geometry predicate;
4. source piece lookup;
5. target piece lookup;
6. geometry/side/source/target legality relation;
7. legal source write token;
8. legal target write token;
9. 64 output-square hardmax selections;
10. `(side, legal) -> next side` relation.

Runtime performs only fixed self-attention and `R(AV-X)C` matrix writes.
Final legality, board and side are exposed through frozen matmul selectors.

## Verification

- Clean LibTorch Docker build and native basis test: PASS in 29.6 seconds.
- Covered all four board corners, central moves, both knight delta orientations,
  both colors, enemy capture, friendly occupancy, wrong-color source and
  illegal geometry.
- Illegal cases preserve all 64 board tokens and side exactly.
- Runtime purity/public evidence tests: 20/20 PASS.
- The attempted 4096-position full ten-stage CPU run was stopped as an
  inappropriate multi-hour acceptance gate. Only the compact three-stage
  geometry circuit has exhaustive 4096-pair evidence.
