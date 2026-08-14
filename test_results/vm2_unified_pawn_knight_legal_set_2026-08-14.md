# VM2 unified pawn/knight legal set — 2026-08-14

- `compile_chess1_circuit()` has no requested-move argument.
- `bind_chess1_input()` writes board, side and requested candidate only into
  tokens; the circuit's frozen weights, masks and projections remain shared.
- Stage 9 selects a requested pawn candidate through fixed Q/K equality on
  source square and move kind, not through a candidate-specific mask.
- `LegalSetAssembler` combines pawn and knight predicate streams using exactly
  two frozen matrix multiplications and addition.
- The shared fallback policy hardmax selects across the combined predicate set.
- Exact evidence: 23/23 purity/public gates; legal-set native test passed;
  pawn capture/binding regression passed in 469.2 seconds.
- Docker had no NVIDIA driver, so GPU residency and performance remain
  unverified in this run.
