# Prompt history

## 2026-08-10

- Continue the clean orphan implementation strictly through attention weights
  and tokens.
- Add predicate-token control flow so chess rules can later emerge from repeated
  inference of explicitly compiled frozen rules.
- Do not use host `if HALT then stop`; execution itself must remain fixed
  matrix/self-attention inference.
- Audit the architecture for hidden procedural semantics and fix every detected
  boundary violation before adding predicates.
- Implement predicate-token comparison and branching entirely through frozen
  attention inference.
- Begin real chess semantics with a white-pawn single-push rule whose legality
  set and board transition are both emitted solely by fixed attention inference.
- Create a separate GitHub Pages site that visualizes real tokens, Q/K/V,
  `QK^T`, hardmax and `AV`, with claims gated by implemented evidence.

## 2026-08-11

- Publish the visual site inside the current ChessMachineZero repository using
  GitHub Pages.
- Keep the approved scientific dark design, add meaningful animations, and
  expose Percepta-style token traces.
- Continue after interrupted connectivity and extend the attention-only chess
  system without introducing host chess logic.
