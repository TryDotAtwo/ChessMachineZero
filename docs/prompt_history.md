# Prompt history

- 2026-08-13: User requested that the right-side matrices clearly change with
  the clicked board cells and remain understandable.

- 2026-08-13: User preferred the earlier landing-page style and requested a
  clickable knight board plus a clearer explanation of the matrices.

- 2026-08-13: User requested continuing implementation and updating the site
  to show the current state beautifully and accurately.

- 2026-08-13: User requested continuing from pawn rules toward full chess.

- 2026-08-13: User said "Делай" to continue computational optimization.

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
- Replace all decorative/request-response content with literal numeric matrices
  and an exhaustive path inspector, then deploy it.
- Continue after interrupted connectivity and extend the attention-only chess
  system without introducing host chess logic.
- Continue the persistent implementation goal after connectivity interruption;
  preserve the rule that runtime semantics are only frozen attention inference.
- Continue toward complete chess after publishing Pages; add the next rule
  circuit with exact checks and no chess-aware runtime path.
