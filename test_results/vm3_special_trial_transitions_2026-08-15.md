# VM3 special trial-transition evidence — 2026-08-15

## Implemented boundary

- One serialized immutable program image: 112 physical `d_head=2` attention
  stages, identical 60-token packet layout, 4272 universal candidates.
- Frozen rule predicates cover ordinary moves, four promotions, en-passant
  geometry/state and all four castling geometry/right/rook/empty-path cases.
- Matrix trial writes produce board, castling rights, raw EP, halfmove and
  two-digit radix fullmove state for every candidate.
- The recurrent ring selects the complete candidate trial tensor and writes it
  back to the canonical same-shape state. There is no runtime special-move
  `if`/`switch` or compiler linkage.

## Exact gates

- Native CTest: `11/11` VM3 tests passed (`93.15 s`).
- GPU: `cmz_vm3_test_special_moves --cuda` passed, including backward from the
  special rule/trial result to canonical input state.
- GPU: current `cmz_vm3_test_minimal_ring --cuda` passed after expansion to the
  112-stage image.
- Python suite: `51/51` tests passed. This includes source-purity/public-claim
  gates, state oracle and the 520-position GPU pseudo-legal oracle.
- Oracle policy: 8 curated positions cover white/black castling, white/black
  en-passant and promotions; 512 deterministic random-walk positions validate
  the full ordinary pseudo-legal bank. Random positions intentionally clear
  rights/EP because attacked-castling king safety belongs to Task 11.
- `git diff --check`: clean apart from platform line-ending notices.

## Honest limitation

This is full special **pseudo-legality and trial transition**, not complete
legal chess. The image does not yet reject moves that leave the king attacked,
does not yet reject castling through attack, and does not yet derive checkmate,
stalemate, repetition or move-rule terminal outcomes.

The evidence uses the existing `cmz-native-dev:2026-05-26` image (CUDA 12.8.1,
LibTorch 2.7.1+cu128). A clean rebuild from a separately pinned base image
remains a later reproducibility gate.
