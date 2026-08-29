# Prompt History

## 2026-08-28 — Explain tensor meaning, not internal SSA names

User clarified that labels must say what each input tensor and frozen matrix represents, including the meaning of its axes, rather than exposing unexplained `SSA: v0`-style implementation identifiers. User also required row-plus-column highlighting only for actual multiplication and a centered explicit argmax operator.

## 2026-08-28 — Transformer-style bilingual matrix explorer

User required RU/EN switching, human-readable tensor meanings, conventional row-by-column matrix multiplication, and hover/click provenance for any cell of every operation from the beginning to the end of the artifact. Implemented the approved v2 matrix-flow concept across all 45 operations.

## 2026-08-28 — Show matrix multiplication proof on the site

User required the website to show matrices and every inference stage in enough detail to prove that position reconstruction is only frozen tensor inference. Added exact equations, the generic C++ runtime primitive map, and a generated 45-operation SSA trace tied directly to the artifact compiler.

## 2026-08-28 — Replace shape diagrams with actual matrices

User rejected marketing-oriented shape cards and required actual numeric matrices, products, and operations that can be followed end-to-end. Added an exact executed COO trace, all 45 operation inputs/weights/outputs, explicit attention intermediates, per-cell arithmetic, and complete JSON download.

## 2026-08-28 — Rebase the Git-linked site on the current VM

User requested that the Git-linked website be rebuilt on the current inference-only architecture. Implemented the site in the clean VM branch without touching the dirty main checkout, preserving the distinction between executable position reconstruction and the not-yet-implemented recurrent legality/status artifact stages.

## 2026-08-28

- User correction: do not migrate the old prototype; write the VM from scratch in a new branch.
- User correction: encode the resulting moving piece in the third move token so position reconstruction does not need to propagate piece identity through the entire history.
