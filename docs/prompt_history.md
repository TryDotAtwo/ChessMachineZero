# Prompt History

## 2026-09-01 — Automatic claimable draws

User selected automatic draw termination for now. The recurrent VM must match
the development oracle's `outcome(claim_draw=True)`: claimable fifty-move and
threefold positions emit `DRAW` without a separate `CLAIM_DRAW` request token.

## 2026-08-31 — Full executable VM walkthrough, inline only

User requested a website showing the complete VM step by step, then explicitly
directed implementation without subagents. The stated implementation scope is
to complete the executable recurrent artifact and expose genuine recorded
execution, not relabel the existing position-only graph. No online inference
service is added. Design and plan live under `docs/superpowers/` with the
`2026-08-31-full-vm-walkthrough` prefix. Work is not accepted as complete until
the full native transition and corresponding website trace pass their gates.

## 2026-08-31 — Approved audit corrections and a truthful bilingual site

User confirmed remediation: fix confirmed numerical/runtime defects with
regressions, then explain one consistent execution with precise tensor semantics
and complete RU/EN. Preserve generic tensor-only production execution; do not
replace the architecture or claim complete chess. Implementation plan:
`docs/superpowers/plans/2026-08-31-audit-fixes.md`.

## 2026-08-31 — Audit correctness and honesty before improving the site

User requested an audit of the current project and then better website work.
Audited the clean branch rather than the dirty legacy root. Found and
reproduced a long-history reconstruction error, backward/reference mismatches,
primitive-validation gaps and misleading website claims. Proposed bounded
correctness fixes and a single-fixture, semantic, bilingual matrix inspector;
confirmation was pending at that audit snapshot and is recorded above. Findings and command evidence are recorded in
`test_results/audit_2026-08-31.md`. No implementation/deployment is claimed.

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
