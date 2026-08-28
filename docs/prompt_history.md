# Prompt History

## 2026-08-28 — Rebase the Git-linked site on the current VM

User requested that the Git-linked website be rebuilt on the current inference-only architecture. Implemented the site in the clean VM branch without touching the dirty main checkout, preserving the distinction between executable position reconstruction and the not-yet-implemented recurrent legality/status artifact stages.

## 2026-08-28

- User correction: do not migrate the old prototype; write the VM from scratch in a new branch.
- User correction: encode the resulting moving piece in the third move token so position reconstruction does not need to propagate piece identity through the entire history.
