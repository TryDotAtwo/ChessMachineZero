# Prompt History

## 2026-08-28 — Show matrix multiplication proof on the site

User required the website to show matrices and every inference stage in enough detail to prove that position reconstruction is only frozen tensor inference. Added exact equations, the generic C++ runtime primitive map, and a generated 45-operation SSA trace tied directly to the artifact compiler.

## 2026-08-28 — Rebase the Git-linked site on the current VM

User requested that the Git-linked website be rebuilt on the current inference-only architecture. Implemented the site in the clean VM branch without touching the dirty main checkout, preserving the distinction between executable position reconstruction and the not-yet-implemented recurrent legality/status artifact stages.

## 2026-08-28

- User correction: do not migrate the old prototype; write the VM from scratch in a new branch.
- User correction: encode the resulting moving piece in the third move token so position reconstruction does not need to propagate piece identity through the entire history.
