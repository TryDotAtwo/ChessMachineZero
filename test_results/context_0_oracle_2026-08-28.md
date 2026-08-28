# context_0 and transition-oracle evidence

- RED context: `vm_compiler.context` was absent.
- GREEN context: `[2045,128]` is exact hard one-hot with 1200 padding history rows, 20 literal sorted opening moves, 708 remaining LEGAL padding rows, `OK`, and 76 service padding rows.
- RED oracle: `vm_compiler.oracle` was absent.
- GREEN oracle: literal e2-e4 history and black legal-set expectations passed; illegal e2-e5 preserved history/legal rows and emitted `ILLEGAL_MOVE`; Fool's Mate emitted `BLACK_WIN` with an empty LEGAL_SET.
- The oracle replays only chronological native move triples from the standard initial position. It stores no board in recurrent context and is development-only.
- RED artifact compiler: `vm_compiler.compiler` was absent.
- GREEN artifact compiler: packed E2M1 `context_0` decodes bit-exactly to the one-hot matrix and survives strict artifact serialization round-trip.
- RED native loader: `FrozenVm::initial_context` was absent.
- GREEN native CUDA loader: immutable artifact tensor `context_0` expanded to `[3,2045,128]` on CUDA with exact values.
