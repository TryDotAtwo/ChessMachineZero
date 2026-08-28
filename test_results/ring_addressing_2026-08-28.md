# Exact d_head=2 ring-address evidence

- RED: `vm_compiler.addressing` was absent.
- GREEN: all 128 vocabulary keys remained vertices of the computed 2D convex hull and each self-query selected its matching stable index.
- GREEN FP4: E2M1 sign/unit magnitude plus one positive scale per coordinate decoded bit-exactly to every float32 ring coordinate.
- GREEN capacity check: the same hull builder retained all 2048 input-row keys as outer vertices (`0..2047`).
- RED bootstrap: the artifact contained only `context_0`.
- GREEN bootstrap: strict serialization round-trip contains `context_0`, `[128,2]` vocabulary addresses, and `[2048,2]` input-row addresses.
- Full Python suite: 35 passed; measured slowest compiler tests were 1.79 s and 1.27 s on the verification run.
