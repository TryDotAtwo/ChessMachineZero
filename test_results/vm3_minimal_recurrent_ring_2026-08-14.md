# VM3 minimal pawn-and-knight recurrent ring evidence — 2026-08-14

## Architecture boundary

- One universal 4272-candidate bank; no pawn/knight sub-runtimes and no
  synthetic legal-set concatenation.
- One immutable 34-stage rule list. Every stage has physical Q/K width two and
  frozen weights, masks, eligibility and row routers.
- The compiler executable can write the binary image; acceptance serializes it,
  verifies every SHA-256 in the fail-closed loader and executes the loaded
  object on CPU or directly on CUDA.
- Every candidate executes the same 47-token packet graph. Source, target,
  middle-square and side activity enter through frozen matrix routing; piece
  selection and every Boolean truth gate are hard/ST attention operations.
- Runtime links no compiler objects and contains no chess branch, scalar read,
  argmax transition, CPU transfer, detach, dynamic mask or integer predicate
  cast.
- Selected source/target packets and the next board/side are matrix products of
  floating exact/ST selections. One MOVE and the selected policy K/V are
  appended without detaching.

## Exact assertions

- Initial mixed board LEGAL set is exactly white knight `b1d2/b1a3/b1c3` plus
  pawn `a2a3/a2a4`; first-legal selects `b1d2`.
- The returned board becomes the direct second input; black LEGAL is exactly
  `h7h5/h7h6` plus `g8f6/g8h6/g8e7`, and first-legal selects `h7h5`.
- `d4` pawn push and captures on `c5/e5` match the accepted VM2 pawn fixture;
  occupied intermediate square rejects `a2a4`; same-side occupancy rejects a
  knight target.
- Board and side writes are exact, output shape equals input shape, and one
  active trajectory slot is added per committed ply.
- A loss on the next board produces non-zero gradient in the LEGAL tensor and
  the input state through the full frozen-rule/ST-policy path.
- Every compiler tensor, including every stage component, matches its recorded
  SHA-256 manifest entry.

## Gates

- Full VM3 CTest: `9/9 passed`; serialized-image minimal ring CPU test `25.54 s`.
- Full Python oracle/purity suite: `50 passed`.
- Policy and minimal-ring executables also passed on NVIDIA GeForce RTX 3070
  Laptop GPU, driver `572.70`.
- Historical VM2 knight boundary executable passed. The historical monolithic
  VM2 pawn suite did not finish within 600 seconds and its narrow capture binary
  did not finish within 300 seconds; no fresh VM2 pawn GREEN is claimed. VM3
  assertions mirror the already accepted VM2 source fixtures and independently
  verify their canonical legal IDs/board effects in the unified ABI.

Tests used `cmz-native-dev:2026-05-26`; the separate clean pinned-image rebuild
gate remains open because Docker VM registry TLS is unavailable.
