# VM3 universal candidate bank evidence — 2026-08-14

## Contract

- 4096 ordinary source-major, target-minor records.
- 176 promotion records: 44 source/target pairs times Q/R/B/N.
- 4272 contiguous move IDs and one final empty selector sentinel (4273 rows).
- Frozen identity tensors contain only source/target square, source/target
  file/rank and promotion one-hots.

## TDD evidence

- RED: `cmz_vm3_test_candidate_bank` failed to compile because
  `cmz_vm3/candidate_bank.h` did not exist.
- GREEN: focused native CTest passed (`1/1`).
- Full Python regression: `50 passed`.
- Full native regression: `7/7` CTests passed.

The native tests used the existing `cmz-native-dev:2026-05-26` image. This does
not close the separate clean pinned-image gate: Docker VM registry TLS remains
an external blocker for rebuilding that image from its pinned base.
