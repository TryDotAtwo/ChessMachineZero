# Result-piece latest-event reconstruction evidence

- RED: protocol and state-circuit tests failed because `PIECE_CHANNELS` and `reconstruct_position_from_history` did not exist.
- Move language: every move is `[FROM, TO, RESULT_PIECE]`; only channels 96 through 107 are accepted in the third row.
- Promotion representation: the third row names the resulting promoted piece directly; there is no separate promotion code.
- Latest-event attention reference: each square selects the latest matching initial/FROM/TO event; the tested `e2-e4, d7-d5, e4xd5` history reconstructs exact one-hot occupancy and piece identity.
- Validation oracle: geometrically legal `e2-e4` with a falsely declared white knight is rejected as `ILLEGAL_MOVE`.
- The recurrent context contains history, LEGAL_SET, status, and padding service rows; it carries no hidden board matrix.
- Full Python suite: 50 passed.
