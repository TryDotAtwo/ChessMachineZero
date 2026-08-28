# Special derived-event evidence

- RED castling: after `e1-g1`, latest-event reconstruction left the rook on `h1` and `f1` empty.
- GREEN castling: the same move emits `h1=EMPTY` and `f1=WHITE_ROOK`; all four color/side variants are stored as `[4,7,128]` exact one-hot E2M1 FP4.
- RED en passant: after `e5xd6 e.p.`, reconstruction left the immediately preceding black pawn on `d5`.
- GREEN en passant: a current diagonal pawn event paired with the immediately previous opposing double-step emits `d5=EMPTY` at the current timestamp.
- Both special cases feed the same per-square latest-event hardmax as ordinary moves; no recurrent board was introduced.
- Full Python suite: 52 passed.
