# Executable position artifact evidence

- Artifact: 45 generic SSA operations; final internal tensor `[B,64,128]`; no chess-specific runtime opcode.
- Input routing: `ROW_ROUTE(start,end,stride=3)` separates 400 `FROM`, `TO`, and `RESULT_PIECE` rows without CPU copies.
- Frozen special recognition: three-token hardmax over four castling patterns and six-token hardmax over 28 adjacent-move en-passant patterns.
- Latest-event attention: 64 parabolic 2D queries select among 2064 initial, ordinary, and derived square events with exact chronological tie resolution.
- Development executor: exact full-board equality for ordinary capture, castling, and en passant; backward reaches the final move's `RESULT_PIECE` input.
- Exhaustive compiled-pattern gate: all 4 castling and all 28 en-passant patterns match the independent position reference, 32/32.
- Native build: changed artifact loader, executor, and CUDA acceptance sources compile under MSVC 19.44 with LibTorch 2.8/CUDA 12.5.
- Native CUDA run: `cmz_position_artifact_test.exe position_reconstruction.cmz` returned status `0` on NVIDIA GeForce RTX 3070 Laptop GPU (sm86), comparing all `[3,64,128]` values and checking nonzero input gradient.
- Full Python regression: 57 passed.
- Production purity search: no chess piece, board, castling, or en-passant terms occur under `include/` or `src/`.
- CMake note: this host's CMake 3.29 CUDA ABI/native-architecture try executable hangs after child compiler exit; native evidence used the same direct MSVC link path as earlier verified tests, including the Torch CUDA registration anchor.
