# Tensor board workspace evidence

- RED: the workspace test failed at import because no frozen initial-board bias existed.
- GREEN: adding the `[2048,128]` bias replaces padding only in 64 internal service rows and leaves every external protocol row unchanged.
- Exact artifact encoding: `initial_piece_state`, `square_decoder`, and `initial_workspace_bias` round-trip through E2M1 FP4; the bias decodes only to `-1`, `0`, and `1`.
- Tensor transition assertions: `e2-e4`, `d7-d5`, and `e4xd5` produce exact one-hot boards through the same branch-free ordinary-move formula.
- Full Python suite: 46 passed.
- Native unchanged-path verification: recurrent, artifact executor, and HullKV executables returned status `0` with LibTorch/CUDA DLLs on `PATH`.
- Native limitation: the existing standalone STE executable exits with `0xC0000409`; it was built by the older direct-link command without the required Torch CUDA registration anchor. No native source changed in this increment.
