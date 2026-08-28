# Artifact graph executor evidence

- RED compile: `Artifact::from_records` and `FrozenVm(Artifact, Device)` did not exist.
- GREEN behavior: two identical `ROW_ROUTE -> TOKEN_PROJECT` graphs with identity versus zero FP4 weights produced the exact sliced input versus an exact zero output on CUDA.
- GREEN autograd: the identity graph propagated unit gradient to recurrent rows and zero gradient to the three consumed move rows.
- RED validation: a graph reading undefined SSA value `99` loaded successfully.
- GREEN validation: the same malformed graph is rejected during VM construction, before execution.
- Full Python suite: 25 passed.
- Native MSVC/LibTorch CUDA executable: status `0` after linking the required Torch CUDA registration anchor.
- Diagnostic correction: earlier direct link commands omitted `/INCLUDE:?warp_size@cuda@at@@YAHXZ`; this prevented `torch_cuda.dll` operator registration. The corrected command is the native evidence used here.
- `HullAttention2D` remains explicitly rejected until its artifact binding is implemented; no fallback or dense substitution is used.
