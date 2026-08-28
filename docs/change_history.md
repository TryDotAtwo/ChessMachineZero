# Change History

## 2026-08-28

- Added exact 2D HullKV CUDA support routing and differentiable hard-attention backward over exact nested-hull top-k competitors.
- Added deterministic dense-equivalence coverage for nested convex layers, including interior points, duplicate quantized keys, stable ties, and zero queries.
- Added fresh hardmax and FP4 custom-autograd STE primitives with exact Python references and real CUDA C++ tests.
- Added a fresh strict FP4 graph artifact writer/parser and independent C++ loader with explicit opcodes and no tensor-count inference.
- Added the fresh tensor-native LibTorch module and a real CUDA autograd test; no legacy executor or loader was copied.
- Created an empty orphan implementation branch after the user rejected migration of the legacy prototype.
- Added only the authoritative recurrent numeric protocol, exact tests, and architecture/project documentation.
