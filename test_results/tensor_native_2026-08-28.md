# Fresh tensor-native VM evidence

- RED: direct MSVC compilation of `tests/native/recurrent_test.cpp` failed because `cmz/vm_module.h` did not exist.
- GREEN: `module.cpp` and the real LibTorch C++ test compiled and linked against PyTorch `2.8.0+cu128`.
- GPU execution returned status `0` on the CUDA device after asserting `[2,2048,128] -> [2,2045,128]`, same device, retained autograd, zero request-prefix gradient, and unit recurrent-context gradient.
- The production module contains only tensor validation and differentiable row routing. It has no host copy, FFI, artifact loader, chess type, integer move token, or legacy executor.
