# STE primitive evidence

- RED Python: `vm_compiler.ste` was absent.
- RED C++: `cmz/ste.h` was absent.
- GREEN Python: exact tied masked-hardmax forward/backward and FP4 quantized-forward/identity-backward tests passed.
- GREEN C++/CUDA: LibTorch custom-autograd executable linked and returned status `0` on the real CUDA device.
- Hardmax forward chooses the lowest tied index; backward is the masked softmax Jacobian at the declared temperature.
- FP4 forward chooses the nearest E2M1 lattice value; backward passes a floating identity gradient.
- Production `src/ste.cpp` contains no detach, item, CPU transfer, host argmax, top-k, or procedural loop.
