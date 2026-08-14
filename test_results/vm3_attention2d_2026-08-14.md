# VM3 attention2d evidence — 2026-08-14

- Hardware: NVIDIA GeForce RTX 3070 Laptop GPU, driver 572.70.
- Runtime: `cmz-native-dev:2026-05-26`, CUDA 12.8.1, LibTorch 2.7.1+cu128.
- CPU and CUDA invocations of `cmz_vm3_test_attention2d` passed.
- Fixture shape: `[batch=2, heads=2, tokens=4, features=3]` with physical Q/K
  width 2, two immutable key blocks, one disabled surrogate route and a global
  cross-block tie fixture.
- Dense and streamed paths are byte-equal for hard indices and hard output.
- Gradients for X, Q, K, V, Wq, Wk and Wv are equal with `atol=1e-12, rtol=0`.
- `python -m pytest -q`: 40 passed after the expanded adversarial purity gate.
- This proves the attention primitive only. It does not yet prove recurrent
  chess state, full legality, whole-game BPTT or clean pinned-image CI.
