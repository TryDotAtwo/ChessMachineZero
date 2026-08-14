# VM3 bounded attention-only policy evidence — 2026-08-14

## TDD boundary

- RED 1: `cmz_vm3/policy.h` absent.
- RED 2: declared `TransformerPolicy` constructor/forward/encode symbols absent.
- GREEN: exact ZeroPolicy and TransformerPolicy acceptance in one native gate.

## Proved by assertions

- effective parameters and reusable activations remain finite in `[-1,1]` for
  finite raw values up to the float64 maximum;
- all policy Q/K tensors have physical final dimension two;
- local candidate/control logits are `[B,N,2]`, with no candidate-axis
  parameter or bias and permutation-equivariant candidate scoring;
- frozen legal/witness/control matrices enforce first-legal, literal
  claim-after, forced auto-claim and all-illegal halt behavior;
- both side modules execute and a one-hot tensor route selects policy output
  and emitted move K/V without host dispatch;
- gradients reach candidate/control logits and every active historical K/V
  source; post-terminal transition gradients are zero;
- adding an inactive structural history slot changes neither forward output nor
  any earlier move/policy gradient, and the inactive move receives zero gradient;
- preflight rejects missing bounded/rebounded/local-scoring/history proof bits,
  insufficient margin, understated fan-in, bad Q/K shape and non-finite raw
  parameters before launch.

## Commands and results

- `py -m pytest -q`: `50 passed`.
- full native CTest: `8/8 passed`.
- the complete policy executable passed on CPU and CUDA.
- CUDA device: NVIDIA GeForce RTX 3070 Laptop GPU, driver `572.70`.

Tests used the existing `cmz-native-dev:2026-05-26` image. The separate clean
pinned-image rebuild remains open because Docker VM registry TLS is unavailable.
