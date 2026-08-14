# Differentiable recurrent architecture audit — 2026-08-14

## Scope

- Branch: `codex/percepta-transformer-vm`.
- Audited target: current `native/vm2` runtime, pawn/knight slices,
  `LegalSetAssembler`, temporary move policy, recurrent state surface,
  trajectory/KV requirements, autograd path, terminal rules and publication
  gates.
- This artifact records architecture evidence only. It does not claim VM3 is
  implemented.

## Verified gaps in the current target

- `native/vm2/src/attention.cpp:12-18` implements `max -> one_hot`; the chosen V
  path remains differentiable, but scores, Q and K do not receive the required
  straight-through gradient. The block variants repeat it at lines 32-55 and
  66-86 and use rank-specific `.t()`/axis assumptions.
- `native/vm2/src/move_policy.cpp:19-24` repeats the same gradient cut at move
  selection; the learned policy seam is therefore not end-to-end trainable.
- `native/vm2/src/knight_runtime.cpp:5` casts the selected legal result to
  `Int64`; that path cannot carry a legality gradient.
- Pawn and knight circuits use different token counts, widths and depths and do
  not execute against one canonical recurrent state.
- `native/vm2/src/legal_set.cpp:23-28` matrix-combines already-produced
  predicate streams; its test uses synthetic inputs and is not evidence for a
  unified chess graph.
- The policy winner is not matrix-routed into a common pawn/knight board
  transition.
- There is no compact append-only move trajectory, policy KV cache, exact
  repetition identity, complete chess metadata, terminal circuit or one-loss
  whole-game BPTT trainer in this orphan branch.
- Current Q/K projections are 320/208-dimensional tensors, not a strict
  physical `d_head=2` architecture. The production block path is not safe for
  leading candidate batches.
- `native/vm2/src/machine.cpp:8-35` contains only the fixed stage/step scheduler;
  it has no trajectory, policy application or terminal state.
- `native/vm2/tools/audit_runtime.py:10-15` scans only four generic VM2 files and
  cannot prove the complete future runtime closure.

## Corrected target

The active milestone is the clean VM3 design in
`docs/superpowers/specs/2026-08-14-differentiable-recurrent-chess-machine-design.md`.
The executable TDD sequence is
`docs/superpowers/plans/2026-08-14-differentiable-recurrent-chess-machine.md`.

The defining acceptance invariant is one immutable tensor program:

```text
state_t + trajectory
  -> frozen rule trace and LEGAL_SET
  -> attention-only policy
  -> exact hard forward / stable-softmax surrogate backward
  -> frozen matrix transition
  -> one optional committed MOVE token
  -> state_t+1 with the identical ABI
```

Terminal and claim control routes are explicit tensor rows. They are not host
branches and are not counted among the 4272 chess-move candidates.

## Claim boundary

Until VM3 gates pass, only existing VM2 pawn and isolated knight evidence may be
published. In particular, this audit does not prove complete chess, gradient
reachability, GPU-only rollout, million-token training, HullKV speed or playing
strength.

## Verification commands

The docs-only change is checked with:

```powershell
git diff --check
python -m pytest tests/test_vm2_source_purity.py tests/test_public_evidence_gate.py -q
```

Observed on 2026-08-14 in this worktree:

- `git diff --check`: exit `0` (only Git's LF/CRLF conversion warnings were
  printed by the surrounding status command; no whitespace error was found).
- `python -m pytest tests/test_vm2_source_purity.py tests/test_public_evidence_gate.py -q`:
  exit `0`, `23 passed`.

## Independent design gate

The final read-only review approved the design and plan after corrections for:

- eligibility-weighted stable selection before hard/soft routing;
- identical dense/block gradients for Q/K/V and their projections;
- candidate/control-local bounded probabilities without disabled-row coupling;
- serialized, loader-checkable frozen-selector interval proofs;
- fully bounded trainable policy parameters, activations and accumulators with
  fail-closed validation before rollout launch.
