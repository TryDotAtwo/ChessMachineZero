# Differentiable Recurrent ChessMachineZero Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single immutable GPU-resident chess ring whose exact hard
forward executes complete orthodox move legality and the explicitly declared
v10 claim/terminal set, and whose deterministic straight-through backward
carries terminal loss through every ply to a swappable trainable attention-only
policy. General FIDE dead-position recognition remains a separate proof gate.

**Architecture:** Keep `native/vm2` as accepted hardmax reference evidence and
build the new contract under `native/vm3`. VM3 uses true two-dimensional lookup
heads, a canonical fixed-shape recurrent state, 4272 chess-move candidates plus
factorized claim/halt control, one append-only logical move token per
committed ply, frozen rule tensors, a trainable policy module, exact-forward
custom-autograd selection and an absorbing terminal state. Dense attention is
the training/reference backend;
2D convex-hull lookup is a forward-equivalent inference backend added only
after semantic and gradient acceptance.

**Tech Stack:** C++17, LibTorch 2.7.1/CUDA 12.8, OpenSSL 3 SHA-256,
CMake/Ninja/CTest, Python 3.11 pytest with `python-chess` test oracle,
TypeScript/Vite site, GitHub Actions.

## Global Constraints

- Production runtime contains no chess-aware branches, semantic scalar reads,
  CPU copies, FEN/UCI conversion, state-dependent weights/masks/block plans or
  dynamic candidate dispatch.
- Candidate identity is input data; legality is produced only by frozen rule
  attention.
- Every selector uses an exact hard lowest-index custom-autograd forward and the
  canonical stable-softmax straight-through backward. Literal floating-point
  `hard + soft - soft.detach()` is forbidden because it is not a proof of
  byte-exact hard output.
- Frozen rule tensors have no parameter gradients; recurrent state and history
  are never detached between plies.
- Only policy parameters train. No tree search, handcrafted evaluation, engine
  labels, tablebase labels or human-game data.
- Python chess is restricted to test-oracle code.
- Each committed move appends exactly one logical move token; claim and
  terminal routes append none.
- Full FIDE state includes 13 square states, side, castling rights, en-passant,
  halfmove/fullmove counters, claims, terminal/result and exact repetition
  identity.
- Training/reference lookup heads and optimized lookup heads have `d_head=2`.
- Exact dense softmax backward is not claimed to be logarithmic. HullKV speed
  claims apply only to measured hard-forward lookup.
- Every milestone is TDD red-green-refactor, updates project/change/prompt
  history, stores exact evidence in `test_results/`, and advances public claims
  only after native acceptance.

---

## File map

New VM3 files have one responsibility each:

All interface snippets below use `using Tensor = torch::Tensor;` and fully
qualified standard-library containers in implementation headers.

```text
native/vm3/
  CMakeLists.txt
  include/cmz_vm3/
    tensor_contract.h      # dimensions, enums and immutable ABI constants
    st_selector.h          # canonical hard-forward/soft-backward primitive
    attention2d.h          # dense and block 2D attention interfaces
    chess_state.h          # canonical recurrent state layout and views
    trajectory.h           # one-token append protocol and active-row routing
    candidate_bank.h       # 4272 raw move identities and sentinel descriptor
    policy.h               # ZeroPolicy and trainable policy ABI
    program_image.h        # versioned/checksummed frozen tensor image
    recurrent_ring.h       # generic one-step and fixed whole-game executor
    trace_exporter.h       # native forward/backward artifact schema
  src/
    st_selector.cpp
    attention2d.cpp
    chess_state_views.cpp
    trajectory.cpp
    policy.cpp
    program_loader.cpp
    recurrent_ring.cpp
    trainer.cpp
    hull_kv2d.cpp
    trace_exporter.cpp
  tests/
    test_st_selector.cpp
    test_attention2d.cpp
    test_state_schema.cpp
    test_trajectory.cpp
    test_candidate_bank.cpp
    test_policy.cpp
    test_minimal_ring.cpp
    test_pseudo_legal.cpp
    test_king_safety.cpp
    test_special_moves.cpp
    test_terminal.cpp
    test_full_ring.cpp
    test_full_bptt.cpp
    test_hull_kv2d.cpp
  tools/
    audit_runtime.py
    export_ring_trace.cpp
  compiler/
    candidate_bank.cpp
    initial_state_binder.cpp
    rule_compiler.cpp
    pawn_rules.cpp
    piece_rules.cpp
    attack_rules.cpp
    special_move_rules.cpp
    terminal_rules.cpp
    compile_program.cpp
tests/
  oracle/test_vm3_oracle.py
  test_vm3_source_purity.py
  test_vm3_public_evidence_gate.py
docker/vm3/Dockerfile
tools/vm3-ci.ps1
tests/requirements-vm3.txt
.github/workflows/vm3.yml
```

### Task 1: Freeze the VM3 contract and adversarial purity gate

**Files:**
- Create: `native/vm3/CMakeLists.txt`
- Create: `native/vm3/include/cmz_vm3/tensor_contract.h`
- Create: `native/vm3/include/cmz_vm3/program_image.h`
- Create: `native/vm3/src/program_loader.cpp`
- Create: `native/vm3/tools/audit_runtime.py`
- Create: `tests/test_vm3_source_purity.py`
- Modify: root `CMakeLists.txt`
- Create: `docker/vm3/Dockerfile`
- Create: `tools/vm3-ci.ps1`
- Create: `tests/requirements-vm3.txt`
- Create: `.github/workflows/vm3.yml`

**Interfaces:**
- Produces: `kLookupHeadDim=2`, `kMoveCandidateCount=4272`,
  `kCandidateSelectorRouteCount=4273`, `kPolicyControlCount=3`,
  `kControlRouteCount=4`,
  `kPieceStateCount=13`, `kHalfmoveAutomaticLimit=150`, runtime/compiler target
  names and forbidden-operation policy consumed by every later task.

- Defines four disjoint CMake targets:
  `cmz_vm3_executor`, `cmz_vm3_policy`, `cmz_vm3_compiler` and
  `cmz_vm3_program_loader`. Only the compiler target may compile files below
  `native/vm3/compiler/`. The executor's transitive link closure is
  `executor + policy + loader + LibTorch`; compiler objects are forbidden.
- `FrozenChessProgram` serialization contains schema version, tensor names,
  shapes, dtypes, stage order, SHA-256 payload checksums, interval-proof nodes
  and one score certificate per frozen selector. The generic loader recomputes
  every interval from canonical input domains and loaded fixed tensor values;
  it never trusts claimed bounds or derives chess rules.

```cpp
struct ProgramTensorManifest {
  std::string name;
  std::vector<int64_t> shape;
  c10::ScalarType dtype;
  std::array<std::uint8_t, 32> sha256;
};
enum class IntervalOp : std::uint8_t {
  CanonicalInput, BoundedPolicyProbability, FrozenMatmul, Add, Multiply,
  EligibilityMargin, ReduceMax
};
struct IntervalProofNode {
  IntervalOp op;
  std::vector<std::uint32_t> inputs;
  std::string tensor_name;
  double claimed_lower, claimed_upper;
};
struct SelectorScoreCertificate {
  std::string selector_name;
  std::uint32_t proof_root;
  double enabled_score_lower;
  double disabled_score_upper;
  std::int64_t sentinel_row;
};
struct Attention2dBlock {
  Tensor query_router, key_router, output_router, fixed_mask, fixed_eligibility,
         global_key_ids;
};
struct Attention2dPlan {
  std::vector<Attention2dBlock> blocks;
  std::vector<std::int64_t> query_group_offsets;
};
struct FrozenStage {
  Tensor wq, wk, wv, fixed_mask, row_router, feature_router;
  Attention2dPlan attention_plan;
};
struct FrozenChessProgram {
  std::uint32_t schema_version;
  std::vector<ProgramTensorManifest> manifest;
  std::unordered_map<std::string, torch::Tensor> tensors;
  std::vector<IntervalProofNode> interval_proof;
  std::vector<SelectorScoreCertificate> selector_certificates;
  std::vector<FrozenStage> pre_policy_stages;
  std::vector<FrozenStage> post_policy_stages;
};
FrozenChessProgram load_frozen_program(const std::filesystem::path& path,
                                       const torch::Device& device);
```

Before device capture, the loader topologically replays every proof node with
outward-rounded interval arithmetic. `FrozenMatmul` derives coefficient bounds
from the checksum-verified tensor payload; `EligibilityMargin` evaluates both
exact domains `e=0` and `e=1`. It rejects cycles, missing roots, non-finite
bounds, mismatched claimed intervals, missing sentinels, or any certificate with
`enabled_score_lower <= disabled_score_upper`. These initialization reads are
generic artifact validation and cannot inspect a live chess state.
`BoundedPolicyProbability` has the fixed domain `[0,1]`; executor preflight may
use that proof node only after the loaded policy's descriptor and exported op
manifest pass the `BoundedPolicyContract` check.

- [ ] **Step 1: Write failing contract tests**

```python
def test_vm3_contract_is_two_dimensional_and_single_image():
    h = (ROOT / "native/vm3/include/cmz_vm3/tensor_contract.h").read_text()
    assert "kLookupHeadDim = 2" in h
    assert "kMoveCandidateCount = 4272" in h
    assert "kCandidateSelectorRouteCount = 4273" in h
    assert "kPolicyControlCount = 3" in h
    assert "kControlRouteCount = 4" in h
    assert "CompileMode" not in h

def test_vm3_auditor_rejects_semantic_mutations(tmp_path):
    result = run_mutation(tmp_path, "if (state.item<int>()) apply_move();")
    assert result.returncode == 1

def test_loader_rejects_forged_selector_bound(vm3_program_copy):
    corrupt_claimed_bound_without_changing_proof(vm3_program_copy)
    assert load_frozen_program_fails(vm3_program_copy, "selector certificate")
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests/test_vm3_source_purity.py -q
```

Expected: FAIL because VM3 files and auditor do not exist.

- [ ] **Step 3: Add exact ABI constants and an auditor that scans the complete
runtime target closure**

```cpp
inline constexpr std::int64_t kLookupHeadDim = 2;
inline constexpr std::int64_t kMoveCandidateCount = 4096 + 176;
inline constexpr std::int64_t kCandidateSelectorRouteCount = kMoveCandidateCount + 1;
inline constexpr std::int64_t kPolicyControlCount = 3; // MOVE, CLAIM_NOW, CLAIM_AFTER
inline constexpr std::int64_t kControlRouteCount = kPolicyControlCount + 1; // HALT
inline constexpr std::int64_t kPieceStateCount = 13;
inline constexpr std::int64_t kHalfmoveClaimLimit = 100;
inline constexpr std::int64_t kHalfmoveAutomaticLimit = 150;
inline constexpr std::int64_t kMaxOrthodoxGamePlies = 19050;
inline constexpr std::int64_t kMaxOrthodoxRingSteps = 19051;
```

The auditor must reject `.item`, `to(torch::kCPU)`, `NoGradGuard`, semantic
`if/switch`, `index_put_` in the captured executor, state-dependent masks, every
recurrent `detach`, compiler linkage and chess vocabulary in generic attention/
executor files. Structural manifest/error checks are allowed only in the
initialization loader, before capture. The custom ST operation contains no
detach at all.

Add a link-closure test that configures CMake, inspects the executor link command
and symbol map, and fails if it contains `cmz_vm3_compiler`, `pawn_rules`,
`piece_rules`, `attack_rules`, `special_move_rules` or `terminal_rules`.

Add an exported operation-manifest gate over the captured executor. The
forward allowlist is matrix multiply/batched matrix multiply, transpose/view,
fixed add/subtract/multiply, max/argmax, one-hot, clone, frozen matrix write and
stable coordinate-wise/local-two-class softmax only inside registered bounded
policy operations. Training additionally allows eligibility-weighted softmax
and its tensor reductions inside the declared custom ST backward. Gather,
nonzero, tensor-driven indexing,
`where`, sort/top-k, host callbacks and unregistered custom operations fail the
gate.

The Dockerfile starts from
`nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`, installs CMake, Ninja, GCC,
OpenSSL development headers and Python, downloads the already checksum-pinned
`torch-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl` with SHA-256
`d6c3cba198dc93f93422a8545f48a6697890366e4b9701f54351fc27e2304bd3`, and
exposes its LibTorch CMake prefix at `/opt/pytorch/share/cmake`.

`tools/vm3-ci.ps1` is the one reproducible entrypoint used by every later task:

```powershell
param(
  [string]$Target = "all",
  [string]$CTestRegex = "cmz_vm3",
  [string]$PytestArgs = "",
  [switch]$WithGpu
)
$gpuArgs = if ($WithGpu) { @("--gpus", "all") } else { @() }
$pytestCommand = if ($PytestArgs) { " && python3 -m pytest $PytestArgs" } else { "" }
docker build -f docker/vm3/Dockerfile -t cmz-vm3-dev:2.7.1-cu128 .
docker run --rm @gpuArgs -v "${PWD}:/work" -w /work cmz-vm3-dev:2.7.1-cu128 `
  bash -lc "cmake -S . -B build/vm3 -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH=/opt/pytorch/share/cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && cmake --build build/vm3 --target $Target && ctest --test-dir build/vm3 --output-on-failure -R '$CTestRegex'$pytestCommand"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

`tests/requirements-vm3.txt` pins `pytest==8.4.1` and
`python-chess==1.999`; the Docker build installs exactly this file.

The workflow runs the same script for CPU semantic/purity gates. GPU
determinism/residency gates use a labeled self-hosted runner and are never
silently replaced by CPU checks.

- [ ] **Step 4: Run GREEN and mutation cases**

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_contract -CTestRegex '^cmz_vm3_contract$' -PytestArgs 'tests/test_vm3_source_purity.py -q'
```

Expected: PASS, including every adversarial fixture.

- [ ] **Step 5: Commit**

```powershell
git add CMakeLists.txt native/vm3 docker/vm3 tools/vm3-ci.ps1 tests/requirements-vm3.txt tests/test_vm3_source_purity.py .github/workflows/vm3.yml
git commit -m "test: define vm3 tensor-only contract"
```

### Task 2: Canonical deterministic straight-through selector

**Files:**
- Create: `native/vm3/include/cmz_vm3/st_selector.h`
- Create: `native/vm3/src/st_selector.cpp`
- Create: `native/vm3/tests/test_st_selector.cpp`
- Modify: `native/vm3/CMakeLists.txt`

**Interfaces:**
- Produces:

```cpp
struct StSelection {
    torch::Tensor probabilities;
    torch::Tensor hard_indices;
    torch::Tensor hard;
    torch::Tensor straight_through;
};
StSelection deterministic_st_select(const torch::Tensor& scores, double temperature);
StSelection deterministic_st_select(const torch::Tensor& scores,
                                    const torch::Tensor& surrogate_eligibility,
                                    double temperature);
```

- [ ] **Step 1: Write forward, tie, gradient and stability tests**

```cpp
auto scores = torch::tensor({{2., 2., -4.}}, options.requires_grad(true));
auto out = deterministic_st_select(scores, 0.5);
require(torch::equal(out.hard, out.straight_through), "custom forward is exact");
require(torch::equal(out.straight_through,
                     torch::tensor({{1., 0., 0.}}, options)), "hard forward");
torch::matmul(out.straight_through, values).sum().backward();
require(scores.grad().abs().sum().item<double>() > 0., "score gradient");
```

Also compare `scores.grad()` and `surrogate_eligibility.grad()` with the exact
eligibility-weighted surrogate formula below, including an `eligibility=0`
boundary. Test finite positive temperature validation, the certified finite
score range, a fixed finite sentinel for otherwise disabled rows and no NaN/Inf.
A selector invocation with no enabled real row must return the sentinel. A
fixture whose score/margin certificate permits an ineligible maximum must be
rejected by compiler/loader validation, not inspected through a runtime scalar.

- [ ] **Step 2: Run RED**

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_st_selector -CTestRegex '^cmz_vm3_st_selector$'
```

Expected: FAIL with missing selector symbols.

- [ ] **Step 3: Implement an exact-forward custom autograd operation**

```cpp
// `scores` already contains eligibility/priority margins validated either by a
// frozen-program interval proof or by the bounded policy contract, so its
// global maximum is guaranteed to be an eligible finite row.
const auto shifted = scores - std::get<0>(scores.max(-1, true));
const auto unnormalized = torch::exp(shifted / temperature)
                        * surrogate_eligibility;
const auto soft = unnormalized / unnormalized.sum(-1, true);
const auto index = std::get<1>(scores.max(-1, false));
const auto hard = at::one_hot(index, scores.size(-1)).to(scores.scalar_type());

class ExactHardSoftBackward final
    : public torch::autograd::Function<ExactHardSoftBackward> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext*,
                               torch::Tensor hard_forward,
                               torch::Tensor soft_surrogate) {
    (void)soft_surrogate;
    return hard_forward.clone();
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext*,
      torch::autograd::variable_list upstream) {
    // No gradient through argmax; exactly the same upstream is sent through
    // the already-built stable-softmax graph.
    return {torch::Tensor(), upstream.at(0)};
  }
};

const auto st = ExactHardSoftBackward::apply(hard, soft);
```

`st` must be byte-equal to `hard`; its score and eligibility gradients must
equal the gradient of replacing `st` by the formula's `soft`. A frozen interval
proof or audited bounded-policy contract plus the sentinel proves that the
global maximum is eligible and the denominator positive. Initialization must
reject non-positive temperature or an invalid score contract, never through a
runtime scalar read. This definition needs no `gt`, `where`, dynamic mask
materialization or non-finite score.

- [ ] **Step 4: Run GREEN plus purity**

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_st_selector -CTestRegex '^cmz_vm3_st_selector$'
python -m pytest tests/test_vm3_source_purity.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add native/vm3 tests/test_vm3_source_purity.py
git commit -m "feat: add deterministic straight-through selector"
```

### Task 3: True two-dimensional dense and block attention

**Files:**
- Create: `native/vm3/include/cmz_vm3/attention2d.h`
- Create: `native/vm3/src/attention2d.cpp`
- Create: `native/vm3/tests/test_attention2d.cpp`
- Modify: `native/vm3/CMakeLists.txt`

**Interfaces:**
- Consumes: `deterministic_st_select`.
- Produces:

```cpp
struct Attention2dOutput { Tensor q, k, v, hard_indices, output; };
struct DenseAttention2dResult {
  Attention2dOutput common;
  Tensor scores, probabilities, hard;
};
DenseAttention2dResult dense_attention2d(Tensor x, Tensor wq, Tensor wk,
                                         Tensor wv, Tensor fixed_mask,
                                         Tensor fixed_eligibility,
                                         double temperature);
Attention2dOutput block_attention2d(Tensor x, Tensor wq, Tensor wk, Tensor wv,
                                    const Attention2dPlan&, double temperature);
```

`wq` and `wk` must have final dimension exactly two; leading batch and head
dimensions are supported.

- [ ] **Step 1: Write RED tests for `d_head=2`, batch safety and dense/block
forward-and-gradient parity for X, Q, K, V, Wq, Wk and Wv**

```cpp
require(dense.common.q.size(-1) == 2 && dense.common.k.size(-1) == 2, "real 2D heads");
require(torch::equal(dense.common.hard_indices, block.hard_indices), "hard parity");
require(torch::allclose(dense_x_grad, block_x_grad, 1e-12, 0.), "gradient parity");
require(torch::allclose(dense_wq_grad, block_wq_grad, 1e-12, 0.), "Wq gradient");
require(torch::allclose(dense_wk_grad, block_wk_grad, 1e-12, 0.), "Wk gradient");
require(torch::allclose(dense_wv_grad, block_wv_grad, 1e-12, 0.), "Wv gradient");
```

- [ ] **Step 2: Run RED**, expecting missing attention implementation.

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_attention2d -CTestRegex '^cmz_vm3_attention2d$'
```

- [ ] **Step 3: Implement dense reference with
`K.transpose(-2,-1)`, last-axis ST selection and frozen block row matrices.**

```cpp
q = torch::matmul(x, wq);                         // [..., H, Nq, 2]
k = torch::matmul(x, wk);                         // [..., H, Nk, 2]
v = torch::matmul(x, wv);                         // [..., H, Nk, Dv]
scores = torch::matmul(q, k.transpose(-2, -1)) + fixed_mask;
selection = deterministic_st_select(scores, fixed_eligibility, temperature);
output = torch::matmul(selection.straight_through, v);
```

The block plan groups contiguous, globally key-ordered blocks with immutable
`query_group_offsets`. Every block carries fixed row routers and explicit
`global_key_ids`; it does not choose routes from tensor values. The block path
uses the same `wq/wk/wv` tensors as dense attention and streams the
compiler-fixed key blocks twice: pass one computes
each query's global finite max and lexicographic `(score, -global_index)` hard
winner over enabled rows; pass two computes the global shifted exponential
denominator and the score-surrogate reductions. A dedicated
`ExactHardAttentionBackward` returns `hard_one_hot @ V` exactly, routes the
score/Q/K gradient through the stable-softmax Jacobian, and routes the V/Wv
gradient through the hard one-hot, exactly like dense
`straight_through_selection @ V`. Wrapping `hard_selected_V` and
`soft_weighted_V` in `ExactHardSoftBackward` is explicitly forbidden because
that would incorrectly send a soft, rather than hard, gradient to V/Wv. This
avoids a dense score tensor while preserving dense forward and backward
semantics.

For upstream `G`, hard matrix `H` and soft surrogate `P`, both implementations
must realize the same derivatives:

```text
dV = H^T G
dP = G V^T
dS = (P * (dP - rowsum(P * dP))) / temperature
dQ = dS K
dK = dS^T Q
```

The block kernel accumulates these reductions over immutable key blocks; it
does not materialize a different surrogate.
Block enumeration is structural and immutable, supports leading batch/head
dimensions, and never chooses a block from tensor values.

- [ ] **Step 4: Run GREEN on single, batched and masked fixtures.**

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_attention2d -CTestRegex '^cmz_vm3_attention2d$'
```

- [ ] **Step 5: Commit** as `feat: add batched two-dimensional ST attention`.

### Task 4: Canonical recurrent chess state ABI

**Files:**
- Create: `native/vm3/include/cmz_vm3/chess_state.h`
- Create: `native/vm3/src/chess_state_views.cpp`
- Create: `native/vm3/compiler/initial_state_binder.cpp`
- Create: `native/vm3/tests/test_state_schema.cpp`
- Create: `tests/oracle/test_vm3_state_oracle.py`

**Interfaces:**
- Produces:

```cpp
enum class PieceState : int64_t { Empty, WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK };
enum class Side : int64_t { White, Black };
struct MoveSlot { Tensor move, active; };
struct PolicyKvSlot { Tensor key, value; };
struct TrajectoryTape {
  std::vector<MoveSlot> move_slots;
  std::vector<PolicyKvSlot> policy_kv_slots;
};
struct ChessRingState { Tensor tokens; TrajectoryTape trajectory; Tensor cursor; };
enum class ClaimMode : int64_t { LiteralClaim, AutoClaimSelfPlay };
enum class TerminalKind : int64_t {
  Running, WhiteCheckmated, BlackCheckmated, Stalemate,
  MaterialDeadSubset, Fivefold, SeventyFiveMove,
  ClaimedThreefold, ClaimedFiftyMove, CapacityExhausted
};
enum class GameResult : int64_t { Running, WhiteWin, BlackWin, Draw };
struct InitialPosition { std::array<PieceState,64> board; Side side; uint8_t castling;
                         int64_t ep_square; int64_t halfmove; int64_t fullmove;
                         ClaimMode claim_mode; };
ChessRingState bind_initial_state(const FrozenChessProgram&, const InitialPosition&);
```

The fixed token layout uses 64x13 board one-hots, two side rows, one of 16
castling-right rows, one of 65 raw-EP rows, one of 151 halfmove rows, two
base-128 fullmove digits (maximum accepted value 9526), three base-128
trajectory/cursor digits, claim-mode/claim/terminal/result rows and fixed
workspace. The compiler publishes every offset and width in the manifest;
input and output shapes are identical.
Every semantic alternative occupies a fixed predicate-token row. Column zero is
the active value written by the initialization binder and later frozen matrix
writes; all remaining identity/features and trailing workspace rows come from
the immutable `state_template`. The checksum-loaded `state_layout` tensor lists
the eleven `(offset,width)` core segments and must match the compiled ABI before
binding. The two claim-availability rows are `CLAIM_NOW` and `CLAIM_AFTER`.

- [ ] **Step 1: Write exact one-hot and FEN round-trip oracle tests** covering
all 13 square states, 16 castling combinations, `EP=NONE|square`, both claim
modes, clock bounds, claim/terminal/result defaults and identical input/output
token shapes. The FEN adapter is test-only and never linked to VM3 runtime.

- [ ] **Step 2: Run RED**, expecting missing schema.

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_state_schema -CTestRegex '^cmz_vm3_state_schema$' -PytestArgs 'tests/oracle/test_vm3_state_oracle.py -q'
```

- [ ] **Step 3: Implement initialization-only binding.** All `index_put_` calls
remain in compiler/binder targets and returned rule tensors share storage.

```cpp
ChessRingState bind_initial_state(const FrozenChessProgram& program,
                                  const InitialPosition& input) {
  Tensor tokens = program.tensors.at("state_template").clone();
  bind_board_one_hots(tokens, input.board);
  bind_side_rights_ep_and_clocks(tokens, input);
  bind_claim_mode(tokens, input.claim_mode);
  return {tokens,
          TrajectoryTape{},
          torch::zeros({}, tokens.options())};
}
```

The three `bind_*` helpers compile only into the binder target. Runtime
`chess_state_views.cpp` contains fixed offset/view functions and no FEN, UCI,
piece move or rule logic. A link test proves the executor does not link the
binder.

- [ ] **Step 4: Run GREEN** in CTest and pytest oracle adapter.

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_state_schema -CTestRegex '^cmz_vm3_state_schema$' -PytestArgs 'tests/oracle/test_vm3_state_oracle.py -q'
```

- [ ] **Step 5: Commit** as `feat: define canonical recurrent chess state`.

### Task 5: One-token append-only trajectory protocol

**Files:**
- Create: `native/vm3/include/cmz_vm3/trajectory.h`
- Create: `native/vm3/src/trajectory.cpp`
- Create: `native/vm3/tests/test_trajectory.cpp`

**Interfaces:**
- Produces:

```cpp
struct MoveTokenFields { Tensor source, target, promotion, derived_flags,
                         ply_address, mover_side, post_position_identity; };
MoveSlot make_training_move_slot(const FrozenChessProgram&,
                                 const Tensor& move_token,
                                 const Tensor& commit_predicate);
```

- [ ] **Step 1: Write RED tests** proving one active logical slot per committed
ply, inactive claim/terminal padding slots, unchanged prior slot tensors, device
cursor increment, gradient reachability to every active move slot, no
autograd-visible in-place write, no `cat`, and no full-capacity copy per ply.

- [ ] **Step 2: Run RED.**

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_trajectory -CTestRegex '^cmz_vm3_trajectory$'
```

- [ ] **Step 3: Implement a static functional training tape.** The unroll
builder creates one structural slot per fixed step. Each slot is a new tensor
`active * emitted_move + (1-active) * PAD`. Previous slots remain separate
autograd tensors, so no history tensor is mutated and no full-capacity tensor is
copied. Task 7 adds policy K/V tensors to the matching structural slot. Host
slot order is fixed before CUDA capture and never depends on move/terminal data.

```cpp
MoveSlot make_training_move_slot(const FrozenChessProgram& program,
                                 const Tensor& emitted_move,
                                 const Tensor& commit) {
  const Tensor& pad = program.tensors.at("trajectory_pad_token");
  return {commit * emitted_move + (1.0 - commit) * pad, commit};
}
```

The unroll builder stores this returned tensor in the already assigned
structural vector slot. It never mutates a previous tensor or uses the device
cursor as a host index.

- [ ] **Step 4: Run GREEN, mutation purity and allocation tests.** Require
`O(fixed_steps * d_model)` tape/KV storage before checkpointed activations and
report the expected `O(fixed_steps^2)` exact dense-softmax training work
separately. Device-paged inference append is deferred explicitly to Task 16,
not silently substituted into autograd training.

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_trajectory -CTestRegex '^cmz_vm3_trajectory$' -PytestArgs 'tests/test_vm3_source_purity.py -q'
```

- [ ] **Step 5: Commit** as `feat: add matrix-routed move trajectory`.

### Task 6: Universal fixed candidate bank

**Files:**
- Create: `native/vm3/include/cmz_vm3/candidate_bank.h`
- Create: `native/vm3/compiler/candidate_bank.cpp`
- Create: `native/vm3/tests/test_candidate_bank.cpp`

**Interfaces:**
- Produces compiler-only `CandidateBank compile_candidate_bank()` with 4096
ordinary source/target rows followed by 176 Q/R/B/N promotion rows and one
generic empty-selector sentinel. Runtime rows expose only source, target,
promotion and raw source/target coordinate one-hots. They contain no deltas,
ray/direction kind, piece kind, geometry class, occupancy or legality.

- [x] **Step 1: Write RED tests** for exact move/sentinel counts `4272/4273`,
unique IDs, deterministic ordering, all 44 promotion source/target pairs times
Q/R/B/N, and absence of delta/ray/piece/geometry/occupancy/legality features.

- [x] **Step 2: Run RED.**

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_candidate_bank -CTestRegex '^cmz_vm3_candidate_bank$'
```

- [x] **Step 3: Implement compiler-only enumeration.** Candidate rows encode
only raw identity and coordinate one-hots. The 44 ordinary promotion-square
source/target rows remain present but are made illegal by frozen rule stages
when `promotion=NONE`; the 176 extra rows carry the four explicit choices.

```cpp
for (int source = 0; source < 64; ++source) {
  for (int target = 0; target < 64; ++target) {
    bank.push_back(raw_candidate(source, target, Promotion::None));
  }
}
for (const Side side : {Side::White, Side::Black}) {
  const int source_rank = side == Side::White ? 6 : 1;
  const int target_rank = side == Side::White ? 7 : 0;
  for (int source_file = 0; source_file < 8; ++source_file) {
    for (int target_file = std::max(0, source_file - 1);
         target_file <= std::min(7, source_file + 1); ++target_file) {
      for (const Promotion p : {Promotion::Queen, Promotion::Rook,
                                Promotion::Bishop, Promotion::Knight}) {
        bank.push_back(raw_candidate(square(source_file, source_rank),
                                     square(target_file, target_rank), p));
      }
    }
  }
}
require(bank.size() == kMoveCandidateCount); // 4096 + 44*4
```

This loop is compiler-only. `raw_candidate` writes IDs and coordinate one-hots,
not the result of a geometry predicate.

- [x] **Step 4: Run GREEN and snapshot the candidate manifest.**

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_candidate_bank -CTestRegex '^cmz_vm3_candidate_bank$'
```

- [x] **Step 5: Commit** as `feat: compile universal chess candidate bank`.

### Task 7: Swappable bounded attention-only policy and absolute legal gate

**Files:**
- Create: `native/vm3/include/cmz_vm3/policy.h`
- Create: `native/vm3/src/policy.cpp`
- Create: `native/vm3/tests/test_policy.cpp`

**Interfaces:**
- Consumes: state, active trajectory, 4272 raw candidate descriptors and
rule-emitted move legality, current-claim availability, per-candidate
claim-after witness eligibility and halt-required tokens.
- Produces:

```cpp
struct PolicyInput {
  Tensor state_tokens, candidate_tokens, move_legal;
  Tensor claim_now, claim_after_eligible, auto_claim, halt_required;
  const TrajectoryTape& trajectory;
};
struct PolicyOutput {
  Tensor candidate_logits, candidate_probability; // [B,4272,2], [B,4272]
  Tensor control_logits, control_probability;      // [B,3,2], [B,3]
  Tensor move_selection, intended_move_selection, control_selection;
};
struct BoundedPolicyContract {
  int64_t qk_head_dim = 2;
  double parameter_lower = -1.0, parameter_upper = 1.0;
  double activation_lower = -1.0, activation_upper = 1.0;
  double qk_score_lower = -2.0, qk_score_upper = 2.0;
  double eligibility_margin = 5.0;
  int64_t maximum_fan_in;
  c10::ScalarType accumulator_dtype;
  double certified_accumulator_abs_bound;
  bool coordinatewise_signed_binary_softmax = true;
  bool every_projection_rebounded = true;
  bool local_accept_reject_softmax = true;
};
struct PolicyModule : torch::nn::Module {
    virtual const BoundedPolicyContract& contract() const = 0;
    virtual PolicyOutput forward(const PolicyInput&) = 0;
    virtual PolicyKvSlot encode_move(const MoveSlot&) = 0;
};
struct PolicyPair { PolicyModule& white; PolicyModule& black; };
```

Executor preflight recomputes `maximum_fan_in` from actual parameter shapes and
derives `certified_accumulator_abs_bound = fan_in * 1 * 1` with outward rounding;
it does not trust descriptor numbers. It requires that bound plus all fixed
margins be below one quarter of the accumulator dtype's finite maximum. It also
checks every raw parameter finite before the inference capture/launch. These are
generic policy-artifact checks and do not read board, legality or trajectory
state.

- [x] **Step 1: Write RED tests** for ZeroPolicy first-legal behavior,
candidate-permutation equivariance, parameter count independent of candidate
count, absence of candidate-ID parameters/bias, legal dominance under extreme
raw logits, current and intended-move claims, forced auto-claim, all-illegal
halt routing, gradient reachability to every encoded historical K/V slot and
non-zero candidate/control-logit gradients.
Reject a policy whose descriptor or exported operation manifest lacks bounded
signed-binary parameters/activations, rebounded Q/K/V/output/readout
projections, local two-class readouts, an accumulator proof or internal
attention margin `> 4`;
the executor never accepts an arbitrary unproved `PolicyModule` before capture.

```cpp
candidate_accept_reject = shared_candidate_scorer(...); // [B, 4272, 2]
candidate_p = torch::matmul(
    torch::softmax(candidate_accept_reject, -1),
    frozen_accept_column);                              // [B, 4272]
move_allowed = running * move_legal;
witness_allowed = running * claim_after_eligible;
move_score = running * candidate_p + 2.0 * move_allowed;
witness_score = running * candidate_p + 2.0 * witness_allowed;
move_routes = fixed_matrix_pack(move_score, 4.0 * (1.0 - running * any_legal));
witness_routes = fixed_matrix_pack(witness_score, 4.0 * (1.0 - running * any_witness));
move_eligibility = fixed_matrix_pack(move_allowed, 1.0 - running * any_legal);
witness_eligibility = fixed_matrix_pack(witness_allowed,
                                        1.0 - running * any_witness);
move_selection = deterministic_st_select(move_routes, move_eligibility, tau);
intended_selection = deterministic_st_select(witness_routes,
                                             witness_eligibility, tau);

control_accept_reject = shared_control_scorer(...); // [B, 3, 2]
control_p = torch::matmul(
    torch::softmax(control_accept_reject, -1),
    frozen_accept_column);                           // [B, 3]
control_allowed = fixed_matrix_pack(any_legal, claim_now, any_witness);
control_score = running * control_p + 2.0 * running * control_allowed;
control_score += fixed_auto_claim_bonus(auto_claim, claim_now); // +4 to CLAIM_NOW
control_routes = fixed_matrix_pack(control_score, 6.0 * halt_required);
control_eligibility = fixed_matrix_pack(running * control_allowed,
                                        halt_required);
control_selection = deterministic_st_select(control_routes,
                                             control_eligibility, tau);
```

Every `pack` above is a frozen row/feature projection, not `cat` or indexed
assignment. `shared_candidate_scorer` is the same two-class attention scorer
applied independently to each rule-emitted trace row, and
`shared_control_scorer` does the same over the three typed control rows. No
trainable tensor has a candidate axis, and a disabled route cannot rescale
enabled probabilities through a preliminary cross-route softmax. The
eligibility-weighted ST selector is the only cross-route normalization. A sentinel's selected packet is
fixed zero/identity, including its backward path.
In auto-claim mode `CLAIM_NOW` exceeds every ordinary control score; a true halt
still has the highest margin.

- [x] **Step 2: Run RED.**

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_policy -CTestRegex '^cmz_vm3_policy$'
```

- [x] **Step 3: Implement ZeroPolicy and a minimal multi-head `d_head=2`
trajectory-aware `TransformerPolicy` without MLP or handcrafted evaluation.**
`PolicyPair` evaluates both registered slots and a frozen side-token projection
selects their output; runtime never performs `if (white_to_move)` dispatch.
After a committed move, both fixed `encode_move` slots execute and the same
side-token projection places the active emitter's K/V into the matching
structural tape slot. Training keeps those tensors attached and never uses an
inference cache. Every policy attention call consumes all fixed structural
history slots plus their exact tensor `active` flags; it never materializes or
host-builds a variable-length active list.

```cpp
Tensor TransformerPolicy::shared_candidate_accept_reject_logits(
    const PolicyInput& in) {
  Tensor h = bounded_candidate_tokens_to_queries(in.candidate_tokens);
  h = bounded_policy_attend2d(h, state_tokens_to_kv(in.state_tokens),
                              all_active, bounded_policy_contract);
  h = bounded_policy_attend2d(h,
                              trajectory_fixed_slots_to_kv(in.trajectory),
                              trajectory_active_tokens(in.trajectory),
                              bounded_policy_contract);
  return torch::matmul(h, bounded_parameter(shared_accept_reject_raw_));
  // [B,4272,2], each logit has a certified finite accumulator interval
}

Tensor TransformerPolicy::shared_control_accept_reject_logits(
    const PolicyInput& in) {
  Tensor h = bounded_parameter(control_query_tokens_raw_); // 3 typed rows
  h = bounded_policy_attend2d(h, state_tokens_to_kv(in.state_tokens),
                              all_active, bounded_policy_contract);
  h = bounded_policy_attend2d(h,
                              trajectory_fixed_slots_to_kv(in.trajectory),
                              trajectory_active_tokens(in.trajectory),
                              bounded_policy_contract);
  return torch::matmul(h, bounded_parameter(shared_accept_reject_raw_));
  // [B,3,2], each logit has a certified finite accumulator interval
}
```

`bounded_parameter(raw)` uses a frozen feature projection to form `[raw,0]`, a
stable two-class softmax, and the affine map `2*p-1`; raw parameters are never
matmul operands. `bounded_policy_attend2d` uses bounded operands for every
projection, then independently re-bounds each of the two Q/K coordinates and
every V/output coordinate by the same construction. Thus Q/K/V and the next
activation remain in `[-1,1]`, `base=Q*K^T` remains in `[-2,2]`, and the frozen
margin `5*active` dominates PAD without collapsing the physical 2D geometry.
There is no unbounded residual path. Local ACCEPT/REJECT logits are matmuls of
bounded activations and bounded readout weights and are covered by the fixed
fan-in accumulator proof before their stable two-class softmax. The same
sentinel and eligibility-weighted ST selector from Task 2 is used, with no PAD
filtering.
`encode_move`, learned control/query tokens and policy K/V storage use the same
bounded-parameter and re-bounded-output path; no unbounded side channel enters a
later projection.

Every `bounded_policy_attend2d` head has physical width two. There is no MLP, candidate-ID
embedding, candidate-axis parameter or candidate-specific bias. The test
permutes candidate rows and requires logits/selection to permute identically.
Set every raw parameter to both extreme finite dtype values and run optimizer
steps; tests still require all projection operands/accumulators finite, Q/K/V in
`[-1,1]`, QK scores in `[-2,2]`, an active-row hard winner or, when none is
active, a sentinel hard winner, and zero forward contribution from every PAD
slot. Mutation tests remove each rebounding site or understate maximum fan-in
and require preflight failure. Inference validates once before launch; training
performs a device finite reduction immediately after `optimizer.step()` and a
host-visible generic health check before scheduling the next rollout. If a raw
parameter is non-finite, the next rollout launch counter must remain unchanged.

- [x] **Step 4: Run GREEN and verify only policy parameters require grad.**

Also assert that an illegal or sentinel packet has exactly zero forward effect,
illegal descriptors have zero surrogate transition weight, the explicitly
declared gradient to rule eligibility matches the analytic
eligibility-weighted formula,
post-terminal policy logits have zero gradient, and adding terminal padding
steps leaves all pre-terminal gradients identical.

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_policy -CTestRegex '^cmz_vm3_policy$'
```

- [x] **Step 5: Commit** as `feat: add swappable legal-only ST policy`.

### Task 8: Minimal unified pawn-and-knight closed ring

**Files:**
- Modify: `native/vm3/include/cmz_vm3/program_image.h`
- Create: `native/vm3/compiler/rule_compiler.cpp`
- Create: `native/vm3/compiler/pawn_rules.cpp`
- Create: `native/vm3/compiler/piece_rules.cpp`
- Create: `native/vm3/include/cmz_vm3/recurrent_ring.h`
- Create: `native/vm3/src/recurrent_ring.cpp`
- Create: `native/vm3/tests/test_minimal_ring.cpp`

**Interfaces:**
- Produces:

```cpp
struct StepResult { ChessRingState next; Tensor legal; Tensor selected_move;
                    Tensor terminal; PolicyOutput policy; };
StepResult recurrent_step(const FrozenChessProgram&, PolicyPair&, const ChessRingState&);
```

- [ ] **Step 1: Write RED integration tests** using one shared board and one
candidate bank: pawn and knight legal predicates, policy winner, exact board
write, side flip, one trajectory append and direct second-step consumption.

- [ ] **Step 2: Run RED.**

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_minimal_ring -CTestRegex '^cmz_vm3_minimal_ring$'
```

- [ ] **Step 3: Port accepted pawn/knight relations into one serialized VM3
rule image.** Compiler code emits only tensors/stage metadata. The generic
`recurrent_ring.cpp` executes the image without linking compiler objects.
Remove VM2's separate widths/depths, synthetic legal-set assembly and
compile-time auto/requested policy variants.

```cpp
Tensor execute_stage(const Tensor& x, const FrozenStage& s, double tau) {
  const auto y = block_attention2d(
      x, s.wq, s.wk, s.wv, s.attention_plan, tau).output;
  return x + torch::matmul(torch::matmul(s.row_router, y - x),
                           s.feature_router);
}

StepResult recurrent_step(const FrozenChessProgram& program,
                          PolicyPair& policies,
                          const ChessRingState& state) {
  Tensor x = state.tokens;
  for (const auto& stage : program.pre_policy_stages) {
    x = execute_stage(x, stage, kRuleTemperature);
  }
  const auto policy = run_both_and_side_route(policies, x, state.trajectory);
  x = matrix_write_policy_packets(x, policy, program.tensors);
  for (const auto& stage : program.post_policy_stages) {
    x = execute_stage(x, stage, kRuleTemperature);
  }
  return decode_tensor_views_without_scalar_reads(x, state.trajectory, policy);
}
```

The loops above are fixed structural stage iteration. `decode_tensor_views`
returns tensor slices/views only; it does not read values or branch. The
compiler has its own executable that produces and hashes the stage tensors.

- [ ] **Step 4: Run GREEN** and compare canonical semantic projections with VM2:
legal candidate IDs, selected source/target/promotion, output board and side.
Token bytes are deliberately not compared because VM3 has a different ABI.

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_minimal_ring -CTestRegex '^cmz_vm3_minimal_ring$'
```

- [ ] **Step 5: Commit** as `feat: close unified pawn knight recurrent ring`.

### Task 9: Ordinary pseudo-legal geometry and promotion

**Files:**
- Modify: `native/vm3/compiler/pawn_rules.cpp`
- Modify: `native/vm3/compiler/piece_rules.cpp`
- Create: `native/vm3/tests/test_pseudo_legal.cpp`
- Create: `tests/oracle/test_vm3_pseudo_legal_oracle.py`

**Interfaces:**
- Extends rule candidate traces with frozen predicates for pawns, knights,
bishops, rooks, queens and kings, including slider blocker distance and four
promotion choices.

- [ ] **Step 1: Write RED oracle tests** comparing exact UCI candidate sets to
`python-chess.generate_pseudo_legal_moves()` on curated edges and 512 seeded
legal random-walk positions after clearing castling rights and EP squares.
Filter castling and en-passant from this task's expected set; promotion remains.

- [ ] **Step 2: Run RED** and record missing piece classes.

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_pseudo_legal -CTestRegex '^cmz_vm3_pseudo_legal$' -PytestArgs 'tests/oracle/test_vm3_pseudo_legal_oracle.py -q'
```

- [ ] **Step 3: Implement 2D address/ray lookup heads and frozen conjunction
relations.** Do not encode position-specific legality in candidate tokens.

```cpp
builder.lookup2d("source_piece", raw_source_coordinates, board_square_tokens);
builder.lookup2d("target_piece", raw_target_coordinates, board_square_tokens);
builder.lookup2d("abs_file_delta", raw_source_target_coordinates,
                 file_delta_relation_tokens);
builder.lookup2d("abs_rank_delta", raw_source_target_coordinates,
                 rank_delta_relation_tokens);
builder.lookup2d("ray_membership", raw_source_target_coordinates,
                 ordered_ray_relation_tokens);
builder.reduce_frozen("ray_clear", intervening_square_occupancy,
                      all_empty_relation_tokens);
builder.lookup2d("piece_geometry", derived_relation_tokens,
                 piece_geometry_rule_tokens);
builder.reduce_frozen("ordinary_pseudo_legal",
                      {side_match, piece_geometry, target_allowed, ray_clear,
                       promotion_choice_valid}, conjunction_tokens);
```

Every builder call serializes Q/K/V/mask/write tensors. These names and loops
exist only in `cmz_vm3_compiler`; the executor sees an ordered stage array.

- [ ] **Step 4: Run GREEN**, including exhaustive source/target geometry bases.

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_pseudo_legal -CTestRegex '^cmz_vm3_pseudo_legal$' -PytestArgs 'tests/oracle/test_vm3_pseudo_legal_oracle.py -q'
```

- [ ] **Step 5: Commit** as `feat: compute ordinary pseudo legal trace`.

### Task 10: Special pseudo-moves and exact trial transitions

**Files:**
- Create: `native/vm3/compiler/special_move_rules.cpp`
- Create: `native/vm3/tests/test_special_moves.cpp`
- Modify: `tests/oracle/test_vm3_pseudo_legal_oracle.py`

**Interfaces:**
- Extends pseudo-legal traces and candidate-local trial packets with en-passant
captured square, castling rook source/target, promotion piece, next castling
rights, raw next EP, halfmove and fullmove fields.

- [ ] **Step 1: Write RED sequence tests** for all four castling geometries,
rights lost on king/rook moves and original-rook capture, pseudo-legal en
passant, correct off-target captured-square trial write, quiet/capture clock
updates and all promotions/capture-promotions.

- [ ] **Step 2: Run RED.**

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_special_moves -CTestRegex '^cmz_vm3_special_moves$' -PytestArgs 'tests/oracle/test_vm3_pseudo_legal_oracle.py -q'
```

- [ ] **Step 3: Compile frozen special-effect relation tokens and multi-row
trial writes.** Policy still supplies only source/target/promotion raw identity;
the compiler serializes matrices and the executor remains generic.

```cpp
builder.reduce_frozen("ep_candidate",
                      {pawn_diagonal_geometry, target_empty, ep_target_match,
                       adjacent_opponent_pawn}, conjunction_tokens);
builder.matrix_trial_write("ep_capture", source_square, target_square,
                           ep_captured_square);
builder.reduce_frozen("castle_candidate",
                      {king_source, rook_source, castling_right,
                       between_squares_empty}, conjunction_tokens);
builder.matrix_trial_write("castle", king_source_target, rook_source_target);
builder.matrix_trial_write("promotion", source_target, promotion_piece);
builder.lookup2d("next_metadata", selected_raw_move_and_rule_flags,
                 metadata_transition_tokens);
```

- [ ] **Step 4: Run GREEN** against the complete pseudo-legal oracle, including
castling and en-passant, and assert exact trial board/metadata after every
candidate before own-king filtering.

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_special_moves -CTestRegex '^cmz_vm3_special_moves$' -PytestArgs 'tests/oracle/test_vm3_pseudo_legal_oracle.py -q'
```

- [ ] **Step 5: Commit** as `feat: compile special move trial transitions`.

### Task 11: Attack map, own-king safety and final legal set

**Files:**
- Create: `native/vm3/compiler/attack_rules.cpp`
- Create: `native/vm3/tests/test_king_safety.cpp`
- Create: `tests/oracle/test_vm3_legal_oracle.py`

**Interfaces:**
- Consumes every ordinary/special `TRIAL_BOARD`; produces `SIDE_IN_CHECK`,
`OWN_KING_ATTACKED` and final `LEGAL` tokens for all candidates.

- [ ] **Step 1: Write RED tests** for pins, discovered checks, double check,
king adjacency, blocked sliders, en-passant discovered check, illegal castle
from/through/into check, and legal special-move controls.

- [ ] **Step 2: Run RED.**

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_king_safety -CTestRegex '^cmz_vm3_king_safety$' -PytestArgs 'tests/oracle/test_vm3_legal_oracle.py -q'
```

- [ ] **Step 3: Compile common frozen attack heads and legality conjunction over
trial boards.** There is no piece-specific runtime dispatch; the same executor
applies serialized stage tensors to every candidate batch.

```cpp
builder.lookup2d("trial_king_square", trial_board_tokens, king_identity_tokens);
builder.lookup2d("trial_attack_geometry", attacker_and_target_coordinates,
                 attack_relation_tokens);
builder.reduce_frozen("trial_ray_clear", trial_intervening_occupancy,
                      all_empty_relation_tokens);
builder.reduce_frozen("own_king_attacked",
                      {opponent_piece, attack_geometry, attack_path_clear},
                      any_true_relation_tokens);
builder.reduce_frozen("legal",
                      {pseudo_legal, not_own_king_attacked,
                       castle_origin_transit_destination_safe},
                      conjunction_tokens);
```

- [ ] **Step 4: Run GREEN** against exact `python-chess.legal_moves` sets,
start-position perft depths 1-4, and committed Kiwipete, en-passant, promotion
and castling positions at depths 1-3. Store every FEN, expected and observed
count; a timeout fails the gate rather than silently shrinking it.

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_king_safety -CTestRegex '^cmz_vm3_king_safety$' -PytestArgs 'tests/oracle/test_vm3_legal_oracle.py -q'
```

- [ ] **Step 5: Commit** as `feat: filter every trial move by king safety`.

### Task 12: Repetition, clocks, claims and absorbing terminal state

**Files:**
- Create: `native/vm3/compiler/terminal_rules.cpp`
- Create: `native/vm3/tests/test_terminal.cpp`
- Create: `tests/oracle/test_vm3_terminal_oracle.py`

**Interfaces:**
- Produces `CLAIM_NOW`, per-candidate `CLAIM_AFTER_INTENDED_MOVE`,
`AUTO_CLAIM_ALLOWED`, `TERMINAL_KIND`, `RESULT`, `COMMIT_MOVE` and
`HALT_REQUIRED` predicate tokens.

- [ ] **Step 1: Write RED sequence tests** for root-position occurrence,
threefold claim now, intended legal move producing the third occurrence,
50-move claim now, intended non-pawn/non-capture producing halfmove 100, forced
self-play auto-claim, fivefold automatic draw, halfmove 99/100/149/150, mate
priority over the 150th halfmove, stalemate, explicit `MATERIAL_DEAD_SUBSET`
fixtures and post-terminal absorption. An intended claim never commits its
witness move or appends a MOVE token.

- [ ] **Step 2: Run RED.**

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_terminal -CTestRegex '^cmz_vm3_terminal$' -PytestArgs 'tests/oracle/test_vm3_terminal_oracle.py -q'
```

- [ ] **Step 3: Implement exact post-position identity in each move token.**
Compare board, side, rights and legally effective EP without collision-prone
hashes; count matches via fixed matrix reductions and relation heads. Initial
state contributes occurrence one and clocks never enter the identity.

```cpp
builder.reduce_frozen("position_equal",
                      {all_64_piece_rows_equal, side_equal, rights_equal,
                       effective_ep_equal}, conjunction_tokens);
builder.reduce_frozen("repetition_count", prior_post_position_identities,
                      exact_radix_counter_tokens);
builder.reduce_frozen("claim_now",
                      {repetition_at_least_3, halfmove_at_least_100},
                      any_true_relation_tokens);
builder.reduce_frozen("claim_after_intended_move",
                      {trial_repetition_at_least_3,
                       trial_halfmove_at_least_100},
                      any_true_relation_tokens);
```

- [ ] **Step 4: Implement terminal priority and identity route.** Compute the
current position's no-legal/check predicates before policy routing; route mate
or stalemate without selecting an illegal move. `HALT_IDENTITY` preserves
board/meta/history/cursor, appends no token and passes state gradients as an
identity. Name any accepted material-only circuit `MATERIAL_DEAD_SUBSET`; never
serialize or publish it as general `DEAD_POSITION`.

```text
priority 0: existing HALT -> HALT_IDENTITY
priority 1: no LEGAL and SIDE_IN_CHECK -> CHECKMATE
priority 2: no LEGAL and not SIDE_IN_CHECK -> STALEMATE
priority 3: MATERIAL_DEAD_SUBSET
priority 4: repetition_count >= 5
priority 5: halfmove >= 150
priority 6: policy-controlled current/intended claim
priority 7: COMMIT_MOVE
```

The priority is compiled as relation tokens plus fixed dominant score margins;
it is not a C++ switch in the executor.

- [ ] **Step 5: Run GREEN** against oracle sequence outcomes, literal/auto-claim
mode distinctions and byte-identical absorbing states. Record material-subset
coverage separately so it cannot unlock a general FIDE dead-position claim.

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_terminal -CTestRegex '^cmz_vm3_terminal$' -PytestArgs 'tests/oracle/test_vm3_terminal_oracle.py -q'
```

- [ ] **Step 6: Commit** as `feat: compute exact terminal state inside ring`.

### Task 13: Fixed whole-game rollout with no semantic host interaction

**Files:**
- Modify: `native/vm3/include/cmz_vm3/recurrent_ring.h`
- Modify: `native/vm3/src/recurrent_ring.cpp`
- Create: `native/vm3/tests/test_full_ring.cpp`

**Interfaces:**
- Produces:

```cpp
struct RolloutResult { ChessRingState final_state; Tensor state_trace;
                       Tensor policy_trace; Tensor terminal_loss_inputs;
                       Tensor committed_move_count; };
RolloutResult rollout_fixed(const FrozenChessProgram&, PolicyPair&,
                            ChessRingState initial, int64_t fixed_steps);
```

- [ ] **Step 1: Write RED tests** that reject `.item`, host-selected moves,
early terminal breaks and per-ply binders; compare multi-ply state/trajectory
bytes with oracle while running beyond terminal. Use `fixed_steps`, not
`max_plies`: terminal adjudication may consume a no-commit step after the final
move.

The tests also prove the conservative bound from `96` maximum pawn moves plus
`30` maximum captures and reject treating `CAPACITY_EXHAUSTED` as mate, draw or
a usable terminal-training result.

- [ ] **Step 2: Run RED.**

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_full_ring -CTestRegex '^cmz_vm3_full_ring$'
```

- [ ] **Step 3: Implement a structural fixed loop over the identical recurrent
step.** `HALT` is handled only by the internal absorbing selector.

```cpp
RolloutResult rollout_fixed(const FrozenChessProgram& program,
                            PolicyPair& policies,
                            ChessRingState state,
                            int64_t fixed_steps) {
  RolloutResult out;
  for (int64_t structural_step = 0; structural_step < fixed_steps;
       ++structural_step) {
    auto step = recurrent_step(program, policies, state);
    retain_training_views(out, step, structural_step);
    state = std::move(step.next);
  }
  out.final_state = std::move(state);
  return out;
}
```

`structural_step` addresses prebuilt tape slots only. The loop has no tensor
read, early break, move decode or binder call and is replaced by the one-launch
device schedule at Task 15.

- [ ] **Step 4: Run GREEN** on curated games ending by every implemented
terminal kind, one deliberately truncated short horizon, and the full
19,051-step structural capacity with an early absorbing terminal fixture.

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_full_ring -CTestRegex '^cmz_vm3_full_ring$'
```

- [ ] **Step 5: Commit** as `feat: unroll declared chess ring without host semantics`.

### Task 14: One-loss full-game gradient-transport harness

**Files:**
- Create: `native/vm3/src/trainer.cpp`
- Create: `native/vm3/tests/test_full_bptt.cpp`
- Modify: `native/vm3/CMakeLists.txt`

**Interfaces:**
- Produces `train_rollout(...)` over a `PolicyPair`. Tensor side routing runs
both fixed policy slots without a semantic host branch. Each update trains one
side while freezing the opponent's parameters but retaining the opponent
activation path with respect to recurrent state.

- [ ] **Step 1: Write RED tests** named:

```text
two_ply_terminal_loss_reaches_ply_zero_logits
terminal_loss_reaches_first_middle_and_last_retained_policy_logits
frozen_rule_weights_have_no_grad_but_state_does
trajectory_remains_attached
one_backward_and_one_optimizer_step_per_rollout
capacity_exhausted_skips_backward_and_optimizer
```

- [ ] **Step 2: Run RED.**

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_full_bptt -CTestRegex '^cmz_vm3_full_bptt$'
```

- [ ] **Step 3: Implement the exact alternating pathwise losses from
rule-emitted result one-hot.** For a White update use
`loss = -(result[WHITE_WIN] - result[BLACK_WIN])`; for a Black update use its
negative. Call one rollout, one `backward()` and one optimizer step only for the
active side. A shared-policy mode is allowed solely as a gradient experiment
and is not called a correct zero-sum self-play objective. Do not introduce
material values or external labels; entropy may cover only rule-legal traces.

```cpp
const Tensor white_utility = result.select(-1, WHITE_WIN)
                           - result.select(-1, BLACK_WIN);
const Tensor loss = train_side == TrainingSide::White
                  ? -white_utility
                  :  white_utility; // train_side is an initialization command,
                                    // never a tensor-derived runtime branch
if (rollout_status == HostVisibleStatus::CapacityExhausted) {
  return TrainReport::without_update(); // read only after rollout completion
}
optimizer.zero_grad();
loss.backward();
optimizer.step();
```

The two host branches above happen only after the captured rollout has returned
and select the requested optimizer/update command, never a chess move or tensor
transition. The native test uses counters to prove one backward and one step.

- [ ] **Step 4: Run GREEN** and assert all gradients are finite and repeatable.

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_full_bptt -CTestRegex '^cmz_vm3_full_bptt$'
```

- [ ] **Step 5: Commit** as `feat: prove terminal gradient transport through full game`.

### Task 15: Deterministic GPU residency and activation recomputation

**Files:**
- Create: `native/vm3/tests/test_gpu_rollout.cpp`
- Create: `native/vm3/src/gpu_schedule.cpp`
- Modify: `native/vm3/src/trainer.cpp`
- Create: `test_results/vm3_gpu_residency_2026-08-14.md`

**Interfaces:**
- Adds a preinitialized one-launch device schedule for a declared fixed horizon
and segment checkpoint/recompute mode without changing `rollout_fixed`
semantics. CUDA Graph, conditional graph nodes or a persistent executor are
acceptable only if the profiler proves the same no-host interval.

- [ ] **Step 1: Write RED GPU tests** for same-device bitwise forward and
gradient repeatability, no D2H synchronization between plies, no detached
checkpoint boundary, identical checkpointed/uncheckpointed gradients, exactly
one host launch for the captured rollout and fail-closed capacity exhaustion.

- [ ] **Step 2: Run RED on the declared CUDA profile.**

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_gpu_rollout -CTestRegex '^cmz_vm3_gpu_rollout$' -WithGpu
```

- [ ] **Step 3: Enable deterministic algorithms and construct the fixed device
schedule after all buffers are allocated.** Use recomputation segments; never
truncate BPTT. Record graph construction time/node count/memory for the chosen
horizon. Do not claim the 19,051-step universal bound unless that exact schedule
is built and profiled; bounded training graphs retain the explicit capacity
status.

```cpp
struct CheckpointSegments { int64_t steps_per_segment; };
struct CapturedRollout {
  at::cuda::CUDAGraph graph;
  ChessRingState static_input;
  RolloutResult static_output;
  int64_t fixed_steps;
};
CapturedRollout capture_rollout(const FrozenChessProgram&, PolicyPair&,
                                int64_t fixed_steps,
                                CheckpointSegments segments);
```

- [ ] **Step 4: Run GREEN and collect Nsight/PyTorch profiler evidence.**

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_gpu_rollout -CTestRegex '^cmz_vm3_gpu_rollout$' -WithGpu
```

- [ ] **Step 5: Commit** as `perf: capture deterministic gpu chess rollout`.

### Task 16: Exact 2D convex-hull KV inference backend

**Files:**
- Create: `native/vm3/src/hull_kv2d.cpp`
- Create: `native/vm3/tests/test_hull_kv2d.cpp`
- Modify: `native/vm3/include/cmz_vm3/attention2d.h`
- Create: `test_results/vm3_hull_kv2d_2026-08-14.md`

**Interfaces:**
- Produces `InferenceTrajectoryPages`: device-resident raw move-token pages,
active length, per-head `d_head=2` K/V pages and immutable-route hull metadata.
Commit appends exactly one logical move entirely on device; claim/halt append
none. Training continues to use the functional tape and dense/block
stable-softmax backward.

- [ ] **Step 1: Write RED equivalence tests** over adversarial ties, duplicate
keys, collinear faces, `q=(0,0)`, active-prefix boundaries, every immutable
route class, paged appends and random traces: dense and hull winner/output must
match exactly with lowest-original-index tie resolution. The raw logical move
sequence must match the training tape for the same hard-forward rollout.

- [ ] **Step 2: Run RED.**

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_hull_kv2d -CTestRegex '^cmz_vm3_hull_kv2d$' -WithGpu
```

- [ ] **Step 3: Implement device hull pages and logarithmic support queries.**
Only pure two-term dot-product heads are eligible. Compile fixed masks into
separate immutable route-class hulls; store minimum original index metadata for
duplicate vertices and collinear support faces; route zero query to the minimum
active index. No scalar winner may cross to the CPU.

```cpp
struct HullQueryResult { Tensor hard_index, selected_value; };
struct InferenceTrajectoryPages {
  Tensor move_pages, key_pages, value_pages, active_length;
  std::vector<DeviceHullRouteClass> hulls;
};
void append_inference_move(InferenceTrajectoryPages&, const Tensor& move,
                           const PolicyKvSlot&, const Tensor& commit);
HullQueryResult query_hull2d(const InferenceTrajectoryPages&,
                             const Tensor& query,
                             int64_t immutable_route_class);
```

- [ ] **Step 4: Run GREEN with TF32 disabled and identical explicit
`q0*k0 + q1*k1` ordering, then benchmark 1K, 10K, 100K and 1M synthetic move
tokens.** Report append, hull maintenance/build, query and end-to-end latency,
memory, dtype and hardware separately; do not extrapolate.

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_test_hull_kv2d -CTestRegex '^cmz_vm3_hull_kv2d$' -WithGpu
```

- [ ] **Step 5: Commit** as `perf: add exact two-dimensional hull lookup`.

### Task 17: Oracle corpus, exhaustive local truth tables and claim acceptance

**Files:**
- Expand: `tests/oracle/test_vm3_*.py`
- Expand: `tests/test_vm3_source_purity.py`
- Create: `test_results/vm3_full_chess_acceptance_2026-08-14.md`
- Modify: `.github/workflows/pages.yml`
- Modify: `.github/workflows/vm3.yml`

**Interfaces:**
- Produces the evidence gate required before the site may say “complete
orthodox move legality and the declared v10 terminal set” or “end-to-end
differentiable”. It must continue to exclude general FIDE dead-position unless
that separate proof gate exists.

- [ ] **Step 1: Add exhaustive finite truth tables** for coordinate geometry,
piece/side/occupancy relations, predicate conjunctions, write routing, counters
and selector ties. Add a deterministic oracle-validated position/sequence corpus
for composed chess behavior; persist seeds, FENs and mismatches. Do not call the
corpus a formal proof of every reachable chess position.

- [ ] **Step 2: Add mutation fixtures** for state detach, integer legality,
host terminal branch, dynamic mask, compiler linkage, candidate delta/geometry
input labels, candidate-specific trainable weights/flat head, literal
floating-point detach ST, CPU sync, per-ply optimizer, frontend rule logic and
FEN/UCI move application in any host orchestrator.

- [ ] **Step 3: Run full native CTest, pytest oracle, warning-as-error, purity,
gradient and GPU gates.**

```powershell
powershell -File tools/vm3-ci.ps1 -Target all -CTestRegex '^cmz_vm3_' -PytestArgs 'tests/oracle tests/test_vm3_source_purity.py -q' -WithGpu
```

- [ ] **Step 4: Record exact pass counts, timings, device and limitations.**

The Pages workflow consumes only a committed trace artifact whose program hash
and evidence-gate ID appear in the acceptance report. CPU GitHub-hosted jobs may
validate schema/site behavior; GPU-residency or HullKV claims require the
matching self-hosted GPU job and cannot be marked successful by substitution.

- [ ] **Step 5: Commit** as `test: validate full differentiable chess ring corpus`.

### Task 18: Native forward/backward trace debugger and migration

**Files:**
- Create: `native/vm3/include/cmz_vm3/trace_exporter.h`
- Create: `native/vm3/src/trace_exporter.cpp`
- Create: `native/vm3/tools/export_ring_trace.cpp`
- Modify: `site/src/TraceDebugger.tsx`
- Modify: `site/src/trace-schema.ts`
- Modify: `site/src/styles.css`
- Modify: `tests/test_vm3_public_evidence_gate.py`
- Modify: `README.md`, `docs/project_memory.md`, `docs/change_history.md`

**Interfaces:**
- Produces a versioned native artifact containing state, trajectory move,
candidate predicates, policy logits/probabilities, exact `X/Wq/Wk/Wv/Q/K/S/M`,
stable-softmax, hard/ST selection, `AV/Y`, every row/feature matrix write,
terminal predicates, selected-cell dot-product arithmetic and native backward
derivatives through Q/K/V to retained early-ply policy logits.

- [ ] **Step 1: Write RED schema/frontend tests** proving every displayed
number and cell arithmetic term comes from the artifact and no TypeScript
rule, winner, softmax, matrix product or gradient calculation exists. Missing
native provenance/version/checksum makes the viewer fail closed.

- [ ] **Step 2: Run RED.**

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_export_ring_trace -CTestRegex '^cmz_vm3_trace_schema$' -PytestArgs 'tests/test_vm3_public_evidence_gate.py -q'
npm --prefix site test
```

- [ ] **Step 3: Export a passing VM3 trace and implement synchronized board,
token, matrix and backward-flow animation.** Keep VM2 pawn trace selectable as
historical evidence.

```ts
type NativeTensor = {
  dtype: "float64" | "float32";
  shape: number[];
  data: number[];
  provenance: "native";
};
type Vm3Trace = {
  schema: "cmz-vm3-trace-v1";
  program_sha256: string; // schema test requires /^[0-9a-f]{64}$/
  evidence_gate: string;
  step: number;
  stage: number;
  tensors: Record<"X"|"Wq"|"Wk"|"Wv"|"Q"|"K"|"S"|"M"|
                  "soft"|"hard"|"AV"|"Y", NativeTensor>;
  writes: NativeTensor[];
  selected_cell_arithmetic: NativeTensor[];
  backward: Record<"dQ"|"dK"|"dV"|"early_policy_logit_grad", NativeTensor>;
};
```

The exporter writes actual native numeric arrays, dtype, shape and artifact
offsets. The frontend only indexes those arrays and animates provenance links.

- [ ] **Step 4: Run native artifact validation, frontend tests/build, desktop
and mobile browser checks, then Pages evidence workflow.**

```powershell
powershell -File tools/vm3-ci.ps1 -Target cmz_vm3_export_ring_trace -CTestRegex '^cmz_vm3_trace_schema$' -PytestArgs 'tests/test_vm3_public_evidence_gate.py -q'
npm --prefix site test
npm --prefix site run build
```

- [ ] **Step 5: Switch README/site claims only to the highest passing gate and
commit** as `docs: publish differentiable chess ring evidence`.

## Final acceptance checklist

- [ ] One immutable rule image executes every position and ply.
- [ ] Exact complete legal sets and transitions match the test oracle.
- [ ] Promotion, en passant, castling, own-king safety and all declared terminal
  reasons pass exact sequence tests.
- [ ] General FIDE dead-position is either backed by a separate complete proof
  gate or explicitly excluded; material-only tests never unlock that claim.
- [ ] Output state ABI equals input ABI and directly feeds the next step.
- [ ] Exactly one move token is appended per committed move.
- [ ] Terminal padding preserves state bytes, adds zero policy/history gradient,
  and passes incoming state gradient through the identity route unchanged.
- [ ] Final loss reaches first-ply policy logits through one whole-game graph.
- [ ] No recurrent activation detach or semantic CPU interaction occurs.
- [ ] Every lookup head is physically two-dimensional.
- [ ] Dense and HullKV hard forward agree exactly.
- [ ] Softmax training cost and HullKV inference cost are reported separately.
- [ ] Site displays only validated native artifacts and honest claim boundaries.
