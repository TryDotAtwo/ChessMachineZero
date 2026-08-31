# Audit correction evidence — 2026-08-31

Status: integration in progress; only completed checks below are evidence. Source worktree: `codex/pure-frozen-transformer-vm-clean`. Original audited commit: `f7165cd`; historical findings: [audit_2026-08-31.md](audit_2026-08-31.md).

## Python compiler/reference corrections

Commit `5202cbd` fixes chronological addressing, attention surrogate parity, exact hard-forward STE, FP4 finite saturation, stable collinear HullKV ties and development-oracle terminal absorption. Independent task review: Approved, no Critical/Important findings. Optional test-memory observation: the400-ply helper retains~401MiB of source snapshots.

| Regression | RED before fix | GREEN after fix |
| --- | --- | --- |
| Full legal-history boards and materialized score margin |2failed/2passed; promotion prefix283 wrong | focused8passed; old-epsilon mutation still reproduces bad full boards at283,399,400 |
| Selected-top-k Q/K/V reference attention |5failed; literal dQ difference13.62329 and unstable candidate ties | independent selected-softmax derivatives, k1/2/3, batching, subsets and vector values pass |
| Standalone hardmax/FP4 STE |8failed/9passed; nonexact binary output and extreme FP4 saturation wrong |17passed |
| Collinear/duplicate HullKV ties |3failed/4passed |9passed with addressing tests |
| Development transition oracle |8failed/4passed; fractional rows accepted and terminal histories resumed |12passed; whole canonical terminal context remains unchanged |
| Artifact hardmax dispatch |6failed/1passed; empty mask and invalid temperatures accepted |40passed with standalone/reference attention coverage |

Controller's fresh full command after commit and review:

```powershell
python -B -m pytest -o addopts= -q -p no:cacheprovider --rootdir=. --confcutdir=. tests
```

Result: **123 passed in13.29s**, exit0. This is a CPU/reference gate, not native GPU acceptance. Generator `python -m vm_compiler.site_trace` regenerated the numeric export; complete exported matrices are compared with fresh reference execution in the test suite.

Measured materialized FP32 invariants over64queries×64addresses×401timestamps:

- Minimum exact-square vs latest different-square margin: `5.340576171875e-05`.
- Minimum equal-square chronological score step: `4.76837158203125e-07`.
- Compiler epsilon now `2**-21`; previous `1e-6` violated square priority on long histories.

Backward reference now follows native's intended contract: hard winning V forward, selected-top-k softmax surrogate for Q/K/V backward. Literal audited case yields dQx≈−1.9661194 and dV≈[.7310586,.2689414,0]. These are first-order surrogate checks, not a true derivative of chess or a learning-benefit result.

## Native build diagnosis (baseline only)

Freshly compiled current `src/ste.cpp` and `tests/native/ste_test.cpp` objects were linked two ways with the same Torch/CUDA library list:

| Link variant | Process exit |
| --- | --- |
| Without CUDA-registration anchor |−1073740791 |
| With `/INCLUDE:?warp_size@cuda@at@@YAHXZ` |0 |

This isolates the prior STE failure to dropped CUDA registration in the executable link. Installed Torch2.8's Windows extension linker uses the same anchor. Output directory: ignored `build/ste-direct-baseline/`. This diagnostic baseline does not replace the new native regression gate.

Verified environment: Windows, Python3.11.5, PyTorch2.8.0+cu128, NumPy2.2.6, pytest8.4.1, chess1.11.2, MSVC19.44/toolset14.44.35207, Windows SDK10.0.26100.0, CUDA toolkit12.5/nvcc12.5.82, RTX3070Laptop sm86/8GiB, driver572.70.

The direct PowerShell MSVC/link path succeeded. A Torch cpp_extension/Ninja attempt compiled objects but hung after its compiler children exited; it was interrupted and is not counted as accepted build evidence. Git Bash also fails in this sandbox creating a signal pipe (Win32error5). No cached executable is reported as a fresh build.

## Native correction acceptance

Commit `b3ba731`: symbolic shapes and candidate metadata are checked before device access; hardmax rejects empty mask rows asynchronously; direct attention guards invalid indices before dereference; FP4 finite extremes saturate. The runtime remains generic tensor C++/CUDA, without chess-aware branches.

The fresh direct Windows build passed **15/15 checked native processes**: 18 malformed-graph cases, all five independent Q/K/V derivative/isolation variants, STE and generic recurrent invariants, isolated invalid-mask/index cases, and **52/52 exact complete boards**. FP32 board execution explicitly disables TF32. Full commands, RED/GREEN results, fixture hashes and environment are recorded in [task2_native_2026-08-31.md](task2_native_2026-08-31.md). Independent task review: **Approved**, no findings.

Controller verification on the same unchanged-source binaries:

- Final complete ordinary rerun: **15/15 processes passed**, including all 52 exact boards. SHA-256 checks confirmed every native build source still matched the fresh-build manifest before execution. The generated local log is `build/task2-green/controller-final.log`.

- Ordinary attention rerun: all five exact Q/K/V and batch-isolation variants passed, exit 0.
- Compute Sanitizer 2024.2, `--tool memcheck --error-exitcode 99`, valid attention suite: exit 0, **ERROR SUMMARY: 0 errors**.
- The same memcheck on the position executable and all 52 fixture states: exit 0, **ERROR SUMMARY: 0 errors**. Target exit-code checking and CUDA-initialization requirement were enabled.
- Initial sandbox instrumentation attempts failed to launch the target (exit 13); the subsequently authorized instrumentation runs above are the acceptance evidence. No kernel/test result is inferred from the failed launches.

The PowerShell fresh-build route is verified; a successful CMake/Ninja build is not. The position acceptance is forward-only and replaces a previous weak nonzero-only position-gradient check. **Integrated position backward is not covered**; primitive derivatives and generic recurrent checks are separate. These checks do not establish useful learning through a full game.

## Site QA baseline

Local URL `http://127.0.0.1:8765/` before UI corrections reproduced the audit's seven-move JS fixture alongside a different three-move matrix export. The visible page mixed English and Russian and advertised Run artifact/verified/68tests. Browser error/warning log was empty: absence of console errors did not establish truthful content. Final UI checks are pending.

## Remaining acceptance

- Whole-page RU/EN single-fixture inspector and arbitrary-cell/producer navigation.
- Final integrated suite, independent whole-diff review, browser screenshots/mobile/console checks.
- Fast-forward publication and live Pages verification.

Full LEGAL_SET/status/recurrent chess, lower-precision execution equivalence, full native-intermediate parity, training benefit and performance superiority remain outside this correction's demonstrated scope.
