# Audit correction evidence — 2026-08-31

Status: approved corrections complete and published; local and live acceptance passed for site/code revision `721db71`. Source worktree: `codex/pure-frozen-transformer-vm-clean`. Original audited commit: `f7165cd`; historical findings: [audit_2026-08-31.md](audit_2026-08-31.md).

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

Local URL `http://127.0.0.1:8765/` before UI corrections reproduced the audit's seven-move JS fixture alongside a different three-move matrix export. The visible page mixed English and Russian and advertised Run artifact/verified/68tests. Browser error/warning log was empty: absence of console errors did not establish truthful content.

## Final site and integrated acceptance

Site commits: `58284fa`, failure-state language follow-up `baf94a5`, readiness-lifecycle correction `102e599`, and cache-version correction `721db71`. The page uses one generated reference trace, not an independent JavaScript chess replay. All 24 frozen tensors, 46 SSA values and two derived attention matrices have bilingual purpose/axis metadata and an arbitrary-coordinate reader. Producer navigation and scalar arithmetic remain available; technical IDs are secondary mappings.

The validator rejects missing/malformed topology, COO, fixture or arithmetic before rendering. The dedicated negative regression suite observed **124 failed / 3 passed before correction**, then **127 passed**. It checks generic FP32 arithmetic consistency, not independent chess legality. Hash fields are not cryptographic authentication, and metadata validation does not prove the natural-language descriptions truthful.

Controller's fresh final command after all source changes:

```powershell
python -B -m pytest -q -p no:cacheprovider
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
foreach ($jsFile in @('site/app.js', 'site/i18n.js', 'site/matrix_inspector.js', 'site/trace_model.js')) {
    node --check $jsFile
    if ($LASTEXITCODE -ne 0) { throw "JavaScript syntax failed: $jsFile" }
}
git diff --check
```

Final result after `721db71`: **257 passed in 20.92s**, all four individual JS syntax checks passed, diff check passed. Earlier gates: `102e599` 255 passed in13.44s; `baf94a5` 255 passed in17.36s. Native sources were unchanged since the fresh build and controller native/memcheck acceptance above; no cached executable was substituted for a rebuild of changed source.

Controller browser checks on the final local site:

| Surface | Observed result |
| --- | --- |
| Every operation | All 45 reachable and explained at 1536 px desktop and 390×844 mobile; no matrix-flow or attention-stage overflow; minimum matrix-card widths 438 px desktop / 312 px mobile |
| GEMM hover | Real pointer hover selects only the left row and right column, one output cell, and its dot-product terms |
| Copy and hardmax | Copy selects contributing cells only; hardmax shows centered ARGMAX, eligible row and winning cell, without false full-column highlighting |
| Final attention | Three explicit stages: real Q×Kᵀ with Kᵀ shape 2×2064; scores→ARGMAX→A; A×V→board |
| Arbitrary coordinates | Input, frozen, SSA, scores and hard-attention values read exactly; fractional coordinates rejected; producer jump preserves selected coordinates |
| Chronological precision | Scores [0,28,466]=2.0664076805114746 and [0,28,465]=2.0664072036743164; selected attention [0,28,466]=1; output [0,28,96]=1 |
| Fixture/board | Exactly e2e4,d7d5,e4d5; board decoded only from v45 has 64 cells, d5 white pawn and e2/e4/d7 empty |
| RU/EN | Page, board descriptions, controls and explanations switch; mobile language controls fit; screenshots visually inspected |
| Ready-state regression | Final review reproduced ready→Loading after repeated language handlers; explicit loading/ready/error ownership in102e599 now preserves initial readiness and RU→EN→RU readiness in the actual browser |
| Invalid export | Separate local HTTP fixture removes values.v1: zero board cells, zero operation/matrix options, explicit validation failure; RU→EN→RU still switches the localized error prefix |
| Console | No error/warning entries on valid or intentionally invalid export pages |

The negative browser harness is a local ignored development fixture, not production fallback logic. Screenshots were inspected in the browser session; no screenshot files are claimed as committed evidence. A repeated linear COO lookup for late hard-attention queries remains an optional UI optimization; no responsiveness benchmark is claimed.

Published-data identity for this source revision:

- Executor: `vm_compiler.reference_executor`; dtype: `float32`.
- Artifact SHA-256: `b2405abab2073dbd54d4879495563c8570cc12a846a470fe59449124d48414c6` (also the native fixture artifact).
- Final export source SHA-256: `01d1e09b4a3cc927c4280718595af6c5ed58d1fd4c519a8f7a78c3d4fcb5dae0`; prior pre-cache-fix export was `36a8ebd35a9e232fd57be2bcc6da967b299e518256939e19f66c1bf73a7c9398`. The change reflects exporter source changes, not altered chess arithmetic.
- Fixture triples: `[[52,54,96],[47,45,102],[54,45,96]]`.

## Release review and publication

Independent core review through `d31c8fb`: Approved, no Critical/Important or new Minor findings. Independent site review through `58284fa` found no new blockers; the tracked failure-state language issue is fixed in `baf94a5`. Final integration review then found a real readiness-label regression; `102e599` closes it. Both independent Task3 and final integration addendum reviews are **Approved**, no remaining Critical/Important findings.

The first ordinary atomic fast-forward publication to main and the clean branch reached `a657fa6`. [Pages run33355778653](https://github.com/TryDotAtwo/ChessMachineZero/actions/runs/33355778653) succeeded, **but live browser acceptance failed**: new HTML loaded an obsolete cached app.js (old line72 missing-button exception; current app was13lines), leaving the board and matrix selector empty. Workflow success alone was not treated as completion.

Correction `721db71` fingerprints CSS, all four JS files and numeric JSON using normalized-content SHA-256 query versions. App fetch and download use the same versioned link; the existing export generator refreshes references after JSON generation. Tests verify content changes alter only the affected version, LF/CRLF parity, all actual published links matching their files, and the app requesting its declared download URL. Focused cache2/2 and site-contract19/19 passed; final full257 gate is above. An independent bounded review approved the correction with no findings. Local browser loaded all six versioned resources and showed ready/45operations/72matrices.

Final ordinary atomic fast-forward push to both agreed refs published `721db7129a176890d8b4b530aa8de6b75581adb0`. [Pages run33356317808](https://github.com/TryDotAtwo/ChessMachineZero/actions/runs/33356317808) completed successfully for that exact revision. Controller checked [the public site](https://trydotatwo.github.io/ChessMachineZero/) after deployment:

- The previously broken cached tab recovered after ordinary reload. All six content-versioned dependency links were present; source hash matched `01d1e09b4a3cc927c4280718595af6c5ed58d1fd4c519a8f7a78c3d4fcb5dae0`.
- Fresh tab at1536px: all45 operations reachable, each with contributing matrix cards and nonempty cell explanation; no matrix-flow overflow. Final attention had three distinct stages.
- Exactly72 tensor choices,64 board cells and the same three fixture moves; d5 is a white pawn. Live score[0,28,466]=2.0664076805114746 and v45[0,28,96]=1; producer navigation opens operation45 with row28/column96 intact.
- RU→EN→RU retained readiness; live GEMM cell expansion displayed its full reduction. Final screenshots were visually inspected. Fresh-tab console error/warning log was empty; the old tab's retained historical pre-fix error is not presented as a new-release error.
- Own temporary local HTTP servers were stopped after verification. The dirty legacy root checkout was not modified.

The subsequent report/checklist commit changes documentation only; the deployed `site/` bytes remain those of `721db71`.

Full LEGAL_SET/status/recurrent chess, lower-precision execution equivalence, full native-intermediate parity, training benefit and performance superiority remain outside this correction's demonstrated scope.
