# VM inspector site

The static site presents the exact currently executable artifact boundary. It
replays a fixed browser fixture for interaction; it does not pretend to execute
the native CUDA artifact in JavaScript. The native evidence and tensor shapes
shown in the UI are pinned by `tests/test_site_contract.py`.

`python -m vm_compiler.site_trace` regenerates `artifact_trace.json` directly
from `build_position_reconstruction_artifact()`. The site renders all 45 SSA
operations from that file; the contract test rejects any stale or hand-edited
trace.

`numeric_trace.json` contains the exact executed COO values. The bilingual
matrix inspector renders every operation as a semantic tensor transformation;
for GEMM it highlights the contributing row and column and expands their dot
product, while route, concat, add, hardmax, and attention show their exact
operation-specific provenance.

Serve locally from the repository root:

```powershell
python -m http.server 4173 --directory site
```

GitHub Pages publishes only this directory through `.github/workflows/pages.yml`
after the change reaches `main` and Pages is configured to use GitHub Actions.
