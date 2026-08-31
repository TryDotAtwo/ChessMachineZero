# Pure frozen Transformer VM — executable subgraphs

Current executable subgraphs reconstruct the **exact full board from an already
valid chronological history** (up to 400 plies), and enumerate an ordered
**complete legal set** including king safety, castling, en passant and promotions.
They do not yet validate/apply a requested move, adjudicate terminal status, or
emit the full recurrent next context. `FrozenVm::execute_graph` returns either
the position artifact's `[B,64,128]` board or the legal artifact's `[B,768,128]`
256 move triples with padding. `forward` requires `[B,2045,128]` and rejects both
partial output shapes. There is no
trained player or demonstrated full-game learning result. See
[architecture](docs/architecture.md) for the precise implemented/target boundary.

Production is generic LibTorch C++/CUDA. Python, NumPy and python-chess are
development/compiler/test tools, never production chess logic. The static site
shows an exported Python reference trace, not native runtime intermediates.
It still shows the 45-operation position subgraph, not the new legal-set graph
or a complete VM. [Current foundation evidence](test_results/full_vm_foundation_2026-08-31.md).

## Precision and backward

Artifacts store packed E2M1/FP4 codes **and FP32 scales** (some use one scale per
element, costing 4.5 bytes/element before metadata). Native weights decode to
FP32 CUDA tensors. This is not native FP4 compute or uniform four-bit memory.
FP16/BF16/TF32 equivalence and throughput advantages are unverified. Exact-board
acceptance uses FP32 with TF32 disabled.

Hardmax emits the lowest-index eligible maximum and uses a softmax-Jacobian
surrogate backward; empty mask rows fail with a device-side asynchronous assert
on CUDA. Attention differentiates selected-top-k softmax values for **Q, K and
V**, not full-candidate softmax. Native derivative tests compare every component
to independent scalar-double formulas (`rtol=2e-5`, `atol=2e-6`); winner and
full-board assertions use exact equality. These surrogates do not prove useful
learning. All 2064 position-event candidates are scanned; no sublinear search or
allocation-free execution claim is made.

## Python development acceptance

From the repository root, with Python 3.11 or newer:

```powershell
python -m pip install -r requirements-dev.txt
# A CPU wheel is sufficient for Python tests. Use a CUDA-enabled Torch instead
# if building native CUDA. The verified native environment used torch 2.8.0+cu128.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q
```

Do not replace a configured CUDA installation with the CPU example when native
tests are needed. Native builds require CUDA-enabled LibTorch (this script uses
the installed Python Torch package to discover its headers/import libraries),
Visual Studio C++ Build Tools, Windows SDK, CUDA Toolkit, and a compatible NVIDIA
driver. Production executables themselves do not invoke Python.

## Reproducible Windows native build

Use **new** output directories; the script rejects existing directories and
compiles current sources, recording SHA-256 inputs, toolchain metadata, exact
commands and output in `build-manifest.json` and `build.log`.

```powershell
python tests/export_native_fixtures.py build/native-fixtures
./scripts/build_native.ps1 -CudaArchitecture 86 -BuildDirectory build/native-sm86 -FixtureDirectory build/native-fixtures -RunTests -AllowUnsupportedCompiler
```

`86` is explicit for the tested RTX 3070 Laptop; choose the actual target
architecture. `-AllowUnsupportedCompiler` is an explicit opt-in needed for the
observed CUDA 12.5 / MSVC 19.44 pair; omit it for a supported toolkit/compiler
pair. The script discovers Torch/VS/SDK/CUDA, restores its process environment,
uses Windows short paths only for NVCC's ANSI filename limitation, and retains
Torch CUDA registration with `/INCLUDE:?warp_size@cuda@at@@YAHXZ`. No cached
binary counts as current acceptance. Source folders must support short filenames
when their paths contain non-ASCII characters.

The suite includes artifact checks, exact recurrent tensor invariants,
pre-device malformed graph rejection, hardmax/FP4 checks, Q/K/V derivatives for
three attention forms, batch isolation, non-default-stream execution, isolated
bad-index assertions, and 52 exact full-board fixture states. Test-only GPU
synchronization/readback observes results; production hot paths contain neither.
Invalid-mask/index children intentionally print device assertion diagnostics
and succeed only when the expected assertion is observed.

### Legal-set artifact acceptance

The 549-operation legal artifact is independently checked against every value
of 79 complete oracle legal sets. It is not an opening-position lookup table:
the compiler emits 7,780 position-independent geometry candidates and reusable
occupancy/king-safety circuits. The tests also cover long histories through
400 plies in the Python reference.

```powershell
python tests/export_legal_fixtures.py build/legal-fixtures
./scripts/build_native.ps1 -CudaArchitecture 86 -BuildDirectory build/legal-native -Targets tensor_fixture -TensorArtifact build/legal-fixtures/legal.cmz -TensorFixtures build/legal-fixtures/legal.bin -RunTests -AllowUnsupportedCompiler
```

This generic tensor-fixture target is opt-in; the default native command does
not silently count it as tested. The manifest records artifact/fixture hashes
and logical FP32 payload sizes. Those sizes exclude autograd/workspaces and do
not establish peak memory, throughput, or useful player gradients.

## CMake / CTest alternative

This alternative assumes a supported CUDA/host-compiler pair. For the observed
CUDA 12.5 / MSVC 19.44 pair, prefer the explicit opt-in PowerShell command above;
the CMake commands below do not silently bypass compiler compatibility checks.

```powershell
$torchCmake = python -c 'import json,torch; print(json.dumps(torch.utils.cmake_prefix_path))' | ConvertFrom-Json
cmake -S . -B build/cmake-native -DCMAKE_PREFIX_PATH="$torchCmake" -DCMAKE_CUDA_ARCHITECTURES=86 -DCMZ_POSITION_ARTIFACT="$PWD/build/native-fixtures/position.cmz" -DCMZ_POSITION_FIXTURES="$PWD/build/native-fixtures/positions.bin"
cmake --build build/cmake-native --config Release
ctest --test-dir build/cmake-native -C Release --output-on-failure
```

Add Torch's `lib` directory to the process DLL search `PATH` when running
executables/CTest separately. If either generated fixture path is absent,
CMake explicitly reports that position acceptance is **not registered**; such a
CTest run is incomplete. The observed Windows environment hangs in the
CMake/Ninja toolchain probes even after compiler children exit. The direct
PowerShell path is the verified workaround; the CMake edits are not claimed as
an executed successful build in that environment.
