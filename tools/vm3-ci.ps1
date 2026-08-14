param(
  [string]$Target = "all",
  [string]$CTestRegex = "cmz_vm3",
  [string]$PytestArgs = "",
  [switch]$WithGpu
)

$ErrorActionPreference = "Stop"
$gpuArgs = if ($WithGpu) { @("--gpus", "all") } else { @() }
$pytestCommand = if ($PytestArgs) { " && python3 -m pytest $PytestArgs" } else { "" }

docker build -f docker/vm3/Dockerfile -t cmz-vm3-dev:2.7.1-cu128 .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker run --rm @gpuArgs -v "${PWD}:/work" -w /work cmz-vm3-dev:2.7.1-cu128 `
  bash -lc "cmake -S . -B build/vm3 -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH=/opt/pytorch/share/cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && cmake --build build/vm3 --target $Target && ctest --test-dir build/vm3 --output-on-failure -R '$CTestRegex'$pytestCommand"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
