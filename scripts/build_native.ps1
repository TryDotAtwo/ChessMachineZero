[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9]{2,3}$')][string]$CudaArchitecture,
    [Parameter(Mandatory = $true)][string]$BuildDirectory,
    [string]$Python = 'python',
    [string[]]$Targets = @('artifact', 'recurrent', 'ste', 'hullkv', 'graph_validation', 'position_artifact'),
    [switch]$AllowUnsupportedCompiler,
    [switch]$RunTests,
    [string]$FixtureDirectory
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = Split-Path -Parent $PSScriptRoot
$build = [IO.Path]::GetFullPath($BuildDirectory)
if (Test-Path -LiteralPath $build) { throw "BuildDirectory must be NEW: $build" }
$knownTargets = @('artifact', 'recurrent', 'ste', 'hullkv', 'graph_validation', 'position_artifact')
foreach ($target in $Targets) {
    if ($target -notin $knownTargets) { throw "Unknown target: $target" }
}
if ($RunTests -and 'position_artifact' -in $Targets -and !$FixtureDirectory) {
    throw 'Position acceptance requires -FixtureDirectory (see tests/export_native_fixtures.py)'
}
$savedEnvironment = @{}
foreach ($name in @('PATH', 'INCLUDE', 'LIB')) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
function Invoke-Checked([string]$Command, [string[]]$Arguments) {
    $description = $Command + ' ' + ($Arguments -join ' ')
    Write-Host $description
    Add-Content -LiteralPath (Join-Path $build 'build.log') -Value $description -Encoding utf8
    & $Command @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $build 'build.log') -Append | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "$Command exited $LASTEXITCODE" }
}
try {
    # JSON is deliberately ASCII escaped: PowerShell/OEM decoding corrupts raw Unicode paths.
    $torchJson = & $Python -c 'import json,pathlib,sys,torch; print(json.dumps(dict(root=str(pathlib.Path(torch.__file__).parent),version=torch.__version__,cuda=torch.version.cuda,python=sys.version)))'
    if ($LASTEXITCODE -ne 0) { throw 'Python/Torch discovery failed' }
    $torch = $torchJson | ConvertFrom-Json
    if (!$torch.cuda) { throw 'Install a CUDA-enabled Torch distribution for the native runtime' }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or !$vs) { throw 'Visual Studio C++ Build Tools not found' }
    $vc = Get-ChildItem -LiteralPath (Join-Path $vs 'VC/Tools/MSVC') -Directory |
        Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
    $sdkRoot = (Get-ItemProperty 'HKLM:/SOFTWARE/Microsoft/Windows Kits/Installed Roots').KitsRoot10
    $sdk = Get-ChildItem -LiteralPath (Join-Path $sdkRoot 'Include') -Directory |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'um/Windows.h') } |
        Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
    $nvcc = (Get-Command nvcc.exe -ErrorAction Stop).Source
    $cuda = Split-Path -Parent (Split-Path -Parent $nvcc)
    $compilerDirectory = Join-Path $vc.FullName 'bin/Hostx64/x64'
    $cl = Join-Path $compilerDirectory 'cl.exe'
    $link = Join-Path $compilerDirectory 'link.exe'
    $torchLib = Join-Path $torch.root 'lib'
    $env:PATH = "$torchLib;$compilerDirectory;$(Join-Path $cuda 'bin');$env:PATH"
    $env:INCLUDE = (@((Join-Path $vc.FullName 'include')) +
        @('ucrt', 'shared', 'um', 'winrt' | ForEach-Object { Join-Path $sdk.FullName $_ })) -join ';'
    $env:LIB = (@((Join-Path $vc.FullName 'lib/x64')) +
        @('ucrt/x64', 'um/x64' | ForEach-Object { Join-Path (Join-Path $sdkRoot "Lib/$($sdk.Name)") $_ })) -join ';'
    New-Item -ItemType Directory -Path $build | Out-Null
    $includes = @((Join-Path $repo 'include'), (Join-Path $torch.root 'include'),
        (Join-Path $torch.root 'include/torch/csrc/api/include'), (Join-Path $cuda 'include'))
    $cppFlags = @('/nologo', '/std:c++17', '/EHsc', '/MD', '/O2', '/utf-8', '/wd4267', '/wd4996') +
        @($includes | ForEach-Object { "/I$_" })
    $sourceNames = @('artifact.cpp', 'module.cpp', 'ste.cpp', 'hullkv_autograd.cpp', 'hullkv.cu')
    $sources = @($sourceNames | ForEach-Object { Join-Path $repo "src/$_" })
    $manifestSources = @($sources) + @($Targets | ForEach-Object { Join-Path $repo "tests/native/$($_)_test.cpp" }) +
        @(Get-ChildItem -LiteralPath (Join-Path $repo 'include/cmz') -File | ForEach-Object { $_.FullName }) + @($PSCommandPath)
    $manifest = [ordered]@{ utc = [DateTime]::UtcNow.ToString('o'); torch = $torch;
        vc = $vc.FullName; sdk = $sdk.Name; cuda = $cuda; architecture = $CudaArchitecture;
        allowUnsupportedCompiler = [bool]$AllowUnsupportedCompiler; targets = $Targets;
        sources = @($manifestSources | Get-FileHash -Algorithm SHA256 | Select-Object Path, Hash) }
    $manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Encoding utf8
    Invoke-Checked $nvcc @('--version')
    $objects = @()
    # NVCC 12.x forwards Unicode filenames to MSVC through an ANSI intermediate.
    # Existing 8.3 paths avoid that conversion; never copy or compile stale sources.
    $fileSystem = New-Object -ComObject Scripting.FileSystemObject
    $shortBuild = $fileSystem.GetFolder($build).ShortPath
    foreach ($source in $sources) {
        $object = Join-Path $build ([IO.Path]::GetFileNameWithoutExtension($source) + '.obj')
        if ($source.EndsWith('.cu')) {
            $flags = @('-std=c++17', '-O2', '-lineinfo', '-diag-suppress=1388,1394,1390', '-Xcompiler', '/MD,/EHsc,/utf-8,/wd4267,/wd4996',
                '-ccbin', $compilerDirectory, '-gencode', "arch=compute_$CudaArchitecture,code=sm_$CudaArchitecture") +
                @($includes | ForEach-Object { '-I' + $fileSystem.GetFolder($_).ShortPath })
            if ($AllowUnsupportedCompiler) { $flags += '--allow-unsupported-compiler' }
            $shortObject = Join-Path $shortBuild ([IO.Path]::GetFileName($object))
            Invoke-Checked $nvcc ($flags + @('-c', $fileSystem.GetFile($source).ShortPath, '-o', $shortObject))
        } else {
            Invoke-Checked $cl ($cppFlags + @('/c', $source, "/Fo$object"))
        }
        $objects += $object
    }
    foreach ($target in $Targets) {
        $testObject = Join-Path $build "$($target)_test.obj"
        Invoke-Checked $cl ($cppFlags + @('/c', (Join-Path $repo "tests/native/$($target)_test.cpp"), "/Fo$testObject"))
        $executable = Join-Path $build "cmz_$($target)_test.exe"
        # Same CUDA-registration anchor as Torch's Windows cpp_extension linker.
        Invoke-Checked $link (@('/nologo', "/OUT:$executable", '/INCLUDE:?warp_size@cuda@at@@YAHXZ',
            "/LIBPATH:$torchLib", "/LIBPATH:$(Join-Path $cuda 'lib/x64')") + $objects + @($testObject,
            'torch.lib', 'torch_cpu.lib', 'torch_cuda.lib', 'c10.lib', 'c10_cuda.lib', 'cudart.lib'))
        if ($RunTests) {
            $testArguments = @()
            if ($target -eq 'position_artifact') {
                $fixture = [IO.Path]::GetFullPath($FixtureDirectory)
                $testArguments = @((Join-Path $fixture 'position.cmz'), (Join-Path $fixture 'positions.bin'))
            }
            Invoke-Checked $executable $testArguments
            if ($target -eq 'ste') {
                Invoke-Checked $executable @('empty-mask-cpu')
                Invoke-Checked $executable @('empty-mask-cuda')
                Invoke-Checked $executable @('invalid-temperature')
                Invoke-Checked $executable @('invalid-scale')
            }
            if ($target -eq 'hullkv') {
                foreach ($case in @('metadata', 'index-negative', 'index-large', 'index-negative-batched', 'index-large-batched')) {
                    Invoke-Checked $executable @($case)
                }
            }
        }
    }
    Write-Host "Fresh native build complete: $build"
} finally {
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
}
