param(
    [Parameter(Mandatory=$true)]
    [string] $Command,
    [string] $Log = "test_results/native_container_logs/exec.log"
)

$ErrorActionPreference = "Stop"
$Name = "cmz-native-dev"

$LogDir = Split-Path -Parent $Log
if ($LogDir) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
}

$RunId = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
$StreamLog = "/work/test_results/native_container_logs/docker-stream.log"
$WrappedCommand = "mkdir -p /work/test_results/native_container_logs; export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:`$LD_LIBRARY_PATH; { echo '=== cmz exec start $RunId ==='; $Command; _cmz_status=`$?; echo `"=== cmz exec end $RunId status=`$_cmz_status ===`"; exit `$_cmz_status; } 2>&1 | tee -a $StreamLog; exit `${PIPESTATUS[0]}"

$StartInfo = [System.Diagnostics.ProcessStartInfo]::new()
$StartInfo.FileName = "docker"
$StartInfo.UseShellExecute = $false
$StartInfo.RedirectStandardOutput = $true
$StartInfo.RedirectStandardError = $true
$EscapedCommand = $WrappedCommand.Replace('"', '\"')
$StartInfo.Arguments = "exec $Name bash -lc `"$EscapedCommand`""

$Process = [System.Diagnostics.Process]::new()
$Process.StartInfo = $StartInfo
[void] $Process.Start()
$StdOutText = $Process.StandardOutput.ReadToEnd()
$StdErrText = $Process.StandardError.ReadToEnd()
$Process.WaitForExit()

$Combined = $StdOutText + $StdErrText
Set-Content -LiteralPath $Log -Value $Combined
Write-Output $Combined
exit $Process.ExitCode
