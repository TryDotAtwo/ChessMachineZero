param(
    [int] $Port = 8768
)

$ErrorActionPreference = "Stop"
$Name = "cmz-native-dev"
$StreamLog = "/work/test_results/native_container_logs/docker-stream.log"
$DashboardLog = "/work/test_results/native_container_logs/dashboard.log"
$StartedAt = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")

$StopCommand = "(pgrep -f '[c]hess_machine_zero.dashboard.server' || true; pgrep -f '[c]mz-dashboard' || true) | xargs -r kill"
docker exec $Name bash -lc $StopCommand | Out-Null

$LaunchCommand = "mkdir -p /work/test_results/native_container_logs; echo '=== cmz native dashboard start $StartedAt host=0.0.0.0 port=$Port ===' | tee -a $DashboardLog $StreamLog; cd /work/native; export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:`$LD_LIBRARY_PATH; cargo run -p cmz-dashboard -- --host 0.0.0.0 --port $Port 2>&1 | tee -a $DashboardLog $StreamLog"
docker exec -d $Name bash -lc $LaunchCommand

Start-Sleep -Seconds 2
$ProbeCommand = "pgrep -af '[c]mz-dashboard' && echo dashboard-url=http://127.0.0.1:$Port"
docker exec $Name bash -lc $ProbeCommand
docker logs --tail 40 $Name
