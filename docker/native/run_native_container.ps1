$ErrorActionPreference = "Stop"

$Image = "cmz-native-dev:2026-05-26"
$Name = "cmz-native-dev"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
$DashboardHost = "127.0.0.1"
$DashboardPort = 8768

docker build -f "$Root\docker\native\Dockerfile" -t $Image "$Root"
$Existing = docker ps -a --filter "name=^/$Name$" --format "{{.Names}}"
if ($Existing -eq $Name) {
    docker rm -f $Name | Out-Null
}
docker run -d --name $Name --gpus all -p "$DashboardHost`:$DashboardPort`:8768" -v "${Root}:/work" -w /work $Image
docker logs $Name
