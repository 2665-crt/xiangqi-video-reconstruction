param(
    [ValidateRange(1024,65535)][int]$FrontendPort = 5281,
    [ValidateRange(1024,65535)][int]$BackendPort = 8281,
    [switch]$Baseline
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$dataRoot = Join-Path $projectRoot 'data'
[IO.Directory]::CreateDirectory($dataRoot) | Out-Null
$startupLock = [IO.File]::Open((Join-Path $dataRoot 'experiment-start.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
try {
$runtimePath = Join-Path $dataRoot 'runtime.json'
if (Test-Path -LiteralPath $runtimePath) {
    $existing = Get-Content -LiteralPath $runtimePath -Raw | ConvertFrom-Json
    foreach ($serviceName in @('backend','frontend')) {
        $existingId = $existing."${serviceName}Pid"
        if (-not $existingId) { continue }
        $process = Get-Process -Id $existingId -ErrorAction SilentlyContinue
        $ticks = $existing."${serviceName}StartTicks"
        if ($process -and (-not $ticks -or [Math]::Abs($process.StartTime.Ticks - $ticks) -lt 10000)) {
            throw '实验版已有运行实例，请先运行 scripts/stop-experiment.ps1，再切换端口或对照模式。'
        }
    }
}
$pythonExe = Join-Path $projectRoot 'output\github-recognition-study\.venv\Scripts\python.exe'
$frontendRoot = Join-Path $projectRoot 'frontend'
$nodeExe = (Get-Command node.exe -ErrorAction Stop).Source
if ($FrontendPort -eq $BackendPort -or $FrontendPort -in @(5278,8278) -or $BackendPort -in @(5278,8278)) {
    throw '实验端口必须独立，且不能使用正式版 5278/8278。'
}
foreach ($projectPort in @($FrontendPort,$BackendPort)) {
    $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse('127.0.0.1'), $projectPort)
    try { $probe.Start() }
    catch { throw "实验端口 $projectPort 已占用，不会关闭其他程序。" }
    finally { $probe.Stop() }
}
$env:PYTHONPATH = Join-Path $projectRoot '.venv\Lib\site-packages'
$env:VISION_FUSION = '1'
$env:VIDEO_REFERENCE_CACHE = if ($Baseline) { '0' } else { '1' }
$env:VIDEO_DECODE_THREADS = if ($Baseline) { '0' } else { '4' }
$env:EXPERIMENT_FRONTEND_PORT = [string]$FrontendPort
$env:EXPERIMENT_BACKEND_PORT = [string]$BackendPort
$logRoot = Join-Path $projectRoot 'data\logs'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$backendProcess = Start-Process -FilePath $pythonExe -ArgumentList @('-m','uvicorn','app.main:app','--app-dir','backend','--host','127.0.0.1','--port',[string]$BackendPort) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logRoot 'backend.log') -RedirectStandardError (Join-Path $logRoot 'backend-error.log')
$frontendProcess = Start-Process -FilePath $nodeExe -ArgumentList @('node_modules/vite/bin/vite.js','--config','vite.experiment.config.ts','--host','127.0.0.1','--port',[string]$FrontendPort,'--strictPort') -WorkingDirectory $frontendRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logRoot 'frontend.log') -RedirectStandardError (Join-Path $logRoot 'frontend-error.log')
@{ backendPid=$backendProcess.Id; frontendPid=$frontendProcess.Id; backendStartTicks=$backendProcess.StartTime.Ticks; frontendStartTicks=$frontendProcess.StartTime.Ticks; backendPort=$BackendPort; frontendPort=$FrontendPort; baseline=[bool]$Baseline; startedAt=(Get-Date).ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $projectRoot 'data\runtime.json') -Encoding utf8
$ready = $false
for ($attempt=0; $attempt -lt 30; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/api/health" -TimeoutSec 2
        $page = Invoke-WebRequest -Uri "http://127.0.0.1:$FrontendPort/api/health" -TimeoutSec 2
        if ($health.status -eq 'ok' -and $page.StatusCode -eq 200) { $ready=$true; break }
    } catch { Start-Sleep -Milliseconds 250 }
}
if (-not $ready) { throw "实验服务未就绪，请查看 $logRoot；进程信息保存在本实验目录的 data/runtime.json。" }
Write-Output "实验版：http://127.0.0.1:$FrontendPort；后端：$BackendPort；原版对照模式：$Baseline"
} finally {
    $startupLock.Dispose()
}
