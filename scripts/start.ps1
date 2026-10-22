param([switch]$Fusion = $true)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$legacyPython = Join-Path $projectRoot 'output\github-recognition-study\.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $legacyPython) { $pythonExe = $legacyPython }
$frontendRoot = Join-Path $projectRoot 'frontend'
$nodeExe = (Get-Command node.exe -ErrorAction SilentlyContinue).Source
$setupCommand = if ($Fusion) { 'scripts/setup.ps1 -Fusion' } else { 'scripts/setup.ps1' }

if (-not $nodeExe) { throw '未找到 Node.js。请安装 Node.js 22 或更高版本后重新打开 PowerShell。' }
if (-not (Test-Path -LiteralPath $pythonExe)) { throw "缺少项目运行环境，请先运行 $setupCommand。" }
if (-not (Test-Path -LiteralPath (Join-Path $frontendRoot 'node_modules\vite\bin\vite.js'))) { throw "缺少前端依赖，请先运行 $setupCommand。" }

foreach ($projectPort in @(5278, 8278)) {
    $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse('127.0.0.1'), $projectPort)
    try { $probe.Start() }
    catch { throw "端口 $projectPort 已占用。启动已停止，不会关闭其他项目，也不会自动换端口。" }
    finally { $probe.Stop() }
}

$env:PYTHONPATH = Join-Path $projectRoot 'backend'
if ($pythonExe -eq $legacyPython) { $env:PYTHONPATH += ';' + (Join-Path $projectRoot '.venv\Lib\site-packages') }
$env:PYTHONNOUSERSITE = '1'
$env:MPLCONFIGDIR = Join-Path $projectRoot 'data/matplotlib'
if ($Fusion) {
    foreach ($modelName in @('detect-ultra.pt', 'detect-aux.pt')) {
        if (-not (Test-Path -LiteralPath (Join-Path $projectRoot "backend/models/fusion/$modelName"))) {
            throw "仓库模型 $modelName 缺失，请重新完整下载本仓库；无需重新训练。"
        }
    }
    $env:VISION_FUSION = '1'
} else { $env:VISION_FUSION = '0' }
$logRoot = Join-Path $projectRoot 'data\logs'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$backendProcess = Start-Process -FilePath $pythonExe -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--app-dir', 'backend', '--host', '127.0.0.1', '--port', '8278') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logRoot 'backend.log') -RedirectStandardError (Join-Path $logRoot 'backend-error.log')
$frontendProcess = Start-Process -FilePath $nodeExe -ArgumentList @('node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '5278', '--strictPort') -WorkingDirectory $frontendRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logRoot 'frontend.log') -RedirectStandardError (Join-Path $logRoot 'frontend-error.log')
@{ backendPid = $backendProcess.Id; frontendPid = $frontendProcess.Id; backendStartTicks = $backendProcess.StartTime.Ticks; frontendStartTicks = $frontendProcess.StartTime.Ticks; backendPort = 8278; frontendPort = 5278; startedAt = (Get-Date).ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $projectRoot 'data\runtime.json') -Encoding utf8

$ready = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8278/api/health' -TimeoutSec 2
        if ($health.status -ne 'ok') { throw $health.detail }
        if ($Fusion -and -not $health.vision_configured) { throw '本地融合识别尚未就绪' }
        $page = Invoke-WebRequest -Uri 'http://127.0.0.1:5278' -UseBasicParsing -TimeoutSec 2
        if ($page.StatusCode -eq 200) { $ready = $true; break }
    } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $ready) { throw "服务尚未就绪，请查看 $logRoot 中的日志，修正后先运行 scripts/stop.ps1，再运行 $setupCommand 和启动脚本。" }
Write-Output '象棋复盘已启动：http://127.0.0.1:5278'
Write-Output "后端：http://127.0.0.1:8278；自动识别配置就绪：$($health.vision_configured)"
