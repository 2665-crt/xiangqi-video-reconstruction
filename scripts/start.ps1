$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot 'output\github-recognition-study\.venv\Scripts\python.exe'
$frontendRoot = Join-Path $projectRoot 'frontend'
$nodeExe = (Get-Command node.exe -ErrorAction Stop).Source

if (-not (Test-Path -LiteralPath $pythonExe)) { throw '缺少本地融合运行环境，请先按 README 安装项目依赖。' }
if (-not (Test-Path -LiteralPath (Join-Path $frontendRoot 'node_modules\vite\bin\vite.js'))) { throw '缺少前端依赖，请先在 frontend 中执行 npm.cmd install。' }

foreach ($projectPort in @(5278, 8278)) {
    $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse('127.0.0.1'), $projectPort)
    try { $probe.Start() }
    catch { throw "端口 $projectPort 已占用。启动已停止，不会关闭其他项目，也不会自动换端口。" }
    finally { $probe.Stop() }
}

$env:PYTHONPATH = Join-Path $projectRoot '.venv\Lib\site-packages'
$env:VISION_FUSION = '1'
$logRoot = Join-Path $projectRoot 'data\logs'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$backendProcess = Start-Process -FilePath $pythonExe -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--app-dir', 'backend', '--host', '127.0.0.1', '--port', '8278') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logRoot 'backend.log') -RedirectStandardError (Join-Path $logRoot 'backend-error.log')
$frontendProcess = Start-Process -FilePath $nodeExe -ArgumentList @('node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '5278', '--strictPort') -WorkingDirectory $frontendRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logRoot 'frontend.log') -RedirectStandardError (Join-Path $logRoot 'frontend-error.log')
@{ backendPid = $backendProcess.Id; frontendPid = $frontendProcess.Id; backendStartTicks = $backendProcess.StartTime.Ticks; frontendStartTicks = $frontendProcess.StartTime.Ticks; backendPort = 8278; frontendPort = 5278; startedAt = (Get-Date).ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $projectRoot 'data\runtime.json') -Encoding utf8

$ready = $false
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8278/api/health' -TimeoutSec 2
        $page = Invoke-WebRequest -Uri 'http://127.0.0.1:5278' -TimeoutSec 2
        if ($health.status -eq 'ok' -and $page.StatusCode -eq 200) { $ready = $true; break }
    } catch { Start-Sleep -Milliseconds 250 }
}
if (-not $ready) { throw "服务尚未就绪，请查看 $logRoot 中的日志。只创建了本项目进程，PID 记录在 data/runtime.json。" }
Write-Output '象棋复盘已启动：http://127.0.0.1:5278'
Write-Output "后端：http://127.0.0.1:8278；自动识别配置就绪：$($health.vision_configured)"
