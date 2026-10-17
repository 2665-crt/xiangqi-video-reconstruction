$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimePath = Join-Path $projectRoot 'data\runtime.json'
$runtime = Get-Content -LiteralPath $runtimePath -Raw | ConvertFrom-Json
$tasks = Invoke-RestMethod -Uri 'http://127.0.0.1:8278/api/tasks' -TimeoutSec 5
if ($tasks | Where-Object { $_.status -in @('analyzing', 'waiting_service') }) {
    throw '当前有正在运行或自动恢复的任务，未重启后端。'
}
$parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($runtime.backendPid)"
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if ($null -eq $parent -or $parent.ExecutablePath -ne $pythonExe -or $parent.CommandLine -notmatch 'uvicorn.+app.main:app.+8278') {
    throw '运行记录与本项目后端进程不匹配，未停止任何进程。'
}
$process = Get-Process -Id $runtime.backendPid
if ($process.StartTime.Ticks -ne $runtime.backendStartTicks) { throw '进程创建时间不匹配，未停止任何进程。' }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logs = Join-Path $projectRoot 'data\logs'
Copy-Item -LiteralPath $runtimePath -Destination (Join-Path $logs "runtime-before-recovery-$stamp.json")
& taskkill.exe /PID $runtime.backendPid /T /F | Out-Null
if ($LASTEXITCODE -ne 0) { throw '本项目后端未能停止，未继续启动。' }
$backend = Start-Process -FilePath $pythonExe -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--app-dir', 'backend', '--host', '127.0.0.1', '--port', '8278') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs "backend-recovery-$stamp.log") -RedirectStandardError (Join-Path $logs "backend-recovery-$stamp-error.log")
$runtime.backendPid = $backend.Id
$runtime.backendStartTicks = $backend.StartTime.Ticks
$runtime | Add-Member -NotePropertyName backendRestartedAt -NotePropertyValue (Get-Date).ToString('o') -Force
$runtime | ConvertTo-Json | Set-Content -LiteralPath $runtimePath -Encoding utf8
$ready = $false
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8278/api/health' -TimeoutSec 2
        if ($health.status -eq 'ok') { $ready = $true; break }
    } catch { Start-Sleep -Milliseconds 250 }
}
if (-not $ready) { throw '新后端尚未就绪，请检查本次独立日志。' }
$schema = Invoke-RestMethod -Uri 'http://127.0.0.1:8278/openapi.json' -TimeoutSec 5
[pscustomobject]@{ backendReady=$ready; recoveryEndpointReady=($null -ne $schema.paths.'/api/tasks/{task_id}/resume'); frontendPid=$runtime.frontendPid; backendPid=$runtime.backendPid } | ConvertTo-Json -Compress
