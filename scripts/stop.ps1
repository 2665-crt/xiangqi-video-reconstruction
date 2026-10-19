[CmdletBinding(SupportsShouldProcess)]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimePath = Join-Path $projectRoot 'data\runtime.json'
if (-not (Test-Path -LiteralPath $runtimePath)) { Write-Output '本项目没有启动记录。'; return }
$runtimeRecord = Get-Content -LiteralPath $runtimePath -Raw | ConvertFrom-Json
$verifiedProcesses = @()
foreach ($serviceName in @('backend', 'frontend')) {
    $recordedId = $runtimeRecord."${serviceName}Pid"
    $recordedTicks = $runtimeRecord."${serviceName}StartTicks"
    $serviceProcess = Get-Process -Id $recordedId -ErrorAction SilentlyContinue
    if (-not $serviceProcess) { continue }
    $commandInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $recordedId"
    if (-not $recordedTicks -or $serviceProcess.StartTime.Ticks -ne $recordedTicks) { throw '进程创建时间不符，停止操作；不会关闭其他项目。' }
    if ($serviceName -eq 'backend') {
        if ($commandInfo.ExecutablePath -ne (Join-Path $projectRoot '.venv\Scripts\python.exe') -or $commandInfo.CommandLine -notlike '*uvicorn app.main:app*--port 8278*') { throw '后端进程身份不符，停止操作。' }
        foreach ($childProcess in (Get-CimInstance Win32_Process -Filter "ParentProcessId = $recordedId" | Where-Object { $_.CommandLine -like '*uvicorn app.main:app*--port 8278*' })) {
            $verifiedProcesses += $childProcess.ProcessId
        }
    } elseif ($commandInfo.CommandLine -notlike '*node_modules/vite/bin/vite.js*--port 5278*--strictPort*') { throw '前端进程身份不符，停止操作。' }
    $verifiedProcesses += $recordedId
}
foreach ($verifiedId in $verifiedProcesses) {
    if ($PSCmdlet.ShouldProcess("本项目进程 $verifiedId", '停止服务')) { Stop-Process -Id $verifiedId -Force -ErrorAction SilentlyContinue }
}
if (-not $WhatIfPreference) { Write-Output '象棋复盘服务已停止。视频和棋谱仍保存在 data。' }
