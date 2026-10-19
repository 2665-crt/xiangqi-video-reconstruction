param([switch]$Fusion)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $python = (Get-Command python.exe -ErrorAction Stop).Source
    & $python -m venv (Join-Path $projectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw '创建 Python 环境失败。' }
}
$requirements = if ($Fusion) { 'backend/requirements-fusion.txt' } else { 'backend/requirements.txt' }
& $pythonExe -m pip install -r (Join-Path $projectRoot $requirements)
if ($LASTEXITCODE -ne 0) { throw '安装 Python 依赖失败。' }
Push-Location (Join-Path $projectRoot 'frontend')
try {
    & npm.cmd ci
    if ($LASTEXITCODE -ne 0) { throw '安装前端依赖失败。' }
} finally { Pop-Location }
Write-Output '依赖已安装。运行 scripts/start.ps1 可使用手动复盘；融合模式需另行配置权重。'
