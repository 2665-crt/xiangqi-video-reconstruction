param([Parameter(Mandatory=$true)][string]$OutputRoot)
$ErrorActionPreference = 'Stop'
$sourceRoot = Split-Path $PSScriptRoot -Parent
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$buildRoot = Join-Path $OutputRoot 'build'
if (Test-Path -LiteralPath (Join-Path $OutputRoot 'XiangQi Vedio Reconstruction')) {
    throw 'The output application folder already exists. Select an empty output directory.'
}
New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
$env:PIP_CACHE_DIR = Join-Path $buildRoot 'cache/pip'
$env:npm_config_cache = Join-Path $buildRoot 'cache/npm'
$env:electron_config_cache = Join-Path $buildRoot 'cache/electron'
$env:MPLCONFIGDIR = Join-Path $buildRoot 'matplotlib'
$env:YOLO_CONFIG_DIR = Join-Path $buildRoot 'ultralytics'
function Run-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}
$python = Join-Path $buildRoot 'venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $python)) { Run-Checked 'py' @('-3.12', '-m', 'venv', (Join-Path $buildRoot 'venv')) }
Run-Checked $python @('-m', 'pip', 'install', 'torch==2.7.1+cpu', 'torchvision==0.22.1+cpu', '--index-url', 'https://download.pytorch.org/whl/cpu')
Run-Checked $python @('-m', 'pip', 'install', '-r', (Join-Path $sourceRoot 'backend/requirements-fusion.txt'), '-c', (Join-Path $sourceRoot 'backend/constraints-fusion.txt'), 'pyinstaller==6.22.3')
$tools = Join-Path $buildRoot 'tools'
New-Item -ItemType Directory -Force -Path $tools | Out-Null
'{"name":"xiangqi-desktop-build-tools","private":true,"devDependencies":{"electron":"44.9.0","@electron/packager":"20.3.0"}}' | Set-Content -LiteralPath (Join-Path $tools 'package.json') -Encoding UTF8
Push-Location $tools
try {
    Run-Checked 'npm.cmd' @('install', '--no-audit', '--no-fund')
    if (!(Test-Path -LiteralPath 'node_modules/electron/dist/electron.exe')) { Run-Checked 'node' @('node_modules/electron/install.js') }
} finally { Pop-Location }
$frontend = Join-Path $buildRoot 'frontend'
New-Item -ItemType Directory -Force -Path $frontend | Out-Null
Get-ChildItem -LiteralPath (Join-Path $sourceRoot 'frontend') | Where-Object Name -NotIn @('node_modules','dist') | Copy-Item -Destination $frontend -Recurse -Force
Push-Location $frontend
try { Run-Checked 'npm.cmd' @('ci', '--no-audit', '--no-fund'); Run-Checked 'npm.cmd' @('run', 'build') } finally { Pop-Location }
Run-Checked $python @('-m', 'PyInstaller', '--noconfirm', '--distpath', $buildRoot, '--workpath', (Join-Path $buildRoot 'pyinstaller'), (Join-Path $PSScriptRoot 'backend.spec'))
Run-Checked 'node' @((Join-Path $PSScriptRoot 'package-app.mjs'), $buildRoot, $OutputRoot)
Run-Checked $python @((Join-Path $PSScriptRoot 'notices.py'), (Join-Path $OutputRoot 'XiangQi Vedio Reconstruction'), $sourceRoot, $buildRoot)
Write-Host "Portable application ready: $OutputRoot\XiangQi Vedio Reconstruction"
