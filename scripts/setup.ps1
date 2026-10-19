param([switch]$Fusion)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimePath = Join-Path $projectRoot 'data/runtime.json'
if (Test-Path -LiteralPath $runtimePath) {
    $runtime = Get-Content -LiteralPath $runtimePath -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($service in @('backend', 'frontend')) {
        $recordedPid = $runtime."${service}Pid"
        if (-not $recordedPid) { continue }
        $runningProcess = Get-Process -Id $recordedPid -ErrorAction SilentlyContinue
        if ($runningProcess -and $runningProcess.StartTime.Ticks -eq $runtime."${service}StartTicks") {
            throw '本项目服务正在运行，请先运行 scripts/stop.ps1，再执行安装或更新。'
        }
    }
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$nodeExe = (Get-Command node.exe -ErrorAction SilentlyContinue).Source
if (-not $nodeExe -or -not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw '请先安装 Node.js 22 或更高版本：https://nodejs.org/；安装后重新打开 PowerShell。'
}
$nodeVersion = & $nodeExe -p "process.versions.node.split('.')[0]"
if ($LASTEXITCODE -ne 0 -or [int]$nodeVersion -lt 22) { throw '需要 Node.js 22 或更高版本，请更新后重新运行安装脚本。' }
$pythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        & $launcher.Source -3.12 -m venv (Join-Path $projectRoot '.venv')
    } else {
        $python = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
        if (-not $python) { throw '请先安装 64 位 Python 3.12：https://www.python.org/downloads/；安装后重新打开 PowerShell。' }
        & $python -m venv (Join-Path $projectRoot '.venv')
    }
    if ($LASTEXITCODE -ne 0) { throw '创建环境失败，请安装 64 位 Python 3.12 后重新运行安装脚本。' }
}
& $pythonExe -c "import sys,struct; sys.exit(0 if sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8 else 1)"
if ($LASTEXITCODE -ne 0) { throw '项目 .venv 必须使用 64 位 Python 3.12；请将现有 .venv 改名备份后重新运行安装脚本。' }
$env:PYTHONPATH = Join-Path $projectRoot 'backend'
$env:PYTHONNOUSERSITE = '1'
$env:MPLCONFIGDIR = Join-Path $projectRoot 'data/matplotlib'
$env:VISION_FUSION = if ($Fusion) { '1' } else { '0' }
$requirements = if ($Fusion) { 'backend/requirements-fusion.txt' } else { 'backend/requirements.txt' }
Write-Output '正在安装 Python 依赖；首次安装需要联网下载，请等待命令完成。'
if ($Fusion) {
    & $pythonExe -m pip install -r (Join-Path $projectRoot $requirements) -c (Join-Path $projectRoot 'backend/constraints-fusion.txt')
} else {
    & $pythonExe -m pip install -r (Join-Path $projectRoot $requirements)
}
if ($LASTEXITCODE -ne 0) { throw '安装 Python 依赖失败。' }
& $pythonExe -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Python 依赖存在冲突，请检查上面的包名和版本。' }
& $pythonExe -c "import av,cv2,numba,onnxruntime; from app.vision import VisionClient; client=VisionClient(); import os; assert os.environ['VISION_FUSION'] != '1' or client.local_classifier is not None; print('Python dependencies and local models: OK')"
if ($LASTEXITCODE -ne 0) { throw '依赖或模型加载失败。请确认完整下载仓库中的 backend/models，并重新运行 scripts/setup.ps1 -Fusion。' }
Push-Location (Join-Path $projectRoot 'frontend')
try {
    & npm.cmd ci
    if ($LASTEXITCODE -ne 0) { throw '安装前端依赖失败。请确认本项目服务已停止，并检查上方的网络或文件占用错误。' }
    & $nodeExe -e "const {chromium}=require('playwright'); (async()=>{const browser=await chromium.launch({channel:'chrome',headless:true}); await browser.close(); console.log('Recording browser: OK');})().catch(error=>{console.error(error.message); process.exitCode=1;});"
    if ($LASTEXITCODE -ne 0) { throw '录像导出浏览器检查失败。请安装 Google Chrome：https://www.google.com/chrome/，然后重新运行安装脚本。' }
} finally { Pop-Location }
if ($Fusion) {
    Write-Output '安装和模型检查完成。运行 scripts/start-fusion.ps1，然后打开 http://127.0.0.1:5278/。本地识别无需 API Key 或重新训练。'
} else {
    Write-Output '基础依赖检查完成。完整本地识别请继续运行 scripts/setup.ps1 -Fusion 和 scripts/start-fusion.ps1；手动复盘或已配置云端接口可运行 scripts/start.ps1。'
}
