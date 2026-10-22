$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimePath = Join-Path $projectRoot 'data\runtime.json'
if (-not (Test-Path -LiteralPath $runtimePath)) { Write-Output '实验版没有启动记录。'; return }
$env:EXPERIMENT_STOP_ROOT = $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot '.venv\Lib\site-packages'
@'
import json,os
from datetime import datetime
from pathlib import Path
import psutil

root=Path(os.environ['EXPERIMENT_STOP_ROOT']).resolve()
runtime=json.loads((root/'data/runtime.json').read_text(encoding='utf-8-sig'))
verified=[]
for kind in ('backend','frontend'):
    pid=runtime[kind+'Pid']
    if not psutil.pid_exists(pid):
        continue
    process=psutil.Process(pid)
    created=datetime.fromtimestamp(process.create_time())
    ticks=((created.toordinal()-1)*864000000000+
           (created.hour*3600+created.minute*60+created.second)*10000000+created.microsecond*10)
    if abs(ticks-runtime[kind+'StartTicks'])>100000:
        raise RuntimeError('进程创建时间已变更，停止操作取消')
    expected=root if kind=='backend' else root/'frontend'
    if Path(process.cwd()).resolve()!=expected:
        raise RuntimeError('进程工作目录不是本实验目录，停止操作取消')
    args=process.cmdline()
    port=str(runtime[kind+'Port'])
    if '--port' not in args or args[args.index('--port')+1]!=port or port in ('5278','8278'):
        raise RuntimeError('进程端口不符合实验记录，停止操作取消')
    if kind=='backend' and not any('uvicorn' in arg for arg in args):
        raise RuntimeError('后端进程类型不符')
    if kind=='frontend' and 'vite.experiment.config.ts' not in args:
        raise RuntimeError('前端进程类型不符')
    verified.extend([process,*process.children(recursive=True)])
for process in reversed(verified):
    try:process.terminate()
    except psutil.NoSuchProcess:pass
_,alive=psutil.wait_procs(verified,timeout=3)
for process in alive:
    try:process.kill()
    except psutil.NoSuchProcess:pass
print('实验版已停止；正式版、视频和任务数据保留。')
'@ | & (Join-Path $projectRoot 'output\github-recognition-study\.venv\Scripts\python.exe') -
if ($LASTEXITCODE -ne 0) { throw '实验进程核验或停止失败，未执行未核实的停止操作。' }
