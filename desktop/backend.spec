from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

root = Path(SPECPATH).parent
datas = [(str(root / 'backend' / 'models'), 'models')]
datas += [(str(source), 'app') for source in (root / 'backend' / 'app').glob('*.py')]
datas += collect_data_files('ultralytics')
hiddenimports = collect_submodules('ultralytics')
hiddenimports += ['uvicorn.logging', 'uvicorn.loops.auto', 'uvicorn.protocols.http.auto',
                  'uvicorn.protocols.websockets.auto', 'uvicorn.lifespan.on']
a = Analysis([str(root / 'desktop' / 'backend_entry.py')], pathex=[str(root / 'backend')],
             binaries=[], datas=datas, hiddenimports=hiddenimports,
             excludes=['tkinter', 'IPython', 'pytest'], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='XiangQiBackend',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='backend')
