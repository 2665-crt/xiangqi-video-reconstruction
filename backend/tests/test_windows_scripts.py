import base64
import os
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'


@pytest.mark.parametrize('script', sorted(SCRIPTS.glob('*.ps1')), ids=lambda path: path.name)
def test_powershell_script_has_no_embedded_byte_order_mark(script):
    assert '\ufeff' not in script.read_text(encoding='utf-8-sig')


@pytest.mark.skipif(os.name != 'nt' or not shutil.which('powershell.exe'), reason='Windows PowerShell required')
def test_startup_preferences_execute_in_windows_powershell():
    prefix = '\n'.join((SCRIPTS / 'start.ps1').read_text(encoding='utf-8-sig').splitlines()[:2])
    command = "$ErrorActionPreference = 'Stop'; & ([scriptblock]::Create('" + prefix.replace("'", "''") + "'))"
    result = subprocess.run(
        ['powershell.exe', '-NoProfile', '-NonInteractive', '-EncodedCommand',
         base64.b64encode(command.encode('utf-16-le')).decode('ascii')],
        capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr.decode(errors='replace')
