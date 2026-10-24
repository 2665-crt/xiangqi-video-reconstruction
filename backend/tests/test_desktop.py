from fastapi.testclient import TestClient

from app.desktop import configure_desktop_app, recording_command
from app.main import create_app


def test_recording_accepts_gui_process_startup_blank_line(tmp_path, monkeypatch):
    import json
    import sys
    import av
    from app.recording import encode_recording
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'total_frames': 1}), encoding='utf-8')
    renderer = tmp_path / 'renderer.py'
    renderer.write_text('''import json, sys
import cv2
import numpy as np
png = cv2.imencode('.png', np.full((1192, 1080, 3), 220, np.uint8))[1].tobytes()
sys.stdout.buffer.write(b'\\r\\n' + (json.dumps({'bytes': len(png), 'repeat': 1}) + '\\n').encode() + png)
''', encoding='utf-8')
    monkeypatch.setattr('app.desktop.recording_command', lambda *_: [sys.executable, str(renderer)])
    output = tmp_path / 'recording.mp4'
    encode_recording(manifest, output, lambda _: None, 'http://127.0.0.1')
    with av.open(str(output)) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == 1
        assert (frames[0].width, frames[0].height) == (1080, 1192)


def test_desktop_serves_local_interface_recording_and_api(tmp_path):
    frontend = tmp_path / 'frontend'
    frontend.mkdir()
    (frontend / 'index.html').write_text('<h1>existing interface</h1>', encoding='utf-8')
    (frontend / 'theme.css').write_text('body { color: green; }', encoding='utf-8')
    app = create_app(tmp_path / 'data', tmp_path / 'videos')
    configure_desktop_app(app, frontend, 'http://127.0.0.1:12345', 'test-token', lambda: None)
    with TestClient(app) as client:
        assert 'existing interface' in client.get('/').text
        assert 'existing interface' in client.get('/recording-render').text
        assert client.get('/theme.css').status_code == 200
        assert client.get('/api/health').json()['status'] == 'ok'
        assert client.get('/api/missing').status_code == 404
    assert app.state.recordings.render_url == 'http://127.0.0.1:12345/recording-render'


def test_desktop_shutdown_requires_launch_token(tmp_path):
    frontend = tmp_path / 'frontend'
    frontend.mkdir()
    (frontend / 'index.html').write_text('interface', encoding='utf-8')
    called = []
    app = create_app(tmp_path / 'data', tmp_path / 'videos')
    configure_desktop_app(app, frontend, 'http://127.0.0.1:12345', 'secret', lambda: called.append(True))
    with TestClient(app) as client:
        assert client.post('/api/desktop/shutdown').status_code == 403
        assert client.post('/api/desktop/shutdown', headers={'X-Desktop-Token': 'wrong'}).status_code == 403
        assert called == []
        assert client.post('/api/desktop/shutdown', headers={'X-Desktop-Token': 'secret'}).status_code == 202
        assert called == [True]


def test_recording_command_preserves_paths_with_spaces_and_unicode(monkeypatch):
    monkeypatch.setenv('XIANGQI_DESKTOP_EXE', 'C:/应用/XiangQi Vedio Reconstruction.exe')
    assert recording_command('C:/任务/a b.json', 'http://127.0.0.1:12345/recording-render') == [
        'C:/应用/XiangQi Vedio Reconstruction.exe', '--recording-render',
        'C:/任务/a b.json', 'http://127.0.0.1:12345/recording-render',
    ]


def test_desktop_data_lock_rejects_a_second_backend_and_releases_on_close(tmp_path):
    from desktop import backend_entry
    assert hasattr(backend_entry, 'acquire_data_lock')
    import pytest
    owner = backend_entry.acquire_data_lock(tmp_path)
    try:
        with pytest.raises(OSError):
            backend_entry.acquire_data_lock(tmp_path)
    finally:
        owner.close()
    reopened = backend_entry.acquire_data_lock(tmp_path)
    reopened.close()


def test_desktop_probe_checks_the_actual_lock_without_importing_models(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    from desktop import backend_entry
    command = [sys.executable, str(Path(backend_entry.__file__)), '--probe-data-root', str(tmp_path)]
    with backend_entry.acquire_data_lock(tmp_path):
        assert subprocess.run(command, capture_output=True).returncode == 73
    assert subprocess.run(command, capture_output=True).returncode == 0
