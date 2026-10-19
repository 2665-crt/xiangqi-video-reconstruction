import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas import VideoMetadata, opening_position


def test_analysis_start_waits_for_brief_progress_read_lock(tmp_path, monkeypatch):
    import app.main as main
    class Process:
        def __init__(self, *args, **kwargs):pass
        def start(self):pass
        def is_alive(self):return False
        def join(self, **kwargs):pass
    monkeypatch.setattr(main.multiprocessing, 'get_context', lambda *a: SimpleNamespace(Process=Process))
    app = create_app(tmp_path)
    task = app.state.store.create('lock.mp4', VideoMetadata(width=128, height=128, duration_s=10, codec='h264'))
    task.initial_position = opening_position()
    app.state.store.save(task, task.revision)
    with TestClient(app) as client:
        lock = app.state.recovery.analysis_lock
        lock.acquire()
        release = threading.Timer(.2, lock.release)
        release.start()
        try:
            response = client.post(f'/api/tasks/{task.id}/analyze?local_only=true')
        finally:
            release.join()
        assert response.status_code == 202, response.text
