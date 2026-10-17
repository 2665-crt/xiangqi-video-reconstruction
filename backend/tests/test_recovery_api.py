import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import create_app
from app.recovery import defer_operation, prepare_recovery
from app.schemas import VideoMetadata
from app.vision import VisionRequestError


def wait_task(app):
    store = app.state.store
    task = store.create('queued.mp4', VideoMetadata(width=128, height=128, duration_s=200, codec='h264'))
    task.status = 'analyzing'
    task = prepare_recovery(task, store)
    defer_operation(store, task, VisionRequestError('offline', 'ConnectError', 'connect'))
    task = store.get(task.id)
    task.recovery.next_attempt_at = 0
    return store.save(task, task.revision)


def processes(monkeypatch):
    created = []
    class Process:
        def __init__(self, **kwargs):
            self.alive = False
            created.append(self)
        def start(self): self.alive = True
        def is_alive(self): return self.alive
        def terminate(self): self.alive = False
        def join(self, timeout=None): pass
    monkeypatch.setattr('app.main.multiprocessing.get_context', lambda method: SimpleNamespace(Process=Process))
    return created


def test_waiting_task_runs_automatically_when_due_and_only_once(tmp_path, monkeypatch):
    app = create_app(tmp_path)
    assert hasattr(app.state, 'recovery')
    created = processes(monkeypatch)
    with TestClient(app):
        task = wait_task(app)
        app.state.recovery.tick()
        app.state.recovery.tick()
        assert app.state.store.get(task.id).status == 'analyzing'
        assert len(created) == 1


def test_stop_waiting_task_cancels_automatic_restart(tmp_path):
    app = create_app(tmp_path)
    assert hasattr(app.state, 'recovery')
    with TestClient(app) as client:
        task = wait_task(app)
        stopped = client.post(f'/api/tasks/{task.id}/stop').json()
        assert stopped['status'] == 'interrupted'
        assert stopped['recovery']['auto_run'] is False
        app.state.recovery.tick()
        assert app.state.store.get(task.id).status == 'interrupted'


def test_repeated_resume_does_not_create_extra_worker(tmp_path, monkeypatch):
    app = create_app(tmp_path)
    assert hasattr(app.state, 'recovery')
    created = processes(monkeypatch)
    with TestClient(app) as client:
        task = wait_task(app)
        response = client.post(f'/api/tasks/{task.id}/resume', json={'revision': task.revision})
        assert response.status_code == 200
        app.state.recovery.tick()
        running = app.state.store.get(task.id)
        assert client.post(f'/api/tasks/{task.id}/resume', json={'revision': running.revision}).status_code == 200
        app.state.recovery.tick()
        assert len(created) == 1


def test_waiting_task_cannot_change_settings_without_cancel(tmp_path):
    app = create_app(tmp_path)
    assert hasattr(app.state, 'recovery')
    with TestClient(app) as client:
        task = wait_task(app)
        assert client.patch(f'/api/tasks/{task.id}/setup', json={'revision': task.revision, 'start_s': 2, 'end_s': 10}).status_code == 409
        stopped = client.post(f'/api/tasks/{task.id}/stop').json()
        changed = client.patch(f'/api/tasks/{task.id}/setup', json={'revision': stopped['revision'], 'start_s': 2, 'end_s': 10})
        assert changed.status_code == 200
        assert changed.json()['recovery'] is None


def test_manual_resume_does_not_bypass_503_retry_after(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        task = wait_task(app)
        task = app.state.store.claim_recovery(task.id, task.recovery.generation, time.time() + 10000)
        defer_operation(app.state.store, task, VisionRequestError('maintenance', 'HTTP503', 'response', retry_after_s=600))
        waiting = app.state.store.get(task.id)
        earliest = waiting.recovery.next_attempt_at
        resumed = client.post(f'/api/tasks/{task.id}/resume', json={'revision': waiting.revision}).json()
        assert resumed['recovery']['next_attempt_at'] >= earliest - .01


def test_initial_recovery_does_not_double_count_failed_usage(tmp_path, video_path, monkeypatch):
    import app.main as main
    from app.initial import run_initial_operation
    from app.recovery import run_saved_operation
    from app.schemas import Calibration
    from app.vision import VisionClient
    import httpx
    import shutil
    app = create_app(tmp_path)
    store = app.state.store
    task = store.create('initial.mp4', VideoMetadata(width=128, height=128, duration_s=2, codec='h264'))
    shutil.copy2(video_path, store.task_dir(task.id) / 'source.mp4')
    task.calibration = Calibration(corners=[{'x': 0, 'y': 0}, {'x': 127, 'y': 0}, {'x': 127, 'y': 127}, {'x': 0, 'y': 127}])
    (store.task_dir(task.id) / 'evidence').mkdir()
    task = prepare_recovery(task, store, 'initial')
    incomplete = 'data: {"choices":[],"usage":{"total_tokens":7}}\n\ndata: {"choices":[{"delta":{"content":"{"}}]}\n\n'
    complete = 'data: {"choices":[{"delta":{"content":"{\\\"candidates\\\":[]}"},"finish_reason":"stop"}]}\n\ndata: {"choices":[],"usage":{"total_tokens":11}}\n\ndata: [DONE]\n\n'
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=incomplete))) as client:
        run_initial_operation(task, store, VisionClient('https://example.test/v1', 'test', 'qwen3.8-omni-flash', client))
    waiting = store.get(task.id)
    assert waiting.status == 'waiting_service'
    assert sum(u.get('total_tokens', 0) for u in waiting.usage) == 7
    store.claim_recovery(task.id, waiting.recovery.generation, time.time() + 10000)
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=complete))) as client:
        run_saved_operation(task.id, store, VisionClient('https://example.test/v1', 'test', 'qwen3.8-omni-flash', client))
    finished = store.get(task.id)
    assert sum(u.get('total_tokens', 0) for u in finished.usage) == 18


def test_repeated_legacy_resume_reuses_copy_and_keeps_original(tmp_path, video_path, monkeypatch):
    import shutil
    app = create_app(tmp_path)
    processes(monkeypatch)
    with TestClient(app) as client:
        store = app.state.store
        original = store.create('old.mp4', VideoMetadata(width=128, height=128, duration_s=2, codec='h264'))
        shutil.copy2(video_path, store.task_dir(original.id) / 'source.mp4')
        original.status = 'failed'
        original = store.save(original, original.revision)
        first = client.post(f'/api/tasks/{original.id}/resume', json={'revision': original.revision})
        second = client.post(f'/api/tasks/{original.id}/resume', json={'revision': original.revision})
        assert first.status_code == second.status_code == 200
        assert first.json()['id'] == second.json()['id']
        assert first.json()['id'] != original.id
        assert store.get(original.id).model_dump() == original.model_dump()
