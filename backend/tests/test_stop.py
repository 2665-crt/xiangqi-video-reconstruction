from fastapi.testclient import TestClient


def test_stop_terminates_worker_preserves_progress_and_unlocks_task(tmp_path):
    from app.main import create_app
    from app.schemas import VideoMetadata, opening_position
    app = create_app(tmp_path)
    store = app.state.store
    task = store.create('stop.mp4', VideoMetadata(width=128, height=128, duration_s=200, codec='h264'))
    task.status, task.processed_video_s = 'analyzing', 91.5
    task.initial_position = opening_position()
    task.analysis_stats.model_requests = 3
    task.usage = [{'total_tokens': 120}]
    store.save(task, 0)

    class Worker:
        alive = True
        terminated = False
        def is_alive(self): return self.alive
        def terminate(self): self.terminated, self.alive = True, False
        def join(self, timeout=None): pass
    worker = Worker()
    app.state.analysis_workers[task.id] = worker
    with TestClient(app) as client:
        # lifespan recovery normally only runs before a live worker exists.
        task = store.get(task.id)
        task.status = 'analyzing'
        store.save(task, task.revision)
        response = client.post(f'/api/tasks/{task.id}/stop')
        assert response.status_code == 200
        saved = response.json()
        assert worker.terminated
        assert saved['status'] == 'interrupted'
        assert saved['processed_video_s'] == 91.5
        assert saved['analysis_stats']['model_requests'] == 3
        assert saved['usage'] == [{'total_tokens': 120}]
        assert saved['initial_position']
        assert client.post(f'/api/tasks/{task.id}/stop').json()['revision'] == saved['revision']
        assert client.patch(f'/api/tasks/{task.id}/setup', json={'revision': saved['revision'], 'start_s': 75, 'end_s': 200}).status_code == 200


def test_stop_other_task_does_not_terminate_active_worker(tmp_path):
    from app.main import create_app
    from app.schemas import VideoMetadata
    app = create_app(tmp_path)
    task = app.state.store.create('other.mp4', VideoMetadata(width=128, height=128, duration_s=2, codec='h264'))
    class Worker:
        terminated = False
        def is_alive(self): return not self.terminated
        def terminate(self): self.terminated = True
        def join(self, timeout=None): pass
    worker = Worker()
    with TestClient(app) as client:
        app.state.analysis_workers['a' * 32] = worker
        assert client.post(f'/api/tasks/{task.id}/stop').status_code == 200
        assert not worker.terminated
        app.state.analysis_workers.clear()


def test_local_only_worker_never_constructs_cloud_client(tmp_path, monkeypatch):
    import app.main as main
    from app.schemas import VideoMetadata, opening_position
    app = main.create_app(tmp_path)
    task = app.state.store.create('local.mp4', VideoMetadata(width=128, height=128, duration_s=2, codec='h264'))
    task.initial_position = opening_position()
    app.state.store.save(task, 0)
    def forbidden_client():
        raise AssertionError('本地模式不应初始化云端客户端')
    monkeypatch.setattr(main, 'VisionClient', forbidden_client)
    created = []
    class Process:
        alive = False
        def __init__(self, **kwargs): self.args = kwargs['args']; created.append(self)
        def start(self): self.alive = True
        def is_alive(self): return self.alive
        def terminate(self): self.alive = False
        def join(self, timeout=None): pass
    from types import SimpleNamespace
    monkeypatch.setattr(main.multiprocessing, 'get_context', lambda method: SimpleNamespace(Process=Process))
    with TestClient(app) as client:
        response = client.post(f'/api/tasks/{task.id}/analyze?local_only=true')
        assert response.status_code == 202
        assert created[0].args[-1] is True
        assert client.post(f'/api/tasks/{task.id}/stop').json()['status'] == 'interrupted'
