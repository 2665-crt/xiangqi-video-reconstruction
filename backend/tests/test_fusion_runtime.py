from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def missing_fusion_dependency(monkeypatch):
    import app.fusion
    monkeypatch.setenv('VISION_FUSION', '1')
    def unavailable():
        raise ModuleNotFoundError("No module named 'ultralytics'", name='ultralytics')
    monkeypatch.setattr(app.fusion, 'local_fusion_classifier', unavailable)


def test_missing_fusion_dependency_is_not_reported_as_ready(tmp_path, missing_fusion_dependency):
    from app.main import create_app
    with TestClient(create_app(tmp_path)) as client:
        response = client.get('/api/health')
        assert response.status_code == 200
        assert response.json()['vision_configured'] is False
        assert response.json()['status'] != 'ok'
        assert 'ultralytics' in response.json()['detail']


@pytest.mark.parametrize('operation', ['recognize-initial', 'analyze'])
def test_missing_fusion_dependency_returns_actionable_error_without_changing_task(tmp_path, missing_fusion_dependency, operation):
    from app.main import create_app
    from test_manual_replay import fixture_task
    app = create_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        task = fixture_task(app.state.store)
        from app.schemas import Calibration
        task.calibration = Calibration(corners=[{'x': 0, 'y': 0}, {'x': 127, 'y': 0}, {'x': 127, 'y': 127}, {'x': 0, 'y': 127}])
        task = app.state.store.save(task, task.revision)
        before = app.state.store.get(task.id)
        response = client.post(f'/api/tasks/{task.id}/{operation}', json={'revision': task.revision})
        assert response.status_code == 503, response.text
        assert 'ultralytics' in response.json()['detail']
        assert 'start-fusion.ps1' in response.json()['detail']
        assert app.state.store.get(task.id) == before
        assert app.state.analysis_workers == {}
