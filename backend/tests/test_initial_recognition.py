import pytest
from fastapi.testclient import TestClient


def test_initial_request_waits_for_short_status_check_lock(tmp_path, video_path, monkeypatch):
    import threading
    import app.main as main
    class Vision:
        configured = True
        usage = []
        last_response = None
        last_request_count = last_image_count = 0
        def recognize_position(self, *args, **kwargs): return []
        def close(self): pass
    monkeypatch.setattr(main, 'VisionClient', Vision)
    with TestClient(main.create_app(tmp_path)) as client:
        with video_path.open('rb') as file:
            task = client.post('/api/tasks', files={'file': ('status-lock.mp4', file, 'video/mp4')}).json()
        task = client.patch(f"/api/tasks/{task['id']}/setup", json={
            'revision': task['revision'], 'start_s': 0, 'end_s': 2,
            'calibration': {'corners': [{'x': 0, 'y': 0}, {'x': 127, 'y': 0},
                                         {'x': 127, 'y': 127}, {'x': 0, 'y': 127}]}}).json()
        lock = client.app.state.recovery.analysis_lock
        lock.acquire()
        release = threading.Timer(.2, lock.release)
        release.start()
        try:
            response = client.post(f"/api/tasks/{task['id']}/recognize-initial", json={'revision': task['revision']})
            assert response.status_code == 200
        finally:
            release.join()


def test_initial_recognition_is_one_image_and_requires_user_confirmation(tmp_path, video_path, monkeypatch):
    import app.main as main
    from app.schemas import opening_position
    from test_tracking import board_image
    monkeypatch.setattr(main, 'iter_frames', lambda *args: ((t, f) for t, f in [(0.5, board_image(opening_position()))]))
    monkeypatch.setattr(main, 'rectify_board', lambda frame, *args, **kwargs: frame)
    monkeypatch.setattr('app.initial.iter_frames', main.iter_frames)
    monkeypatch.setattr('app.initial.rectify_board', main.rectify_board)

    calls = []
    states_at_close = []
    class Vision:
        configured = True
        usage = []
        def recognize_position(self, images, actual=False):
            assert actual
            calls.append(images)
            self.usage.append({"total_tokens": 42})
            return [opening_position()]
        def close(self):
            states_at_close.append(client.app.state.store.get(task["id"]).status)
    monkeypatch.setattr(main, "VisionClient", Vision)
    with TestClient(main.create_app(tmp_path)) as client:
        with video_path.open("rb") as file:
            task = client.post("/api/tasks", files={"file": ("a.mp4", file, "video/mp4")}).json()
        task = client.patch(f"/api/tasks/{task['id']}/setup", json={
            "revision": task["revision"], "start_s": 0.5, "end_s": 2,
            "calibration": {"corners": [{"x": 0, "y": 0}, {"x": 127, "y": 0}, {"x": 127, "y": 127}, {"x": 0, "y": 127}]},
        }).json()
        url = f"/api/tasks/{task['id']}/recognize-initial"
        response = client.post(url, json={"revision": task["revision"]})
        assert response.status_code == 200
        result = response.json()
        assert len(calls) == 1 and len(calls[0]) == 1
        assert calls[0][0].shape == (1001, 901, 3)
        assert result["initial_position"] is None
        assert len(result["initial_candidates"][0]["pieces"]) == 32
        assert result["initial_frame_s"] == pytest.approx(0.5)
        assert result["usage"][-1]["total_tokens"] == 42
        assert result["analysis_stats"]["model_requests"] == 1
        assert result["analysis_stats"]["images_sent"] == 1
        assert client.get(f"/api/tasks/{task['id']}/evidence/{result['initial_evidence_ids'][0]}").status_code == 200
        assert client.post(url, json={"revision": task["revision"]}).status_code == 409
        assert len(calls) == 1
        assert states_at_close == ["uploaded"]


def test_setup_invalidates_recognition_when_board_or_range_changes(tmp_path, video_path):
    from app.main import create_app
    from app.schemas import opening_position
    with TestClient(create_app(tmp_path)) as client:
        with video_path.open("rb") as file:
            task = client.post("/api/tasks", files={"file": ("a.mp4", file, "video/mp4")}).json()
        store = client.app.state.store
        stored = store.get(task["id"])
        stored.initial_candidates = [opening_position()]
        stored.initial_frame_s = 0
        stored.processed_video_s = 1.5
        stored = store.save(stored, stored.revision)
        result = client.patch(f"/api/tasks/{stored.id}/setup", json={"revision": stored.revision, "start_s": 0.5, "end_s": 2}).json()
        assert result["initial_candidates"] == []
        assert result["initial_frame_s"] is None
        assert result["processed_video_s"] == 0.5


def test_calibration_preview_uses_selected_video_time(tmp_path, video_path):
    import cv2
    import numpy as np
    from app.main import create_app
    with TestClient(create_app(tmp_path)) as client:
        with video_path.open("rb") as file:
            task = client.post("/api/tasks", files={"file": ("a.mp4", file, "video/mp4")}).json()
        url = f"/api/tasks/{task['id']}/frame"
        response = client.get(url, params={"time_s": 1})
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        assert 55 < image.mean() < 65
        assert client.get(url, params={"time_s": 3}).status_code == 422


def test_initial_failure_preserves_usage_and_releases_recognition_lock(tmp_path, video_path, monkeypatch):
    import app.main as main
    class Vision:
        configured = True
        closed = False
        last_response = {'request_error': 'ReadTimeout', 'elapsed_s': 90.5}
        def __init__(self):
            self.usage = []
        def recognize_position(self, images, actual=False):
            self.usage.append({"total_tokens": 20})
            raise ValueError("模型输出无法解析")
        def close(self):
            self.closed = True
    monkeypatch.setattr(main, "VisionClient", Vision)
    with TestClient(main.create_app(tmp_path)) as client:
        with video_path.open("rb") as file:
            task = client.post("/api/tasks", files={"file": ("a.mp4", file, "video/mp4")}).json()
        task = client.patch(f"/api/tasks/{task['id']}/setup", json={
            "revision": task["revision"], "start_s": 0, "end_s": 2,
            "calibration": {"corners": [{"x": 0, "y": 0}, {"x": 127, "y": 0}, {"x": 127, "y": 127}, {"x": 0, "y": 127}]},
        }).json()
        for attempt in range(2):
            response = client.post(f"/api/tasks/{task['id']}/recognize-initial", json={"revision": task["revision"]})
            assert response.status_code == 200
            task = response.json()
            assert task["status"] != "analyzing"
            assert task["error"] == "模型输出无法解析"
            assert task["analysis_stats"]["model_requests"] == attempt + 1
            assert len(task["usage"]) == attempt + 1
            import json
            diagnostic = json.loads((client.app.state.store.task_dir(task['id']) / 'vision-last-response.json').read_text(encoding='utf-8'))
            assert diagnostic['request_error'] == 'ReadTimeout'


def test_legacy_analysis_requires_confirmed_initial_position(tmp_path, video_path, monkeypatch):
    import app.main as main
    class Vision:
        configured = True
        def close(self):
            pass
    monkeypatch.setattr(main, "VisionClient", Vision)
    with TestClient(main.create_app(tmp_path)) as client:
        with video_path.open("rb") as file:
            task = client.post("/api/tasks", files={"file": ("a.mp4", file, "video/mp4")}).json()
        result = client.post(f"/api/tasks/{task['id']}/analyze?actual=false")
        assert result.status_code == 409
        assert "初始棋局" in result.json()["detail"]
        assert client.get(f"/api/tasks/{task['id']}").json()["status"] == "uploaded"


def test_initial_api_persists_low_confidence_draft_until_manual_confirmation(tmp_path, video_path, monkeypatch):
    import app.main as main
    from app.schemas import Position, Piece, Square
    from test_tracking import board_image
    draft = Position(pieces=[Piece(id='r', color='red', kind='rook', square=Square(file=4, rank=3))], side_to_move='red')
    monkeypatch.setattr(main, 'iter_frames', lambda *args: ((0., board_image(draft)) for _ in range(1)))
    monkeypatch.setattr(main, 'rectify_board', lambda frame, *args, **kwargs: frame)
    monkeypatch.setattr('app.initial.iter_frames', main.iter_frames)
    monkeypatch.setattr('app.initial.rectify_board', main.rectify_board)
    review = {'square': {'file': 4, 'rank': 3}, 'color': 'red', 'prediction': 'rook', 'score': .65, 'reason': '棋种置信度不足'}
    class Vision:
        configured = True
        usage = []
        last_request_count = last_image_count = 0
        last_response = {'draft_position': draft.model_dump(), 'piece_reviews': [review], 'error': '棋种置信度不足'}
        def recognize_position(self, *args, **kwargs): return []
        def close(self): pass
    monkeypatch.setattr(main, 'VisionClient', Vision)
    with TestClient(main.create_app(tmp_path)) as client:
        with video_path.open('rb') as file:
            task = client.post('/api/tasks', files={'file': ('a.mp4', file, 'video/mp4')}).json()
        setup = {'revision': task['revision'], 'start_s': 0, 'end_s': 2,
                 'calibration': {'corners': [{'x': 0, 'y': 0}, {'x': 127, 'y': 0}, {'x': 127, 'y': 127}, {'x': 0, 'y': 127}]}}
        task = client.patch(f"/api/tasks/{task['id']}/setup", json=setup).json()
        task = client.post(f"/api/tasks/{task['id']}/recognize-initial", json={'revision': task['revision']}).json()
        assert task['initial_position'] is None and task['initial_candidates'] == []
        assert task['initial_draft'] == draft.model_dump() and task['initial_piece_reviews'] == [review | {'state': 'kind'}]
        refreshed = client.get(f"/api/tasks/{task['id']}").json()
        assert refreshed['initial_draft'] == task['initial_draft']
        setup.update(revision=task['revision'], initial_position=draft.model_dump())
        confirmed = client.patch(f"/api/tasks/{task['id']}/setup", json=setup).json()
        assert confirmed['initial_position'] == draft.model_dump()
        assert confirmed['initial_draft'] is None and confirmed['initial_piece_reviews'] == []
