import pytest
from fastapi.testclient import TestClient


def test_task_survives_reopen(tmp_path):
    from app.storage import TaskStore
    from app.schemas import VideoMetadata
    store = TaskStore(tmp_path)
    task = store.create("测试.mp4", VideoMetadata(width=128, height=128, duration_s=2, codec="h264"))
    task.stage = "校正已保存"
    saved = store.save(task, 0)
    again = TaskStore(tmp_path).get(task.id)
    assert again.stage == "校正已保存"
    assert again.revision == saved.revision == 1


def test_stale_revision_conflict(tmp_path):
    from app.storage import TaskStore, RevisionConflict
    from app.schemas import VideoMetadata
    store = TaskStore(tmp_path)
    task = store.create("a.mp4", VideoMetadata(width=128, height=128, duration_s=2, codec="h264"))
    store.save(task, 0)
    with pytest.raises(RevisionConflict):
        store.save(task, 0)


def test_interrupted_task_on_restart(tmp_path):
    from app.storage import TaskStore
    from app.schemas import VideoMetadata
    store = TaskStore(tmp_path)
    task = store.create("a.mp4", VideoMetadata(width=128, height=128, duration_s=2, codec="h264"))
    task.status = "analyzing"
    store.save(task, 0)
    store.recover_interrupted()
    assert store.get(task.id).status == "interrupted"


def test_health_hides_configuration(tmp_path, monkeypatch):
    from app.main import create_app
    monkeypatch.setenv("VISION_API_KEY", "test-private-value")
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert "test-private-value" not in response.text
        assert "VISION_API_KEY" not in response.text


def test_upload_setup_and_reopen(tmp_path, video_path):
    from app.main import create_app
    with TestClient(create_app(tmp_path / "data")) as client:
        with video_path.open("rb") as file:
            response = client.post("/api/tasks", files={"file": ("live.mp4", file, "video/mp4")})
        assert response.status_code == 201
        task = response.json()
        assert task["metadata"]["width"] == 128
        task_id = task["id"]
        response = client.patch(f"/api/tasks/{task_id}/setup", json={"revision": 0, "start_s": 0.3, "end_s": 1.5})
        assert response.status_code == 200
        assert client.get(f"/api/tasks/{task_id}").json()["start_s"] == 0.3
        response = client.patch(f"/api/tasks/{task_id}/setup", json={"revision": 0, "start_s": 0, "end_s": 1})
        assert response.status_code == 409


def test_invalid_media_rejected(tmp_path):
    from app.main import create_app
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/tasks", files={"file": ("bad.mp4", b"not-video", "video/mp4")})
        assert response.status_code == 422
        assert "视频" in response.json()["detail"]


def test_manual_move_saves_video_time(tmp_path, video_path):
    from app.main import create_app
    from app.schemas import opening_position
    with TestClient(create_app(tmp_path)) as client:
        with video_path.open("rb") as file:
            task = client.post("/api/tasks", files={"file": ("a.mp4", file, "video/mp4")}).json()
        task = client.patch(f"/api/tasks/{task['id']}/setup", json={"revision": task["revision"], "start_s": 0, "end_s": 2,
                            "initial_position": opening_position().model_dump(by_alias=True)}).json()
        result = client.patch(f"/api/tasks/{task['id']}/moves/1", json={"revision": task["revision"], "video_time_s": 1.1,
                             "move": {"index": 1, "color": "red", "piece_id": "red_rook_0", "from": {"file": 0, "rank": 9}, "to": {"file": 0, "rank": 8}}})
        assert result.status_code == 200
        assert result.json()["moves"][0]["video_end_s"] == 1.1
