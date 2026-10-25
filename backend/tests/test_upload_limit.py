import pytest
from fastapi.testclient import TestClient


def test_upload_limit_is_five_gib():
    import app.main as main
    assert getattr(main, 'MAX_VIDEO_BYTES', 2 * 1024 ** 3) == 5 * 1024 ** 3


@pytest.mark.parametrize('extra, expected', [(0, 201), (1, 413)])
def test_upload_boundary_and_cleanup(tmp_path, video_path, monkeypatch, extra, expected):
    import app.main as main
    data = video_path.read_bytes()
    monkeypatch.setattr(main, 'MAX_VIDEO_BYTES', len(data), raising=False)
    with TestClient(main.create_app(tmp_path)) as client:
        response = client.post('/api/tasks', files={'file': ('limit.mp4', data + b'\0' * extra, 'video/mp4')})
        assert response.status_code == expected
        assert not list((tmp_path / 'uploads').glob('*.mp4'))
        if expected == 413:
            assert '5 GiB' in response.json()['detail']
            assert client.get('/api/tasks').json() == []
