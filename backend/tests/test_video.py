import numpy as np
from fastapi.testclient import TestClient


def test_range_keeps_absolute_time(video_path):
    from app.video import iter_frames, probe_video
    metadata = probe_video(video_path)
    assert metadata.duration_s == 2
    frames = list(iter_frames(video_path, 0.5, 1))
    assert abs(frames[0][0] - 0.5) < 0.01
    assert frames[-1][0] <= 1
    assert all(b[0] > a[0] for a, b in zip(frames, frames[1:]))


def test_video_range_request(tmp_path, video_path):
    from app.main import create_app
    with TestClient(create_app(tmp_path)) as client:
        with video_path.open("rb") as file:
            task = client.post("/api/tasks", files={"file": ("a.mp4", file, "video/mp4")}).json()
        result = client.get(f"/api/tasks/{task['id']}/video", headers={"Range": "bytes=10-29"})
        assert result.status_code == 206
        assert result.content == video_path.read_bytes()[10:30]


def test_uploaded_preview_exists_on_chinese_path(tmp_path, video_path):
    from app.main import create_app
    with TestClient(create_app(tmp_path / "中文证据目录")) as client:
        with video_path.open("rb") as file:
            task = client.post("/api/tasks", files={"file": ("测试.mp4", file, "video/mp4")}).json()
        response = client.get(f"/api/tasks/{task['id']}/evidence/preview.jpg")
        assert response.status_code == 200
        assert response.content[:2] == b"\xff\xd8"


def test_rotation_maps_intersections():
    from app.calibration import rectify_board
    from app.schemas import Calibration
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[0:20, 0:20] = (10, 30, 200)
    normal = Calibration(corners=[{"x": 0, "y": 0}, {"x": 99, "y": 0}, {"x": 99, "y": 99}, {"x": 0, "y": 99}])
    result = rectify_board(frame, normal)
    assert tuple(result[0, 0]) == (10, 30, 200)
    rotated = normal.model_copy(update={"orientation": "rotated_180"})
    result = rectify_board(frame, rotated)
    assert tuple(result[-1, -1]) == (10, 30, 200)


def test_degenerate_calibration_rejected():
    from app.calibration import validate_calibration
    from app.schemas import Calibration
    import pytest
    with pytest.raises(ValueError):
        validate_calibration(Calibration(corners=[{"x": 1, "y": 1}] * 4), 100, 100)


def test_variable_timestamps_preserved(tmp_path):
    import av
    from fractions import Fraction
    from app.video import iter_frames
    path = tmp_path / "variable.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 128, 128, "yuv420p"
        for pts in [0, 1, 3, 7]:
            frame = av.VideoFrame.from_ndarray(np.zeros((128, 128, 3), np.uint8), format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 10)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    times = [round(t, 3) for t, _ in iter_frames(path, 0, 1)]
    assert times == [0.0, 0.1, 0.3, 0.7]
def test_recognition_crop_preserves_outer_piece_edges():
    import numpy as np
    from app.calibration import rectify_board
    from app.schemas import Calibration
    image = np.zeros((140, 140, 3), dtype=np.uint8)
    image[16:24, 16:24] = 255
    corners = Calibration(corners=[{"x": 20, "y": 20}, {"x": 120, "y": 20}, {"x": 120, "y": 120}, {"x": 20, "y": 120}])
    crop = rectify_board(image, corners, margin=50)
    assert crop.shape == (1001, 901, 3)
    assert crop[35:45, 35:45].mean() > 200

