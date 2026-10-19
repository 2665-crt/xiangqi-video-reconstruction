import json

import httpx
import pytest
from test_chess import kings, position


@pytest.mark.parametrize("unreadable", [False, True])
def test_decoded_video_analysis_persists_real_progress_and_evidence(tmp_path, video_path, unreadable):
    from app.analysis import analyze_task
    from app.schemas import Calibration
    from app.storage import TaskStore
    from app.video import probe_video

    store = TaskStore(tmp_path / "pipeline")
    task = store.create("synthetic-boundary-test.mp4", probe_video(video_path))
    (store.task_dir(task.id) / "source.mp4").write_bytes(video_path.read_bytes())
    (store.task_dir(task.id) / "evidence").mkdir()
    task.initial_position = position(kings() + [("r", "red", "rook", 0, 9)])
    task.calibration = Calibration(corners=[{"x": 0, "y": 0}, {"x": 127, "y": 0}, {"x": 127, "y": 127}, {"x": 0, "y": 127}])
    task.status = "analyzing"
    task = store.save(task, 0)
    before = task.initial_position
    after = position(kings() + [("fresh-label", "red", "rook", 0, 5)], "black")

    class FixtureVision:
        usage = []
        calls = 0
        closed = False
        def recognize_position(self, images, reference=None):
            assert images and images[0].shape == (1001, 901, 3)
            self.calls += 1
            if unreadable:
                return [] if self.calls == 2 else [before]
            return [before if self.calls == 1 else after]
        def close(self):
            self.closed = True

    vision = FixtureVision()
    analyze_task(task.id, store, vision)
    saved = store.get(task.id)
    assert saved.processed_video_s == 2
    assert saved.status == ("needs_review" if unreadable else "ready")
    assert saved.confirmed_prefix_length == (0 if unreadable else 1)
    if not unreadable:
        assert saved.moves[0].move.piece_id == "r"
    assert saved.moves[0].evidence_ids
    assert (store.task_dir(task.id) / "evidence" / saved.moves[0].evidence_ids[0]).is_file()
    assert vision.closed
    assert saved.usage[-1]["elapsed_seconds"] >= 0


@pytest.mark.parametrize("malformed", [False, True])
def test_invalid_observation_and_format_failure_have_distinct_outcomes(tmp_path, video_path, malformed):
    from app.analysis import analyze_task
    from app.schemas import Calibration
    from app.storage import TaskStore
    from app.video import probe_video
    from app.vision import VisionClient

    store = TaskStore(tmp_path / "diagnostic-pipeline")
    task = store.create("invalid-observation.mp4", probe_video(video_path))
    (store.task_dir(task.id) / "source.mp4").write_bytes(video_path.read_bytes())
    (store.task_dir(task.id) / "evidence").mkdir()
    task.calibration = Calibration(corners=[{"x": 0, "y": 0}, {"x": 127, "y": 0}, {"x": 127, "y": 127}, {"x": 0, "y": 127}])
    task.initial_position = position(kings())
    task.status = "analyzing"
    task = store.save(task, 0)
    content = '{"candidates":[' if malformed else '{"candidates":[{"pieces":[],"side_to_move":null}]}'
    payload = {"choices": [{"finish_reason": "length" if malformed else "stop", "message": {"content": content}}],
               "usage": {"total_tokens": 20}}
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
    vision = VisionClient("https://example.test/v1", "private-placeholder-key", "test-vision", client=client)
    analyze_task(task.id, store, vision)
    saved = store.get(task.id)
    assert saved.status == ("failed" if malformed else "needs_review")
    assert saved.confirmed_prefix_length == 0
    assert list((store.task_dir(task.id) / "evidence").glob("*.jpg"))
    assert saved.usage[0] == {"total_tokens": 20}
    if malformed:
        diagnostic = (store.task_dir(task.id) / "vision-last-response.json").read_text(encoding="utf-8")
        assert json.loads(diagnostic)["parse_error"] == "invalid_json"
        assert "private-placeholder-key" not in diagnostic
    else:
        assert saved.moves[0].status == "needs_review"


@pytest.mark.parametrize("cached_initial", [False, True])
def test_static_encoded_video_has_one_request_and_preserves_historical_usage(tmp_path, cached_initial):
    import av
    import numpy as np
    from app.analysis import analyze_task
    from app.schemas import Calibration
    from app.storage import TaskStore
    from app.video import probe_video, iter_frames, save_image
    from app.calibration import rectify_board

    video = tmp_path / "static.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=12)
        stream.width = stream.height = 128
        stream.pix_fmt = "yuv420p"
        for _ in range(60):
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.full((128, 128, 3), 100, np.uint8), format="bgr24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    store = TaskStore(tmp_path / "data")
    task = store.create("static.mp4", probe_video(video))
    (store.task_dir(task.id) / "source.mp4").write_bytes(video.read_bytes())
    (store.task_dir(task.id) / "evidence").mkdir()
    task.initial_position = position(kings())
    task.calibration = Calibration(corners=[{"x": 0, "y": 0}, {"x": 127, "y": 0}, {"x": 127, "y": 127}, {"x": 0, "y": 127}])
    task.usage = [{"total_tokens": 11}]
    if cached_initial:
        frames = iter_frames(video, 0, 1)
        timestamp, image = next(frames)
        frames.close()
        save_image(store.task_dir(task.id) / "evidence" / "initial.jpg", rectify_board(image, task.calibration, margin=50))
        task.initial_evidence_ids, task.initial_frame_s = ["initial.jpg"], timestamp
    store.save(task, 0)
    class Vision:
        usage = []
        calls = 0
        def recognize_position(self, images, reference=None):
            self.calls += 1
            assert len(images) == 1
            self.usage.append({"total_tokens": 7})
            return [task.initial_position]
        def close(self):
            pass
    vision = Vision()
    analyze_task(task.id, store, vision)
    result = store.get(task.id)
    assert vision.calls == result.analysis_stats.model_requests == (0 if cached_initial else 1)
    assert result.analysis_stats.scanned_frames == 120
    assert result.analysis_stats.images_sent == (0 if cached_initial else 1)
    assert result.usage[0] == {"total_tokens": 11}
    assert sum(entry.get("total_tokens", 0) for entry in result.usage) == (11 if cached_initial else 18)


def test_decode_failure_keeps_unsaved_scanned_frames_and_progress(tmp_path, monkeypatch):
    import numpy as np
    import app.analysis as analysis
    from app.schemas import Calibration, VideoMetadata
    from app.storage import TaskStore
    store = TaskStore(tmp_path)
    task = store.create("failed.mp4", VideoMetadata(width=128, height=128, duration_s=2, codec="h264"))
    task.initial_position = position(kings())
    task.calibration = Calibration(corners=[{"x": 0, "y": 0}, {"x": 127, "y": 0}, {"x": 127, "y": 127}, {"x": 0, "y": 127}])
    store.save(task, 0)
    def frames(*args):
        for i in range(4):
            yield i * 0.02, np.full((128, 128, 3), 100, np.uint8)
        raise ValueError("视频解码失败")
    monkeypatch.setattr(analysis, "iter_frames", frames)
    class Vision:
        usage = []
        def close(self):
            pass
    analysis.analyze_task(task.id, store, Vision())
    result = store.get(task.id)
    assert result.status == "failed"
    assert result.analysis_stats.scanned_frames == 4
    assert result.processed_video_s == pytest.approx(0.06)


def test_analysis_worker_does_not_infer_an_unconfirmed_initial_position(tmp_path):
    from app.analysis import analyze_task
    from app.schemas import VideoMetadata
    from app.storage import TaskStore
    store = TaskStore(tmp_path)
    task = store.create("not-confirmed.mp4", VideoMetadata(width=128, height=128, duration_s=2, codec="h264"))
    class Vision:
        usage = []
        def close(self):
            pass
    analyze_task(task.id, store, Vision())
    result = store.get(task.id)
    assert result.status == "needs_review"
    assert "初始棋局" in result.stage


def test_clear_move_pipeline_handles_flat_response_and_stationary_letter_errors(tmp_path, video_path):
    from app.analysis import analyze_task
    from app.chess import apply_move, legal_moves
    from app.schemas import Calibration, opening_position
    from app.storage import TaskStore
    from app.video import probe_video
    from app.vision import VisionClient
    store = TaskStore(tmp_path)
    task = store.create("clear-board.mp4", probe_video(video_path))
    (store.task_dir(task.id) / "source.mp4").write_bytes(video_path.read_bytes())
    (store.task_dir(task.id) / "evidence").mkdir()
    task.calibration = Calibration(corners=[{"x": 0, "y": 0}, {"x": 127, "y": 0}, {"x": 127, "y": 127}, {"x": 0, "y": 127}])
    task.initial_position = opening_position()
    task.status = "analyzing"
    store.save(task, 0)
    move = next(m for m in legal_moves(task.initial_position) if m.source.file == 4 and m.source.rank == 6)
    after = apply_move(task.initial_position, move)
    calls = []
    def respond(request):
        board = task.initial_position if not calls else after
        pieces = [{"color": p.color, "kind": "advisor" if p.color == "red" and p.kind == "elephant" else p.kind,
                   "file": p.square.file, "rank": p.square.rank} for p in board.pieces]
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(pieces)}}],
                                        "usage": {"total_tokens": 10}})
    client = httpx.Client(transport=httpx.MockTransport(respond))
    vision = VisionClient("https://example.test/v1", "placeholder", "test-vision", client)
    analyze_task(task.id, store, vision)
    saved = store.get(task.id)
    assert saved.status == "ready" and saved.error is None
    assert saved.confirmed_prefix_length == len(saved.moves) == 1
    assert saved.moves[0].status == "auto_verified"
    assert saved.moves[0].position_after == after
    assert saved.moves[0].move.piece_id == move.piece_id
    assert saved.moves[0].evidence_ids
    assert len(saved.moves[0].position_after.pieces) == 32
