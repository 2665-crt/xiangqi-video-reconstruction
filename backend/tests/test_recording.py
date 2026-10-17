import time
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from app.schemas import Move, MoveRecord, Piece, Position, Square, Task, VideoMetadata


@pytest.fixture
def recording_task():
    before = Position(side_to_move='red', pieces=[
        Piece(id='r', color='red', kind='rook', square=Square(file=0, rank=9)),
        Piece(id='p', color='black', kind='pawn', square=Square(file=0, rank=5)),
        Piece(id='b', color='black', kind='rook', square=Square(file=8, rank=0)),
    ])
    after = Position(side_to_move='black', pieces=[before.pieces[0].model_copy(update={'square': Square(file=0, rank=5)}), before.pieces[2]])
    move = Move(index=1, color='red', piece_id='r', source=before.pieces[0].square, target=after.pieces[0].square, captured_piece_id='p')
    return Task(id='a' * 32, filename='对局:样片.mp4', metadata=VideoMetadata(width=720, height=1280, duration_s=180, codec='h264'),
                start_s=10, end_s=120, processed_video_s=120, status='ready', replay_mode='actual', initial_position=before,
                moves=[MoveRecord(index=1, move=move, video_start_s=40, video_end_s=60, status='auto_verified', position_before=before, position_after=after)], confirmed_prefix_length=1)


@pytest.mark.parametrize(('interval_s', 'duration', 'frames'), [(1, 2, 120), (2, 4, 240), (3, 6, 360)])
def test_playlist_removes_source_waits_and_keeps_initial_and_final(recording_task, interval_s, duration, frames):
    from app.recording import build_recording_playlist
    payload = build_recording_playlist(recording_task, interval_s)
    assert payload['initial_position']['pieces'][0]['square'] == {'file': 0, 'rank': 9}
    assert payload['records'][-1]['position_after']['pieces'][0]['square'] == {'file': 0, 'rank': 5}
    assert payload['duration_s'] == duration
    assert payload['total_frames'] == frames


@pytest.mark.parametrize('status', ['needs_review', 'tentative'])
def test_unresolved_step_blocks_export_with_step_number(recording_task, status):
    from app.recording import build_recording_playlist
    recording_task.moves[0].status = status
    with pytest.raises(ValueError, match='第 1'):
        build_recording_playlist(recording_task, 1)


def test_initial_draft_is_not_a_confirmed_initial_position(recording_task):
    from app.recording import build_recording_playlist
    recording_task.initial_draft, recording_task.initial_position = recording_task.initial_position, None
    with pytest.raises(ValueError, match='初始'):
        build_recording_playlist(recording_task, 1)


def test_corrected_interrupted_analysis_cannot_export_missing_tail(recording_task):
    from app.recording import build_recording_playlist
    recording_task.processed_video_s = 80
    recording_task.moves[0].status = 'corrected'
    with pytest.raises(ValueError, match='完整'):
        build_recording_playlist(recording_task, 1)


def test_review_only_record_is_not_an_extra_video_step(recording_task):
    from app.recording import build_recording_playlist
    recording_task.moves.append(MoveRecord(index=2, video_start_s=61, video_end_s=90, status='corrected', change_type='review'))
    payload = build_recording_playlist(recording_task, 1)
    assert len(payload['records']) == 1
    assert payload['total_frames'] == 120


def test_missing_position_cannot_silently_skip_a_move(recording_task):
    from app.recording import build_recording_playlist
    recording_task.moves[0].position_after = None
    with pytest.raises(ValueError, match='第 1'):
        build_recording_playlist(recording_task, 1)


def test_out_of_range_record_is_rejected(recording_task):
    from app.recording import build_recording_playlist
    recording_task.moves[0].video_end_s = 121
    with pytest.raises(ValueError, match='选区'):
        build_recording_playlist(recording_task, 1)


def test_export_api_revision_and_status_persist_without_mutating_task(tmp_path, recording_task, monkeypatch):
    from app.main import create_app
    from app.recording import RecordingManager
    monkeypatch.setattr(RecordingManager, '_run', lambda self, job_id: None)
    app = create_app(tmp_path / 'data', recording_root=tmp_path / '录屏存放')
    stored = app.state.store.create(recording_task.filename, recording_task.metadata)
    recording_task.id = stored.id
    stored = app.state.store.save(recording_task, 0)
    url = f'/api/tasks/{stored.id}/recordings'
    with TestClient(app) as client:
        assert client.post(url, json={'revision': 0, 'interval_s': 1}).status_code == 409
        result = client.post(url, json={'revision': stored.revision, 'interval_s': 1})
        assert result.status_code == 202
        job = result.json()
        assert job['status'] == 'queued'
        assert job['total_frames'] == 120
        assert job['filename'].startswith('对局_样片_')
        assert client.post(url, json={'revision': stored.revision, 'interval_s': 1}).json()['id'] == job['id']
        assert client.get(f'/api/recordings/{job["id"]}').json()['id'] == job['id']
        assert client.get(f'/api/tasks/{stored.id}/recordings/latest').json()['id'] == job['id']
        assert app.state.store.get(stored.id).revision == stored.revision
        assert client.get(f'/api/recordings/{job["id"]}/video').status_code == 409


def test_failed_encoding_does_not_publish_a_video(tmp_path, recording_task, monkeypatch):
    from app.recording import RecordingManager
    def fail(*args, **kwargs):
        raise OSError('验证：无法编码')
    monkeypatch.setattr('app.recording.encode_recording', fail)
    manager = RecordingManager(tmp_path / 'jobs', tmp_path / '录屏存放')
    job = manager.start(recording_task, recording_task.revision, 1)
    deadline = time.monotonic() + 3
    while manager.get(job['id'])['status'] not in ('failed', 'completed') and time.monotonic() < deadline:
        time.sleep(.01)
    result = manager.get(job['id'])
    assert result['status'] == 'failed'
    assert result['file_path'] is None
    assert not list((tmp_path / '录屏存放').glob('*.mp4'))
    assert '无法编码' in result['error']


def test_manager_recovers_interrupted_job_as_failed(tmp_path, recording_task, monkeypatch):
    from app.recording import RecordingManager
    monkeypatch.setattr(RecordingManager, '_run', lambda self, job_id: None)
    manager = RecordingManager(tmp_path / 'jobs', tmp_path / '录屏存放')
    job = manager.start(recording_task, recording_task.revision, 1)
    reopened = RecordingManager(tmp_path / 'jobs', tmp_path / '录屏存放')
    assert reopened.get(job['id'])['status'] == 'failed'
    assert '中断' in reopened.get(job['id'])['error']


def test_shutdown_cancels_renderer_waiting_for_its_first_frame(tmp_path, recording_task, monkeypatch):
    from app.recording import RecordingManager
    original_popen = subprocess.Popen
    ready = tmp_path / 'renderer-ready'
    children = []
    def launch(args, **kwargs):
        if len(args) > 1 and str(args[1]).endswith('render-recording.cjs'):
            code = f"from pathlib import Path; import time; Path({str(ready)!r}).write_text('ready'); time.sleep(20)"
            child = original_popen([sys._base_executable, '-c', code], **kwargs)
            children.append(child)
            return child
        return original_popen(args, **kwargs)
    monkeypatch.setattr('app.recording.subprocess.Popen', launch)
    manager = RecordingManager(tmp_path / 'jobs', tmp_path / '录屏存放')
    job = manager.start(recording_task, 0, 1)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        start = time.monotonic()
        manager.close()
        assert time.monotonic() - start < 3
        assert children[0].poll() is not None
        assert not manager.threads[job['id']].is_alive()
        assert manager.get(job['id'])['status'] == 'failed'
        assert not (manager.root / job['id'] / 'video.partial.mp4').exists()
        assert not list(manager.output_root.glob('*.mp4'))
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
        for thread in manager.threads.values():
            thread.join(timeout=5)
