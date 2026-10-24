import numpy as np
import pytest

from app.schemas import Calibration, VideoMetadata, opening_position
from app.storage import TaskStore
from test_piece_classifier import classifier
from test_tracking import board_image


@pytest.mark.parametrize('diagnostic, has_local_classifier', [
    ({'draft_position': opening_position().model_dump(), 'piece_reviews': [
        {'square': {'file': 2, 'rank': 1}, 'color': None, 'prediction': None,
         'score': 0., 'reason': '占位不确定', 'state': 'occupancy'}], 'error': '起始画面占位不确定'}, True),
    ({}, True),
    (None, True),
    ({}, False),
])
def test_unestablished_initial_position_requests_review_without_scanning(tmp_path, monkeypatch, diagnostic, has_local_classifier):
    import app.actual as actual
    import app.scanning as scanning
    from app.analysis import analyze_task
    from app.main import create_app
    from app.recovery import prepare_recovery
    from fastapi.testclient import TestClient

    store = TaskStore(tmp_path)
    task = store.create('uncertain.mp4', VideoMetadata(width=901, height=1001, duration_s=1200, codec='h264'))
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50},
                                           {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    task.calibration_confirmed, task.replay_mode = True, 'actual'
    task.scan_progress_s, task.scan_events, task.pending_event_count = 600., 123, 123
    task = store.save(task, task.revision)
    task = prepare_recovery(task, store)
    scan_calls = []
    monkeypatch.setattr(scanning, 'scan_video', lambda task, store: scan_calls.append(task.id) or task)
    image = board_image(opening_position())
    monkeypatch.setattr(actual, 'find_initial_frame', lambda *a: (0., image))
    monkeypatch.setattr(actual, 'initial_frame_images', lambda *a: [image])

    class Vision:
        local_classifier = has_local_classifier
        usage = []
        last_response = diagnostic
        def recognize_position(self, *a, **k): return []
        def close(self): pass

    if diagnostic is None:
        monkeypatch.setattr('app.piece_classifier.local_piece_classifier', lambda: None)
    analyze_task(task.id, store, Vision() if diagnostic is not None else None)
    saved = store.get(task.id)
    assert scan_calls == []
    assert saved.status == 'needs_review'
    assert saved.initial_position is None and saved.moves == []
    assert saved.scan_progress_s == 600. and saved.scan_events == 123
    assert not saved.recovery.auto_run and saved.recovery.lease_token is None
    if diagnostic and 'draft_position' in diagnostic:
        assert saved.initial_draft == opening_position()
        assert saved.initial_piece_reviews[0].state == 'occupancy'
        assert saved.error == diagnostic['error']
    with TestClient(create_app(tmp_path)) as client:
        response = client.patch(f'/api/tasks/{saved.id}/setup', json={
            'revision': saved.revision, 'start_s': saved.start_s, 'end_s': saved.end_s,
            'calibration': saved.calibration.model_dump(), 'initial_position': opening_position().model_dump()})
        assert response.status_code == 200
        assert response.json()['initial_position'] is not None
        assert response.json()['scan_events'] == 123


@pytest.mark.parametrize('request_fails', [False, True])
def test_initial_review_preserves_existing_scan_when_replaying_old_seed(tmp_path, video_path, monkeypatch, request_fails):
    import shutil
    import app.actual as actual
    import app.scanning as scanning
    from app.recovery import prepare_recovery, run_saved_operation
    from app.vision import VisionRequestError

    store = TaskStore(tmp_path)
    task = store.create('existing-scan.mp4', VideoMetadata(width=128, height=128, duration_s=1200, codec='h264'))
    task.calibration = Calibration(corners=[{'x': 0, 'y': 0}, {'x': 127, 'y': 0},
                                           {'x': 127, 'y': 127}, {'x': 0, 'y': 127}])
    task.calibration_confirmed, task.replay_mode = True, 'actual'
    task = store.save(task, task.revision)
    shutil.copyfile(video_path, store.task_dir(task.id) / 'source.mp4')
    task = prepare_recovery(task, store)
    scan = scanning.VideoScan(task, store)
    scan.index.update(progress_s=1200., complete=True, events=[{'time_s': 200., 'evidence_ids': [], 'uncertain': True}])
    scan.flush()
    scan.update(task)
    task = store.save(task, task.revision)
    image = board_image(opening_position())
    monkeypatch.setattr(actual, 'find_initial_frame', lambda *a: (0., image))
    monkeypatch.setattr(actual, 'initial_frame_images', lambda *a: [image])
    scan_calls = []
    monkeypatch.setattr(scanning, 'scan_video', lambda task, store: scan_calls.append(task.id) or task)

    class Vision:
        local_classifier = None
        usage = []
        last_response = None
        def recognize_position(self, *a, **k):
            if request_fails:
                raise VisionRequestError('temporary failure', 'ReadTimeout', 'response')
            return []
        def close(self): pass

    run_saved_operation(task.id, store, Vision())
    saved = store.get(task.id)
    assert scan_calls == []
    assert saved.status == ('waiting_service' if request_fails else 'needs_review')
    assert saved.scan_progress_s == 1200. and saved.scan_complete
    assert saved.scan_events == 1 and saved.pending_event_count == 1


def test_missing_empty_square_keeps_visible_pieces_and_unknown_marker(classifier, monkeypatch):
    import app.analysis as analysis
    original = analysis.read_board_cells
    def incomplete(image, strict=False):
        cells = original(image, strict)
        cells.pop((2, 1), None)
        return cells
    monkeypatch.setattr(analysis, 'read_board_cells', incomplete)
    candidates = classifier.recognize_position([board_image(opening_position())], actual=True)
    assert candidates == []
    diagnostic = classifier.last_response
    assert len(diagnostic['draft_position']['pieces']) == 32
    assert any(r['square'] == {'file': 2, 'rank': 1} and r['state'] == 'occupancy'
               for r in diagnostic['piece_reviews'])
    assert '遮挡' not in diagnostic['error']


def test_confirmed_calibration_is_preserved_in_initial_recognition(tmp_path, monkeypatch):
    import app.initial as initial
    store = TaskStore(tmp_path)
    task = store.create('confirmed.mp4', VideoMetadata(width=901, height=1001, duration_s=1, codec='h264'))
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50},
                                           {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    task.calibration_confirmed = True
    task = store.save(task, task.revision)
    (store.task_dir(task.id) / 'evidence').mkdir()
    monkeypatch.setattr(initial, 'iter_frames', lambda *a: ((t, f) for t, f in [(0, board_image(opening_position()))]))
    monkeypatch.setattr(initial, 'initial_frame_images', lambda *a: [a[-1]])
    monkeypatch.setattr(initial, 'refine_calibration', lambda *a: (_ for _ in ()).throw(AssertionError('confirmed corners changed')))
    class Vision:
        usage = []
        last_response = None
        def recognize_position(self, images, actual=False): return []
        def close(self): pass
    saved = initial.run_initial_operation(task, store, Vision())
    assert saved.error is None
    assert saved.calibration == task.calibration


def test_confirmed_calibration_is_preserved_when_initial_occupancy_is_incomplete(tmp_path, monkeypatch):
    import app.actual as actual
    from app.analysis import analyze_task
    store = TaskStore(tmp_path)
    task = store.create('confirmed.mp4', VideoMetadata(width=901, height=1001, duration_s=1, codec='h264'))
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50},
                                           {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    task.calibration_confirmed = True
    task.initial_position, task.replay_mode = opening_position(), 'actual'
    task = store.save(task, task.revision)
    (store.task_dir(task.id) / 'source.mp4').touch()
    monkeypatch.setattr(actual, 'iter_frames', lambda *a: iter([(0, np.zeros((1001, 901, 3), np.uint8))]))
    monkeypatch.setattr(actual, 'actual_events', lambda *a, **k: iter([]))
    monkeypatch.setattr('app.calibration.refine_calibration', lambda *a: (_ for _ in ()).throw(AssertionError('confirmed corners changed')))
    analyze_task(task.id, store, None)
    assert store.get(task.id).status == 'ready'
    assert store.get(task.id).calibration == task.calibration


def test_yellow_board_background_is_not_skin_occlusion():
    from pathlib import Path
    import cv2
    from app.analysis import read_board_cells
    evidence = Path(__file__).resolve().parents[2] / 'data/tasks/856d715d27ad41f7bd4a51bf9f94fc2d/evidence/3ce448baa389a6dc0b3734714feb73f5.jpg'
    if not evidence.exists():
        import pytest
        pytest.skip('local real-board regression evidence unavailable')
    image = cv2.imdecode(np.fromfile(evidence, dtype=np.uint8), cv2.IMREAD_COLOR)
    cells = read_board_cells(image, strict=True)
    assert len(cells) == 90
    assert sum(c is not None for c in cells.values()) == 32
