import numpy as np

from app.schemas import Calibration, VideoMetadata, opening_position
from app.storage import TaskStore
from test_piece_classifier import classifier
from test_tracking import board_image


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
