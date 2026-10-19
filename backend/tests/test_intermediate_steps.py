import cv2
import numpy as np

from app.actual import actual_events
from app.chess import board_signature
from app.schemas import Square, opening_position
from test_tracking import board_image


def test_offset_piece_faces_keep_each_intermediate_move():
    initial = opening_position()
    middle = initial.model_copy(deep=True)
    next(p for p in middle.pieces if p.id == 'red_cannon_1').square = Square(file=4, rank=7)
    final = middle.model_copy(deep=True)
    next(p for p in final.pieces if p.id == 'black_horse_1').square = Square(file=2, rank=2)
    frames = []
    for time_s, position in [(0, initial), (.2, middle), (.4, middle), (.6, final), (.8, final)]:
        image = cv2.warpAffine(board_image(position), np.float32([[1, 0, 30], [0, 1, 25]]),
                               (901, 1001), borderValue=(100, 175, 200))
        frames.append((time_s, image))
    positions = [e.position for e in actual_events(iter(frames), initial) if e.position is not None]
    assert [board_signature(p) for p in positions] == [board_signature(p) for p in [initial, middle, final]]


def test_expanded_tracking_context_keeps_each_intermediate_move():
    initial = opening_position()
    middle = initial.model_copy(deep=True)
    next(p for p in middle.pieces if p.id == 'red_cannon_1').square = Square(file=4, rank=7)
    final = middle.model_copy(deep=True)
    next(p for p in final.pieces if p.id == 'black_horse_1').square = Square(file=2, rank=2)
    frames = [(time_s, cv2.copyMakeBorder(board_image(position), 50, 50, 50, 50,
               cv2.BORDER_CONSTANT, value=(100, 175, 200)))
              for time_s, position in [(0, initial), (.2, middle), (.4, middle), (.6, final), (.8, final)]]
    positions = [e.position for e in actual_events(iter(frames), initial) if e.position is not None]
    assert [board_signature(p) for p in positions] == [board_signature(p) for p in [initial, middle, final]]


def test_partial_stable_landing_advances_next_turn_boundary():
    initial = opening_position()
    middle = initial.model_copy(deep=True)
    next(p for p in middle.pieces if p.id == 'red_cannon_1').square = Square(file=4, rank=7)
    final = middle.model_copy(deep=True)
    next(p for p in final.pieces if p.id == 'black_horse_1').square = Square(file=2, rank=2)
    frames = []
    for time_s, position in [(0, initial), (.2, middle), (.4, middle), (.6, final), (.8, final)]:
        image = board_image(position)
        cv2.rectangle(image, (300, 400), (600, 500), (20, 20, 20), -1)
        frames.append((time_s, image))
    events = list(actual_events(iter(frames), initial, turn_boundaries=True))
    departure = next(e for e in events if e.time_s == .6 and e.moving_colors == ('black',))
    assert departure.last_clear_s >= .4


def test_complete_but_wrong_color_read_does_not_create_a_move(monkeypatch):
    import app.analysis as analysis
    original = analysis.read_board_cells
    def polluted(image, strict=False):
        cells = original(image, strict)
        cells[(4, 0)] = 'red'
        return cells
    monkeypatch.setattr(analysis, 'read_board_cells', polluted)
    initial = opening_position()
    events = list(actual_events(iter([(0, board_image(initial)), (.2, board_image(initial)),
                                     (1.3, board_image(initial))]), initial))
    assert len(events) == 1
    assert events[0].position == initial


def test_resumed_tracking_aligns_from_initial_frame_without_changing_user_corners(tmp_path, monkeypatch):
    import app.actual as actual
    from app.schemas import Calibration, VideoMetadata
    from app.storage import TaskStore
    initial = opening_position()
    middle = initial.model_copy(deep=True)
    next(p for p in middle.pieces if p.id == 'red_cannon_1').square = Square(file=4, rank=7)
    final = middle.model_copy(deep=True)
    next(p for p in final.pieces if p.id == 'black_horse_1').square = Square(file=2, rank=2)
    source_frames = [(0, board_image(initial)), (.2, board_image(middle)), (.4, board_image(middle)),
                     (.6, board_image(final)), (.8, board_image(final))]
    source_frames = [(t, cv2.copyMakeBorder(f, 50, 50, 50, 50, cv2.BORDER_CONSTANT,
                     value=(100, 175, 200))) for t, f in source_frames]
    source_frames[1][1][400:501, 300:601] = (20, 20, 20)
    monkeypatch.setattr(actual, 'iter_frames', lambda path, start, end:
                        iter((t, f) for t, f in source_frames if start <= t <= end))
    correct = Calibration(corners=[{'x': 100, 'y': 100}, {'x': 900, 'y': 100},
                                   {'x': 900, 'y': 1000}, {'x': 100, 'y': 1000}])
    shifted = Calibration(corners=[{'x': 115, 'y': 115}, {'x': 915, 'y': 115},
                                   {'x': 945, 'y': 1045}, {'x': 145, 'y': 1045}])
    references = []
    def align(frame, calibration):
        references.append(frame)
        return correct
    monkeypatch.setattr('app.calibration.refine_calibration', align)
    store = TaskStore(tmp_path)
    task = store.create('resume.mp4', VideoMetadata(width=1001, height=1101, duration_s=1, codec='h264'))
    task.initial_position, task.context_position = initial, middle
    task.calibration, task.calibration_confirmed = shifted, True
    task.replay_mode, task.rules_enabled, task.status = 'actual', False, 'analyzing'
    task.resume_video_s = .2
    task = store.save(task, task.revision)
    actual.analyze_actual(task, store, None)
    saved = store.get(task.id)
    assert saved.status == 'ready'
    assert len(saved.moves) == 1
    assert saved.moves[0].move.piece_id == 'black_horse_1'
    assert saved.calibration == shifted
    assert np.array_equal(references[0], source_frames[0][1])
