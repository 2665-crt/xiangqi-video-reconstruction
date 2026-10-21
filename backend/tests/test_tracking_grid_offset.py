import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import actual
from app.schemas import Calibration, Position, VideoMetadata
from app.storage import TaskStore


FIXTURES = Path(__file__).parent / 'fixtures/grid-offset'


@pytest.mark.parametrize('resume', [False, True])
def test_board_shift_cannot_hide_later_cannon_landing_or_create_phantom_pieces(tmp_path, monkeypatch, resume):
    data = json.loads((FIXTURES / 'reference.json').read_text(encoding='utf8'))
    initial = Position.model_validate(data['initial'])
    reference = Position.model_validate(data['before_cannon'])
    frames = [(second, cv2.imdecode(np.frombuffer((FIXTURES / f'{second}s.png').read_bytes(), np.uint8), 1))
              for second in (0, 20, 60)]
    monkeypatch.delenv('VISION_FUSION', raising=False)
    monkeypatch.setattr(actual, 'iter_frames', lambda source, start, *args: iter(frames if start < 60 else frames[-1:]))
    observed = []
    from test_move_evidence import moved
    after_cannon = moved(reference, 'red_cannon_7', 4, 7)
    def inspect_tracking(boards, *args, **kwargs):
        for time_s, board in boards:
            if time_s in (20, 60):
                expected = initial if time_s == 20 else after_cannon
                cells = actual._tracking_cells(board, expected)
                occupied = {(p.square.file, p.square.rank): p.color for p in expected.pieces}
                observed.append((time_s, [(s, col) for s, col in cells.items() if occupied.get(s) != col],
                                 cells.get((7, 7), 'unknown'), cells.get((4, 7), 'unknown')))
        return iter(())
    monkeypatch.setattr(actual, 'actual_events', inspect_tracking)
    store = TaskStore(tmp_path)
    task = store.create('another-camera.mp4', VideoMetadata(width=720, height=1270, duration_s=61, codec='h264'))
    task.initial_position, task.replay_mode, task.rules_enabled = initial, 'actual', True
    task.calibration = Calibration.model_validate(data['aligned_calibration'])
    task.calibration_confirmed = True
    if resume:
        task.resume_video_s = 60
        task.context_position = reference
    store.save(task, 0)
    actual.analyze_actual(task, store, None)
    result = store.get(task.id)
    assert result.status != 'failed', result.error
    assert len(observed) == (1 if resume else 2)
    assert all(not mismatches for _, mismatches, _, _ in observed), observed
    landing = next(item for item in observed if item[0] == 60)
    assert landing[3] == 'red'
    assert landing[2] in (None, 'unknown')  # 手部仍遮住起点时保留未知。
    assert result.calibration == task.calibration


def test_compatible_partial_frame_replaces_a_contradictory_start_picture():
    from app.actual import actual_events, snapshot_record
    from app.schemas import Square, opening_position
    from test_move_evidence import cells, moved
    from test_tracking import board_image
    before = opening_position()
    after = moved(before, 'red_pawn_0', 0, 5)
    first, partial, landing = board_image(before), board_image(before), board_image(after)
    noisy = cells(before)
    noisy[(8, 5)] = 'black'
    clear = cells(before)
    clear.pop((8, 9))
    final = cells(after)
    final.pop((8, 9))
    observed = {id(first): noisy, id(partial): clear, id(landing): final}
    reference, moves = [before], []
    for event in actual_events(iter([(0, first), (1, partial), (2, landing), (2.1, landing)]), before,
            confirmed_position=lambda: reference[0], turn_boundaries=True,
            cell_reader=lambda image, _: observed[id(image)]):
        if event.position is None or event.review_reason or event.moving_colors:
            continue
        if event.board_check is not None and event.board_check.status == 'conflict':
            continue
        action = snapshot_record(reference[0], event.position, 1, 0, event.time_s).move
        if action is not None:
            moves.append(action)
        reference[0] = event.position
    assert len(moves) == 1
    assert moves[0].source == Square(file=0, rank=6)
    assert moves[0].target == Square(file=0, rank=5)


def test_clear_before_evidence_is_kept_when_a_later_compatible_frame_is_partly_occluded():
    from app.schemas import opening_position
    from test_move_evidence import cells, moved
    from test_tracking import board_image
    before = opening_position()
    after = moved(before, 'red_pawn_0', 0, 5)
    first, partial, landing = board_image(before), board_image(before), board_image(after)
    visible = cells(before)
    visible.pop((8, 9))
    final = cells(after)
    final.pop((8, 9))
    observed = {id(first): cells(before), id(partial): visible, id(landing): final}
    reference, checks = [before], []
    for event in actual.actual_events(iter([(0, first), (1, partial), (2, landing), (2.1, landing)]), before,
            confirmed_position=lambda: reference[0], turn_boundaries=True,
            cell_reader=lambda image, _: observed[id(image)]):
        if event.position is not None and actual.snapshot_record(before, event.position, 1, 0, event.time_s).move:
            checks.append(event.board_check)
            reference[0] = event.position
    assert len(checks) == 1
    assert checks[0].visible_before == 90
    assert checks[0].visible_after == 89


def test_backfill_uses_the_same_board_motion_mapping_as_forward_tracking(monkeypatch):
    data = json.loads((FIXTURES / 'reference.json').read_text(encoding='utf8'))
    initial = Position.model_validate(data['initial'])
    frames = [(index, cv2.imdecode(np.frombuffer((FIXTURES / f'{second}s.png').read_bytes(), np.uint8), 1))
              for index, second in enumerate((0, 20))]
    monkeypatch.setattr(actual, 'iter_frames', lambda *a: iter(frames))
    observed = []
    def inspect(boards, *args, **kwargs):
        occupied = {(p.square.file, p.square.rank): p.color for p in initial.pieces}
        for _, image in boards:
            observed.append(actual._tracking_cells(image, initial))
        assert all(occupied.get(s) == color for cells in observed for s, color in cells.items())
        return iter(())
    monkeypatch.setattr(actual, 'actual_events', inspect)
    actual.backfill_interval('unused.mp4', 0, 1, Calibration.model_validate(data['aligned_calibration']), initial, initial)
    assert len(observed) == 2


def test_checkpoint_saves_tracking_coordinates_without_replacing_manual_calibration(tmp_path):
    from app.checkpoint import CheckpointStore
    from app.recovery import prepare_recovery, save_boundary_checkpoint
    data = json.loads((FIXTURES / 'reference.json').read_text(encoding='utf8'))
    store = TaskStore(tmp_path)
    task = store.create('moving-board.mp4', VideoMetadata(width=720, height=1270, duration_s=100, codec='h264'))
    task.initial_position = Position.model_validate(data['initial'])
    task.calibration = Calibration.model_validate(data['calibration'])
    task = store.save(task, task.revision)
    task = prepare_recovery(task, store)
    tracked = Calibration.model_validate(data['aligned_calibration'])
    save_boundary_checkpoint(store, task, 20, task.initial_position, tracked, True, 18)
    saved = store.get(task.id)
    state = CheckpointStore(store.root).load(task.id, saved.recovery.generation, saved.recovery.checkpoint)
    seed = type(task).model_validate(state['seed'])
    assert seed.resume_calibration == tracked
    assert seed.resume_motion_active is True
    assert seed.resume_motion_reference_s == 18
    assert seed.calibration == saved.calibration == task.calibration


def test_checkpoint_retains_recent_reference_mapping_for_unapplied_motion(tmp_path):
    from app.checkpoint import CheckpointStore
    from app.recovery import prepare_recovery, save_boundary_checkpoint
    from app.schemas import opening_position
    from test_board_motion import scene
    _, original = scene()
    reference = original.model_copy(deep=True)
    for corner in reference.corners:
        corner.x += 4
    store = TaskStore(tmp_path)
    task = store.create('slow-camera.mp4', VideoMetadata(width=640, height=480, duration_s=30, codec='h264'))
    task.calibration, task.initial_position = original, opening_position()
    task = store.save(task, task.revision)
    task = prepare_recovery(task, store)
    save_boundary_checkpoint(store, task, 12, task.initial_position, original, False, 10, reference)
    saved = store.get(task.id)
    state = CheckpointStore(store.root).load(task.id, saved.recovery.generation, saved.recovery.checkpoint)
    seed = type(task).model_validate(state['seed'])
    assert seed.resume_calibration == original
    assert seed.resume_motion_reference_calibration == reference
    assert seed.calibration == original


def test_checkpoint_uses_event_time_coordinates_when_later_frames_have_been_prefetched(tmp_path, monkeypatch):
    from app.analysis import VideoEvent
    from app.schemas import opening_position
    from test_board_motion import scene
    from test_move_evidence import cells, moved
    before = opening_position()
    red = moved(before, 'red_pawn_0', 0, 5)
    black = moved(red, 'black_pawn_0', 0, 4)
    image, calibration = scene()
    frames = [(n, cv2.warpAffine(image, np.float32([[1, 0, n * 9], [0, 1, 0]]), (640, 480))) for n in range(4)]
    monkeypatch.delenv('VISION_FUSION', raising=False)
    monkeypatch.setattr(actual, 'iter_frames', lambda *a: iter(frames))
    monkeypatch.setattr(actual, '_tracking_cells', lambda image, reference=None: cells(reference or before))
    def events(boards, *args, **kwargs):
        prepared = list(boards)
        yield VideoEvent(0, [prepared[0][1]], position=before)
        yield VideoEvent(1, [prepared[0][1], prepared[1][1]], position=red, action_start_s=.5)
        yield VideoEvent(2, [prepared[1][1], prepared[2][1]], position=black, action_start_s=1.5)
    monkeypatch.setattr(actual, 'actual_events', events)
    checkpoints = []
    monkeypatch.setattr('app.recovery.save_boundary_checkpoint',
                        lambda store, task, time_s, context, tracked, active, reference_s, reference_calibration: checkpoints.append((time_s, tracked, active)))
    store = TaskStore(tmp_path)
    task = store.create('prefetch.mp4', VideoMetadata(width=640, height=480, duration_s=4, codec='h264'))
    task.initial_position, task.calibration = before, calibration
    task.replay_mode, task.rules_enabled = 'actual', False
    store.save(task, 0)
    actual.analyze_actual(task, store, None)
    result = store.get(task.id)
    assert result.status == 'ready', result.error
    tracked, active = next((cal, active) for time_s, cal, active in checkpoints if time_s == 1)
    assert active is True
    np.testing.assert_allclose([p.x for p in tracked.corners], [p.x + 9 for p in calibration.corners], atol=.4)


def test_backfill_keeps_already_active_motion_tracking(monkeypatch):
    from app.schemas import opening_position
    from test_board_motion import scene
    image, original = scene()
    saved = original.model_copy(deep=True)
    for point in saved.corners:
        point.x += 9
    frames = [(n, cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, 0]]), (640, 480)))
              for n, dx in ((1, 9), (2, 14))]
    monkeypatch.setattr(actual, 'iter_frames', lambda *a: iter(frames))
    mapped = []
    def rectify(frame, calibration, **kwargs):
        mapped.append(calibration)
        return frame
    monkeypatch.setattr(actual, 'rectify_board', rectify)
    monkeypatch.setattr(actual, 'actual_events', lambda boards, *a, **kw: list(boards) and iter(()))
    actual.backfill_interval('moving.mp4', 1, 2, saved, opening_position(), opening_position(), motion_active=True)
    np.testing.assert_allclose([p.x for p in mapped[-1].corners], [p.x + 14 for p in original.corners], atol=1)


def test_backfill_restores_the_reference_before_an_unapplied_small_shift(monkeypatch):
    from app.schemas import opening_position
    from test_board_motion import scene
    image, calibration = scene()
    frames = [(n, cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, 0]]), (640, 480)))
              for n, dx in ((0, 0), (1, 4), (2, 8))]
    monkeypatch.setattr(actual, 'iter_frames', lambda source, start, end: iter((t, f) for t, f in frames if start <= t <= end))
    mapped = []
    def rectify(frame, calibration, **kwargs):
        mapped.append(calibration)
        return frame
    monkeypatch.setattr(actual, 'rectify_board', rectify)
    monkeypatch.setattr(actual, 'actual_events', lambda boards, *a, **kw: list(boards) and iter(()))
    actual.backfill_interval('moving.mp4', 1, 2, calibration, opening_position(), opening_position(), motion_reference_s=0)
    np.testing.assert_allclose([p.x for p in mapped[-1].corners], [p.x + 8 for p in calibration.corners], atol=1)


def test_resumed_analysis_restores_the_reference_before_an_unapplied_small_shift(tmp_path, monkeypatch):
    from app.schemas import opening_position
    from test_board_motion import scene
    from test_move_evidence import cells
    image, calibration = scene()
    frames = [(n, cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, 0]]), (640, 480)))
              for n, dx in ((0, 0), (1, 4), (2, 8))]
    monkeypatch.delenv('VISION_FUSION', raising=False)
    monkeypatch.setattr(actual, 'iter_frames', lambda source, start, end: iter((t, f) for t, f in frames if start <= t <= end))
    mapped = []
    def rectify(frame, calibration, **kwargs):
        mapped.append(calibration)
        return frame
    monkeypatch.setattr(actual, 'rectify_board', rectify)
    monkeypatch.setattr(actual, '_tracking_cells', lambda image, reference=None: cells(reference or opening_position()))
    monkeypatch.setattr(actual, 'actual_events', lambda boards, *a, **kw: list(boards) and iter(()))
    store = TaskStore(tmp_path)
    task = store.create('resumed-moving-board.mp4', VideoMetadata(width=640, height=480, duration_s=2, codec='h264'))
    task.initial_position = opening_position()
    task.calibration = task.resume_calibration = calibration
    task.resume_video_s, task.resume_motion_reference_s = 1, 0
    task.replay_mode, task.rules_enabled = 'actual', False
    store.save(task, 0)
    actual.analyze_actual(task, store, None)
    result = store.get(task.id)
    assert result.status == 'ready', result.error
    np.testing.assert_allclose([p.x for p in mapped[-1].corners], [p.x + 8 for p in calibration.corners], atol=1)
    assert result.calibration == calibration


def test_fresh_analysis_ignores_motion_reference_left_by_an_older_resume(tmp_path, monkeypatch):
    from app.schemas import opening_position
    from test_board_motion import scene
    from test_move_evidence import cells
    image, calibration = scene()
    old_frame = cv2.warpAffine(image, np.float32([[1, 0, 9], [0, 1, 0]]), (640, 480))
    frames = [(0, old_frame), (1, image), (2, image)]
    monkeypatch.delenv('VISION_FUSION', raising=False)
    monkeypatch.setattr(actual, 'iter_frames', lambda source, start, end: iter((t, f) for t, f in frames if start <= t <= end))
    mapped = []
    def rectify(frame, calibration, **kwargs):
        mapped.append(calibration)
        return frame
    monkeypatch.setattr(actual, 'rectify_board', rectify)
    monkeypatch.setattr(actual, '_tracking_cells', lambda image, reference=None: cells(reference or opening_position()))
    monkeypatch.setattr(actual, 'actual_events', lambda boards, *a, **kw: list(boards) and iter(()))
    store = TaskStore(tmp_path)
    task = store.create('restarted.mp4', VideoMetadata(width=640, height=480, duration_s=2, codec='h264'))
    task.start_s = task.initial_frame_s = 1
    task.initial_position = opening_position()
    task.calibration = task.resume_calibration = calibration
    task.resume_motion_reference_s = 0
    task.replay_mode, task.rules_enabled = 'actual', False
    store.save(task, 0)
    actual.analyze_actual(task, store, None)
    result = store.get(task.id)
    assert result.status == 'ready', result.error
    assert all(c == calibration for c in mapped)
