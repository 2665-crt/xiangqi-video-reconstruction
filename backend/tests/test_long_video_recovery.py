import pytest

from app.actual import actual_events
from app.chess import board_signature
from app.schemas import Square, Task, VideoMetadata, opening_position
from app.turns import TurnReplay
from test_tracking import board_image


def moved(before, identity, file, rank):
    after = before.model_copy(deep=True)
    next(p for p in after.pieces if p.id == identity).square = Square(file=file, rank=rank)
    return after


def test_rejected_candidate_does_not_poison_next_before_image():
    before = opening_position()
    rejected = moved(before, 'red_rook_0', 0, 8)
    actual = moved(before, 'red_rook_0', 0, 7)
    reference = [before]
    events = []
    frames = [(0, before), (1, rejected), (1.1, rejected), (2, actual), (2.1, actual)]
    for event in actual_events(iter((t, board_image(p)) for t, p in frames), before,
                               lambda: reference[0], turn_boundaries=True):
        events.append(event)
        if event.position is not None and board_signature(event.position) == board_signature(actual):
            reference[0] = actual
    recovered = next(e for e in events if e.position and board_signature(e.position) == board_signature(actual))
    assert (recovered.images[0] == board_image(before)).all()


def test_rejected_candidate_can_be_retried_after_returning_to_trusted_board():
    before = opening_position()
    after = moved(before, 'red_rook_0', 0, 8)
    reference = [before]
    attempts = []
    frames = [(0, before), (1, after), (1.1, after), (2, before), (3, after), (3.1, after), (3.2, after)]
    for event in actual_events(iter((t, board_image(p)) for t, p in frames), before,
                               lambda: reference[0], turn_boundaries=True):
        if event.position is not None and board_signature(event.position) == board_signature(after):
            attempts.append(event)
            if event.time_s >= 3:
                reference[0] = after
    assert [e.time_s for e in attempts] == [1, 3]
    assert board_signature(reference[0]) == board_signature(after)


def test_one_unexplained_interval_does_not_become_hundreds_of_steps():
    before = opening_position()
    task = Task(id='a' * 32, filename='long.mp4', end_s=2400,
                metadata=VideoMetadata(width=901, height=1001, duration_s=2400, codec='h264'),
                initial_position=before, rules_enabled=True)
    replay = TurnReplay(task, before)
    for t in range(170, 1000):
        replay.uncertain(t, [], [f'{t}.jpg'], '局部画面持续出现未解释的棋子变化' if t % 2 else '落点遮挡')
    replay.end(2400)
    assert len(task.moves) == 1
    assert len(task.moves[0].evidence_ids) <= 6
    assert task.moves[0].video_start_s == 0
    assert task.moves[0].video_end_s == 2400


def test_issue_after_pending_move_uses_latest_observed_state_and_times():
    before = opening_position()
    after = moved(before, 'red_rook_0', 0, 8)
    task = Task(id='b' * 32, filename='long.mp4', end_s=5,
                metadata=VideoMetadata(width=901, height=1001, duration_s=5, codec='h264'),
                initial_position=before, rules_enabled=True)
    replay = TurnReplay(task, before)
    replay.observe(after, 2, ['landing.jpg'], 1)
    replay.uncertain(3, [], ['unknown.jpg'], '失配')
    assert board_signature(replay.unresolved.position_before) == board_signature(after)
    assert replay.unresolved.video_start_s >= 2


def test_backfill_can_recover_after_a_transient_rejected_candidate(monkeypatch):
    from app import actual as module
    before = opening_position()
    rejected = moved(before, 'red_rook_0', 1, 8)
    final = moved(before, 'red_rook_0', 0, 8)
    frames = [(0, before), (1, rejected), (1.3, rejected), (2, final), (2.1, final)]
    monkeypatch.setattr(module, 'iter_frames', lambda *args: iter((t, board_image(p)) for t, p in frames))
    monkeypatch.setattr(module, 'rectify_board', lambda image, *args, **kwargs: image)
    result = module.backfill_interval('clip.mp4', 0, 2.1, None, before, final)
    assert len(result) == 1
    assert board_signature(result[0].position) == board_signature(final)
    assert result[0].board_check.status == 'verified'
    assert result[0].image_times_s[0] == 0


def test_recovered_reference_keeps_its_verified_picture_when_video_has_advanced():
    before = opening_position()
    recovered = moved(before, 'red_rook_0', 0, 8)
    later = moved(recovered, 'black_rook_0', 0, 1)
    reference = [before, board_image(before)]
    events = actual_events(iter([(0, board_image(before)), (2, board_image(later)), (2.1, board_image(later))]),
                           before, lambda: reference[0], turn_boundaries=True,
                           confirmed_image=lambda: reference[1])
    next(events)
    reference[:] = [recovered, board_image(recovered)]
    event = next(e for e in events if e.position and board_signature(e.position) == board_signature(later))
    assert (event.images[0] == board_image(recovered)).all()


def test_actual_mode_also_groups_uncertain_observations(tmp_path, monkeypatch):
    from app import actual as module
    from app.analysis import VideoEvent, analyze_task
    from app.schemas import Calibration
    from app.storage import TaskStore
    before = opening_position()
    candidate = moved(before, 'red_rook_0', 1, 8)
    events = [VideoEvent(0, [board_image(before)], position=before)]
    events += [VideoEvent(t, [board_image(before), board_image(candidate)], True,
                         position=candidate if t % 2 else None, review_reason='失配') for t in range(1, 20)]
    monkeypatch.setattr(module, 'actual_events', lambda *a, **k: iter(events))
    monkeypatch.setattr('app.scanning.scan_video', lambda task, store: task)
    store = TaskStore(tmp_path)
    task = store.create('actual.mp4', VideoMetadata(width=901, height=1001, duration_s=20, codec='h264'))
    task.replay_mode, task.initial_position = 'actual', before
    task.calibration = Calibration(corners=[{'x':50,'y':50},{'x':850,'y':50},{'x':850,'y':950},{'x':50,'y':950}])
    store.save(task, 0)
    analyze_task(task.id, store, None)
    result = store.get(task.id)
    assert len(result.moves) == 1
    assert len(result.moves[0].evidence_ids) <= 6


def test_eof_unknown_after_observed_motion_keeps_moving_step_and_original_base():
    from app.records import movement_records
    from app.schemas import BoardCheck
    before = opening_position()
    after = moved(before, 'red_rook_0', 0, 8)
    task = Task(id='d' * 32, filename='long.mp4', end_s=5,
                metadata=VideoMetadata(width=901, height=1001, duration_s=5, codec='h264'),
                initial_position=before, rules_enabled=True)
    replay = TurnReplay(task, before)
    replay.observe(after, 2, ['before.jpg', 'landing.jpg'], 1)
    replay.attach_evidence(BoardCheck(status='verified', visible_before=90, visible_after=90, reason='证据一致'),
                           ['before.jpg', 'landing.jpg'], [0, 2])
    replay.uncertain(3, [], ['unknown.jpg'], '最终落点尚未确定')
    replay.end(5)
    assert len(movement_records(task)) == 1
    record = task.moves[0]
    assert record.status == 'needs_review'
    assert board_signature(record.position_before) == board_signature(before)
    assert record.candidates[0].source == Square(file=0, rank=9)
    assert record.candidates[0].target == Square(file=0, rank=8)
    assert record.action_end_s is None


def test_accepting_one_candidate_does_not_erase_other_unexplained_observations(tmp_path, monkeypatch):
    from app import actual as module
    from app.analysis import VideoEvent, analyze_task
    from app.schemas import Calibration
    from app.storage import TaskStore
    before = opening_position()
    earlier = moved(before, 'red_rook_0', 0, 7)
    final = moved(before, 'red_rook_0', 0, 8)
    events = [VideoEvent(0, [board_image(before)], position=before),
              VideoEvent(1, [board_image(before), board_image(earlier)], True, position=earlier, review_reason='局部未解释'),
              VideoEvent(2, [board_image(before), board_image(final)], True, position=final, review_reason='落点遮挡'),
              VideoEvent(3, [board_image(before), board_image(final)], position=final, action_start_s=2)]
    monkeypatch.setattr(module, 'actual_events', lambda *a, **k: iter(events))
    monkeypatch.setattr('app.scanning.scan_video', lambda task, store: task)
    store = TaskStore(tmp_path)
    task = store.create('actual.mp4', VideoMetadata(width=901, height=1001, duration_s=4, codec='h264'))
    task.replay_mode, task.initial_position = 'actual', before
    task.calibration = Calibration(corners=[{'x':50,'y':50},{'x':850,'y':50},{'x':850,'y':950},{'x':50,'y':950}])
    store.save(task, 0)
    analyze_task(task.id, store, None)
    result = store.get(task.id)
    assert result.status == 'needs_review'
    assert [r.status for r in result.moves] == ['needs_review', 'auto_verified']
    assert len(result.moves[0].observed_positions) == 2


def test_rejected_opponent_hint_cannot_split_one_sliding_piece_into_two_turns():
    before = opening_position()
    task = Task(id='e' * 32, filename='slide.mp4', end_s=4,
                metadata=VideoMetadata(width=901, height=1001, duration_s=4, codec='h264'),
                initial_position=before, rules_enabled=True)
    replay = TurnReplay(task, before)
    replay.observe(moved(before, 'red_rook_0', 0, 8), 1, ['middle.jpg'], .5)
    replay.movement(('black',), 1.1)
    replay.uncertain(1.2, [], ['rejected.jpg'], '整盘观察包含多枚棋子或身份变化')
    replay.observe(moved(before, 'red_rook_0', 0, 7), 1.5, ['final.jpg'], .5)
    replay.end(4)
    assert len(task.moves) == 1
    assert task.moves[0].status == 'auto_verified'
    assert task.moves[0].move.source == Square(file=0, rank=9)
    assert task.moves[0].move.target == Square(file=0, rank=7)


def test_same_piece_continuation_keeps_independent_opponent_change_evidence():
    before = opening_position()
    task = Task(id='f' * 32, filename='slide.mp4', end_s=4,
                metadata=VideoMetadata(width=901, height=1001, duration_s=4, codec='h264'),
                initial_position=before, rules_enabled=True)
    replay = TurnReplay(task, before)
    middle = moved(before, 'red_rook_0', 0, 8)
    replay.observe(middle, 1, ['middle.jpg'], .5)
    replay.movement(('black',), 1.1)
    conflict = moved(moved(middle, 'black_rook_0', 0, 1), 'black_rook_8', 8, 1)
    replay.uncertain(1.2, [conflict], ['black-multi.jpg'], '整盘观察包含多枚棋子或身份变化')
    replay.observe(moved(before, 'red_rook_0', 0, 7), 1.5, ['final.jpg'], .5)
    replay.end(4)
    assert any('black-multi.jpg' in r.evidence_ids and r.status == 'needs_review' for r in task.moves)
    actual = next(r for r in task.moves if r.move is not None)
    assert actual.move.source == Square(file=0, rank=9)
    assert actual.move.target == Square(file=0, rank=7)


def test_bounded_observation_history_cannot_forget_independent_changes():
    before = opening_position()
    task = Task(id='a1' * 16, filename='slide.mp4', end_s=4,
                metadata=VideoMetadata(width=901, height=1001, duration_s=4, codec='h264'),
                initial_position=before, rules_enabled=True)
    replay = TurnReplay(task, before)
    middle = moved(before, 'red_rook_0', 0, 8)
    replay.observe(middle, 1, ['middle.jpg'], .5)
    conflict = moved(moved(middle, 'black_rook_0', 0, 1), 'black_rook_8', 8, 1)
    observations = [moved(before, 'red_rook_0', 0, 9), conflict,
                    moved(before, 'red_rook_0', 0, 7), moved(before, 'red_rook_0', 0, 5)]
    for i, position in enumerate(observations):
        replay.uncertain(1.1 + i * .1, [position], ['black-multi.jpg' if i == 1 else f'{i}.jpg'], '局部未解释变化')
    assert len(replay.unresolved.observed_positions) <= len(replay.unresolved.review_changed_piece_ids) + 2
    replay.observe(moved(before, 'red_rook_0', 0, 7), 1.5, ['final.jpg'], .5)
    replay.end(4)
    assert any('black-multi.jpg' in r.evidence_ids and r.status == 'needs_review' for r in task.moves)


def test_resolving_new_noise_cannot_delete_queued_independent_changes():
    before = opening_position()
    task = Task(id='b1' * 16, filename='slide.mp4', end_s=4,
                metadata=VideoMetadata(width=901, height=1001, duration_s=4, codec='h264'),
                initial_position=before, rules_enabled=True)
    replay = TurnReplay(task, before)
    middle = moved(before, 'red_rook_0', 0, 8)
    replay.observe(middle, 1, ['middle.jpg'], .5)
    conflict = moved(moved(middle, 'black_rook_0', 0, 1), 'black_rook_8', 8, 1)
    replay.uncertain(1.2, [conflict], ['black-multi.jpg'], '局部画面持续出现未解释的棋子变化')
    replay.observe(moved(before, 'red_rook_0', 0, 7), 1.5, ['final.jpg'], .5)
    replay.uncertain(1.6, [], ['noise.jpg'], '局部画面持续出现未解释的棋子变化')
    replay.observe(middle, 2, ['return.jpg'], .5)
    replay.end(4)
    assert any('black-multi.jpg' in r.evidence_ids and r.status == 'needs_review' for r in task.moves)


def test_rejected_candidate_cannot_corrupt_last_selected_move_extension(tmp_path, monkeypatch):
    from app import actual as module
    from app.analysis import VideoEvent, analyze_task
    from app.schemas import Calibration
    from app.storage import TaskStore
    before = opening_position()
    wrong = moved(before, 'black_rook_0', 0, 1)
    actual = moved(before, 'red_rook_0', 0, 8)
    events = [VideoEvent(0, [board_image(before)], position=before),
              VideoEvent(1.5, [board_image(before), board_image(before)], position=wrong, action_start_s=1.4),
              VideoEvent(2.2, [board_image(before), board_image(actual)], position=actual, action_start_s=1.9)]
    monkeypatch.setattr(module, 'actual_events', lambda *a, **k: iter(events))
    monkeypatch.setattr('app.scanning.scan_video', lambda task, store: task)
    store = TaskStore(tmp_path)
    task = store.create('tail.mp4', VideoMetadata(width=901, height=1001, duration_s=5, codec='h264'))
    task.end_s, task.replay_mode, task.rules_enabled, task.initial_position = 2, 'actual', True, before
    task.calibration = Calibration(corners=[{'x':50,'y':50},{'x':850,'y':50},{'x':850,'y':950},{'x':50,'y':950}])
    store.save(task, 0)
    analyze_task(task.id, store, None)
    result = store.get(task.id)
    record = next(r for r in result.moves if r.status == 'auto_verified' and r.move is not None)
    assert record.move.piece_id == 'red_rook_0'
    assert record.action_end_s == 2.2
    assert result.evidence_end_s == 2.2


def test_continuous_king_adjustment_checks_turn_origin_not_intermediate_square(tmp_path, monkeypatch):
    from app import actual as module
    from app.analysis import analyze_task
    from app.schemas import Calibration
    from app.storage import TaskStore
    before = moved(opening_position(), 'red_king_4', 4, 8)
    middle = moved(before, 'red_king_4', 4, 7)
    final = moved(before, 'red_king_4', 4, 9)
    frames = [(0, before), (1, middle), (1.1, middle), (2, final), (2.1, final), (3, final)]
    monkeypatch.setattr(module, 'iter_frames', lambda *a, **k: iter((t, board_image(p)) for t, p in frames))
    monkeypatch.setattr(module, 'rectify_board', lambda image, *a, **k: image)
    monkeypatch.setattr('app.scanning.scan_video', lambda task, store: task)
    store = TaskStore(tmp_path)
    task = store.create('adjustment.mp4', VideoMetadata(width=901, height=1001, duration_s=3, codec='h264'))
    task.end_s, task.replay_mode, task.rules_enabled, task.initial_position = 3, 'actual', True, before
    task.calibration = Calibration(corners=[{'x':50,'y':50},{'x':850,'y':50},{'x':850,'y':950},{'x':50,'y':950}])
    store.save(task, 0)
    analyze_task(task.id, store, None)
    result = store.get(task.id)
    assert result.status == 'ready'
    assert len(result.moves) == 1
    record = result.moves[0]
    assert record.move.source == Square(file=4, rank=8)
    assert record.move.target == Square(file=4, rank=9)
    assert record.action_end_s == 2


def test_backfill_preserves_continuous_king_adjustment(monkeypatch):
    from app import actual as module
    before = moved(opening_position(), 'red_king_4', 4, 8)
    middle = moved(before, 'red_king_4', 4, 7)
    final = moved(before, 'red_king_4', 4, 9)
    following = moved(final, 'black_rook_0', 0, 1)
    frames = [(0, before), (1, middle), (1.1, middle), (2, final), (2.1, final),
              (3, following), (3.1, following)]
    monkeypatch.setattr(module, 'iter_frames', lambda *a: iter((t, board_image(p)) for t, p in frames))
    monkeypatch.setattr(module, 'rectify_board', lambda image, *a, **k: image)
    recovered = module.backfill_interval('clip.mp4', 0, 3.1, None, before, following)
    assert recovered
    assert board_signature(recovered[-1].position) == board_signature(following)
    assert any(board_signature(e.position) == board_signature(final) for e in recovered)


def test_review_noise_cannot_evict_independent_change_picture_and_time():
    from app.records import merge_review
    from app.schemas import MoveRecord
    before = opening_position()
    record = MoveRecord(index=1, video_start_s=0, video_end_s=0, status='needs_review', position_before=before)
    for i in range(3):
        merge_review(record, i / 10, [before], [f'noise-{i}.jpg'], '噪声', [i / 10])
    conflict = moved(moved(before, 'black_rook_0', 0, 1), 'black_rook_8', 8, 1)
    merge_review(record, 1.2, [conflict], ['independent.jpg'], '独立变化', [1.2])
    for i in range(8):
        merge_review(record, 2 + i / 10, [moved(before, 'red_rook_0', 0, 8 - i % 2)],
                     [f'adjust-{i}.jpg'], '重复调整', [2 + i / 10])
    assert 'independent.jpg' in record.evidence_ids
    assert record.evidence_times_s[record.evidence_ids.index('independent.jpg')] == 1.2
    assert any(board_signature(p) == board_signature(conflict) for p in record.observed_positions)


@pytest.mark.parametrize('outside_color', [(90, 50, 20), (180, 110, 70)])
@pytest.mark.parametrize('flip_x,flip_y', [(False, False), (True, False), (False, True), (True, True)])
def test_empty_corner_uses_visible_board_quadrant_not_off_board_background(outside_color, flip_x, flip_y):
    import cv2
    import numpy as np
    from app.analysis import read_board_cells
    image = np.full((500, 450, 3), (100, 175, 200), np.uint8)
    image[:25, 398:] = (90, 50, 20)
    image[:52, 426:] = outside_color
    cv2.line(image, (398, 25), (425, 25), (30, 30, 30), 2)
    cv2.line(image, (425, 25), (425, 52), (30, 30, 30), 2)
    hidden = image.copy()
    cv2.rectangle(hidden, (405, 25), (425, 45), (90, 130, 190), -1)
    if flip_x:
        image, hidden = cv2.flip(image, 1), cv2.flip(hidden, 1)
    if flip_y:
        image, hidden = cv2.flip(image, 0), cv2.flip(hidden, 0)
    square = (0 if flip_x else 8, 9 if flip_y else 0)
    cells = read_board_cells(image, strict=True)
    assert square in cells
    assert cells[square] is None
    assert square not in read_board_cells(hidden, strict=True)


def test_clear_board_reconfirms_pending_move_after_transient_detection_noise():
    before = opening_position()
    after = moved(before, 'red_rook_0', 0, 8)
    task = Task(id='d3' * 16, filename='noise.mp4', end_s=4,
                metadata=VideoMetadata(width=901, height=1001, duration_s=4, codec='h264'),
                initial_position=before, rules_enabled=True)
    replay = TurnReplay(task, before)
    images = [(t, board_image(p)) for t, p in [(0, before), (1, after), (1.1, after),
                                               (2, after), (2.3, after), (3, after), (4, after)]]
    readings = {}
    for t, image in images:
        position = before if t == 0 else after
        cells = {(f,r):None for f in range(9) for r in range(10)}
        cells.update({(p.square.file,p.square.rank):p.color for p in position.pieces})
        if t in (2, 2.3):
            cells.pop((8,4))
            cells[(4,4)] = 'black'
        readings[id(image)] = cells
    for event in actual_events(iter(images), before, lambda: replay.observed, turn_boundaries=True,
                               cell_reader=lambda image, reference: readings[id(image)]):
        if event.moving_colors:
            replay.movement(event.moving_colors, event.last_clear_s, event.departure_only)
        elif event.position is not None and not event.review_reason:
            replay.observe(event.position, event.time_s, [f'{event.time_s}.jpg'], event.action_start_s)
        else:
            replay.uncertain(event.time_s, [], [f'{event.time_s}.jpg'], event.review_reason)
    replay.end(4)
    assert len(task.moves) == 1
    assert task.moves[0].status == 'auto_verified'
    assert task.moves[0].move.target == Square(file=0,rank=8)
    assert task.moves[0].action_end_s == 1
