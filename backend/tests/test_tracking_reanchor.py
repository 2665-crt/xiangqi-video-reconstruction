import pytest

from app.chess import board_signature
from app.schemas import Square, Task, VideoMetadata, opening_position
from app.turns import TurnReplay, revalidate_turns
from test_move_evidence import cells, moved


def readings(position):
    return {(p.square.file, p.square.rank): {
        'confidence': .9, 'class_probabilities': {f'{p.color}_{p.kind}': .99},
    } for p in position.pieces}


def recovery(reader):
    from app.tracking_recovery import VisualRecovery
    return VisualRecovery(reader)


def task_for(before):
    return Task(id='a' * 32, filename='reanchor.mp4', end_s=30,
                metadata=VideoMetadata(width=901, height=1001, duration_s=30, codec='h264'),
                initial_position=before, replay_mode='actual', rules_enabled=True)


def test_independent_type_recognition_repairs_stale_kind_without_inventing_a_move():
    actual = opening_position()
    stale = actual.model_copy(deep=True)
    next(p for p in stale.pieces if p.id == 'red_horse_1').kind = 'elephant'
    tracker = recovery(lambda image: readings(actual))
    tracker.request(1)
    assert tracker.observe(1, None, stale, cells(actual)) is None
    anchor = tracker.observe(2, None, stale, cells(actual))
    assert board_signature(anchor) == board_signature(actual)
    assert next(p for p in anchor.pieces if p.square == Square(file=1, rank=9)).kind == 'horse'


def test_one_frame_misclassification_and_unknown_cells_cannot_become_an_anchor():
    before = opening_position()
    final = moved(moved(before, 'red_rook_0', 0, 8), 'black_rook_0', 0, 1)
    tracker = recovery(lambda image: readings(image))
    tracker.request(1)
    assert tracker.observe(1, final, before, cells(final)) is None
    assert tracker.observe(2, before, before, cells(before)) is None
    hidden = cells(final)
    hidden.pop((8, 9))
    tracker.request(3)
    for t in (3, 4, 5, 15, 25):
        assert tracker.observe(t, final, before, hidden) is None


def test_compatible_frames_can_complete_coverage_but_conflicting_frames_cannot():
    before = opening_position()
    final = moved(moved(before, 'red_rook_0', 0, 8), 'black_rook_0', 0, 1)
    tracker = recovery(lambda image: readings(final))
    tracker.request(1)
    left, right = cells(final), cells(final)
    left.pop((8, 9))
    right.pop((7, 9))
    assert tracker.observe(1, None, before, left) is None
    assert tracker.observe(2, None, before, right) is not None
    tracker = recovery(lambda image: readings(final))
    tracker.request(1)
    assert tracker.observe(1, None, before, left) is None
    right[(0, 8)] = None
    assert tracker.observe(2, None, before, right) is None


def test_expensive_current_frame_recovery_is_rate_limited_and_backs_off():
    calls = []
    tracker = recovery(lambda image: calls.append(image) or {})
    before = opening_position()
    tracker.request(0)
    for n in range(301):
        tracker.observe(n / 10, n, before, cells(before))
    assert len(calls) <= 7


def test_recovery_can_confirm_clear_frames_after_earlier_failed_attempts():
    before = opening_position()
    final = moved(moved(before, 'red_rook_0', 0, 8), 'black_rook_0', 0, 1)
    tracker = recovery(lambda image: readings(image) if image is not None else {})
    tracker.request(0)
    for t in (0, 1, 2):
        assert tracker.observe(t, None, before, cells(before)) is None
    assert tracker.observe(12, final, before, cells(final)) is None
    assert tracker.observe(13, final, before, cells(final)) is not None


def test_backoff_still_allows_a_pair_of_complementary_partially_visible_frames():
    before = opening_position()
    final = moved(moved(before, 'red_rook_0', 0, 8), 'black_rook_0', 0, 1)
    tracker = recovery(lambda image: readings(image) if image is not None else {})
    tracker.request(0)
    for t in (0, 1, 2):
        assert tracker.observe(t, None, before, cells(before)) is None
    left, right = cells(final), cells(final)
    left.pop((4, 4))
    right.pop((5, 4))
    assert tracker.observe(12, final, before, left) is None
    assert tracker.observe(13, final, before, right) is not None


def test_reanchor_seals_gap_and_allows_two_following_automatic_moves():
    before = opening_position()
    anchor = moved(moved(before, 'red_rook_0', 0, 8), 'black_rook_0', 0, 1)
    task = task_for(before)
    replay = TurnReplay(task, before)
    replay.uncertain(4, [anchor], ['gap.jpg'], '无法唯一还原中间操作')
    replay.reanchor(anchor, 5, ['anchor-1.jpg', 'anchor-2.jpg'], [4, 5])
    # The historical side can be wrong; the next directly observed action establishes the new segment's side.
    black = moved(anchor, 'black_rook_8', 8, 1)
    red = moved(black, 'red_rook_8', 8, 8)
    replay.observe(black, 7, ['black.jpg'], 6)
    replay.observe(red, 9, ['red.jpg'], 8)
    replay.end(10)
    assert [r.status for r in task.moves] == ['needs_review', 'auto_verified', 'auto_verified']
    assert [r.move.piece_id for r in task.moves[1:]] == ['black_rook_8', 'red_rook_8']
    gap = task.moves[0]
    assert gap.move is None and gap.position_after is None
    assert gap.tracking_anchor_s == 5
    assert board_signature(gap.tracking_anchor) == board_signature(anchor)
    assert gap.video_end_s == 5
    assert task.confirmed_prefix_length == 0
    assert task.initial_position == before
    saved = Task.model_validate_json(task.model_dump_json())
    validated = revalidate_turns(saved)
    assert [r.status for r in validated.moves] == ['needs_review', 'auto_verified', 'auto_verified']


def test_reanchor_keeps_completed_move_and_does_not_turn_gap_into_a_move():
    before = opening_position()
    first = moved(before, 'red_rook_0', 0, 8)
    task = task_for(before)
    replay = TurnReplay(task, before)
    replay.observe(first, 2, ['first.jpg'], 1)
    replay.finish(2)
    original = task.moves[0].model_dump_json()
    anchor = moved(first, 'black_rook_0', 0, 1)
    replay.uncertain(3, [anchor], ['gap.jpg'], '失配')
    replay.reanchor(anchor, 5, ['clear.jpg'], [5])
    assert task.moves[0].model_dump_json() == original
    assert task.moves[1].status == 'needs_review'
    assert task.moves[1].move is None
    assert replay.pending is None and replay.unresolved is None and not replay.touched
    assert board_signature(replay.observed) == board_signature(anchor)


@pytest.mark.parametrize('rules_enabled', [False, True])
def test_pipeline_persists_anchor_and_auto_confirms_after_unresolved_history(tmp_path, monkeypatch, rules_enabled):
    from app import actual
    from app.analysis import VideoEvent, analyze_task
    from app.schemas import Calibration
    from app.storage import TaskStore
    from test_tracking import board_image
    before = opening_position()
    anchor = moved(moved(before, 'red_rook_0', 0, 8), 'black_rook_0', 0, 1)
    black = moved(anchor, 'black_rook_8', 8, 1)
    red = moved(black, 'red_rook_8', 8, 8)
    events = [VideoEvent(0, [board_image(before)], position=before),
              VideoEvent(2, [board_image(before), board_image(anchor)], True, position=anchor, review_reason='失配'),
              VideoEvent(5, [board_image(anchor), board_image(anchor)], position=anchor,
                         image_times_s=(4, 5), tracking_anchor=True),
              VideoEvent(7, [board_image(anchor), board_image(black)], position=black, action_start_s=6),
              VideoEvent(9, [board_image(black), board_image(red)], position=red, action_start_s=8)]
    monkeypatch.setattr(actual, 'actual_events', lambda *a, **k: iter(events))
    monkeypatch.setattr('app.scanning.scan_video', lambda task, store: task)
    store = TaskStore(tmp_path)
    task = store.create('resync.mp4', VideoMetadata(width=901, height=1001, duration_s=10, codec='h264'))
    task.initial_position, task.replay_mode, task.rules_enabled = before, 'actual', rules_enabled
    task.calibration = Calibration(corners=[{'x':50,'y':50},{'x':850,'y':50},{'x':850,'y':950},{'x':50,'y':950}])
    store.save(task, 0)
    analyze_task(task.id, store, None)
    result = store.get(task.id)
    assert result.status == 'needs_review', result.error
    assert [r.status for r in result.moves] == ['needs_review', 'auto_verified', 'auto_verified']
    assert result.moves[0].tracking_anchor_s == 5
    assert result.moves[0].position_after is None
    checked = actual.revalidate_actual(result)
    assert [r.status for r in checked.moves] == ['needs_review', 'auto_verified', 'auto_verified']
    assert [r.move.piece_id for r in checked.moves[1:]] == ['black_rook_8', 'red_rook_8']


def test_event_generator_discards_stale_sources_and_rejected_candidates_after_anchor():
    from app.actual import actual_events, snapshot_record
    from app.analysis import VideoEvent
    from test_tracking import board_image
    before = opening_position()
    anchor = moved(moved(before, 'red_rook_0', 0, 8), 'black_rook_0', 0, 1)
    black = moved(anchor, 'black_rook_8', 8, 1)
    red = moved(black, 'red_rook_8', 8, 8)
    frames = [(0, before), (1, anchor), (2, anchor), (3, anchor), (4, black), (4.1, black), (5, red), (5.1, red)]
    reference, actions = [before], []
    def restore(time_s, image, current):
        if time_s == 3:
            return VideoEvent(time_s, [image], position=anchor, tracking_anchor=True)
        return None
    for event in actual_events(iter((t, board_image(p)) for t, p in frames), before,
            confirmed_position=lambda: reference[0], turn_boundaries=True, reanchor_reader=restore):
        if event.tracking_anchor:
            reference[0] = event.position
        elif event.position is not None and not event.review_reason and not event.moving_colors:
            if event.board_check is None or event.board_check.status != 'conflict':
                action = snapshot_record(reference[0], event.position, 1, 0, event.time_s).move
                if action is not None:
                    actions.append(action.piece_id)
                reference[0] = event.position
        elif event.time_s > 3:
            assert not event.review_reason
    assert actions == ['black_rook_8', 'red_rook_8']


def test_manual_correction_before_gap_does_not_rewrite_later_visual_anchor():
    from app.schemas import Move
    from app.turns import correct_turn
    before = opening_position()
    first = moved(before, 'red_rook_0', 0, 8)
    anchor = moved(first, 'black_rook_0', 0, 1)
    task = task_for(before)
    replay = TurnReplay(task, before)
    replay.observe(first, 2, ['first.jpg'], 1)
    replay.finish(2)
    replay.uncertain(4, [anchor], ['gap.jpg'], '中间操作未知')
    replay.reanchor(anchor, 5, ['clear.jpg'], [5])
    reply = moved(anchor, 'red_rook_8', 8, 8)
    replay.observe(reply, 7, ['reply.jpg'], 6)
    replay.end(8)
    correction = Move(index=1, piece_id='red_horse_1', color='red',
                      source=Square(file=1, rank=9), target=Square(file=2, rank=7))
    fixed = correct_turn(task, 1, [correction], 2)
    assert fixed.moves[1].status == 'needs_review'
    assert fixed.moves[2].status == 'auto_verified'
    assert board_signature(fixed.moves[2].position_after) == board_signature(reply)


@pytest.mark.parametrize('ignore', [True, False])
def test_manually_resolving_gap_preserves_independent_continuation_anchor(ignore):
    from app.schemas import Move
    from app.turns import correct_turn
    before = opening_position()
    anchor = moved(moved(before, 'red_rook_0', 0, 8), 'black_rook_0', 0, 1)
    task = task_for(before)
    replay = TurnReplay(task, before)
    replay.uncertain(4, [anchor], ['gap.jpg'], '中间操作未知')
    replay.reanchor(anchor, 5, ['clear.jpg'], [5])
    reply = moved(anchor, 'red_rook_8', 8, 8)
    replay.observe(reply, 7, ['reply.jpg'], 6)
    replay.end(8)
    actions = [] if ignore else [Move(index=1, piece_id='red_rook_0', color='red',
                                     source=Square(file=0, rank=9), target=Square(file=0, rank=8))]
    fixed = correct_turn(task, 1, actions, 5)
    assert fixed.moves[0].tracking_anchor is not None
    assert board_signature(fixed.moves[1].position_after) == board_signature(reply)
