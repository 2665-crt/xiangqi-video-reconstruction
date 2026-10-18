import cv2
import pytest

import app.actual as actual
from app.analysis import VideoEvent, analyze_task
from app.chess import board_signature
from app.schemas import Calibration, Square, Task, VideoMetadata, opening_position
from app.storage import TaskStore
from app.turns import TurnReplay, correct_turn
from test_tracking import board_image


def moved(position, piece_id, file, rank):
    after = position.model_copy(deep=True)
    next(p for p in after.pieces if p.id == piece_id).square = Square(file=file, rank=rank)
    return after


def cells(position):
    occupied = {(p.square.file, p.square.rank): p.color for p in position.pieces}
    return {(f, r): occupied.get((f, r)) for f in range(9) for r in range(10)}


def test_transition_rejects_preexisting_whole_board_mismatch():
    before = opening_position()
    after = moved(before, 'red_rook_0', 0, 8)
    observed_before = cells(moved(before, 'black_rook_8', 8, 1))
    result = actual.check_transition(before, after, observed_before, cells(after))
    assert result.status == 'conflict'
    assert Square(file=8, rank=0) in result.before_mismatches


def test_transition_rejects_extra_change_outside_moving_piece():
    before = opening_position()
    after = moved(before, 'red_rook_0', 0, 8)
    unexpected = moved(after, 'black_rook_8', 8, 1)
    result = actual.check_transition(before, after, cells(before), cells(unexpected))
    assert result.status == 'conflict'
    assert len(result.after_mismatches) == 2


def test_unknown_cells_remain_unknown_without_hiding_visible_conflict():
    before = opening_position()
    after = moved(before, 'red_rook_0', 0, 8)
    observed = cells(after)
    del observed[(8, 0)]
    result = actual.check_transition(before, after, cells(before), observed)
    assert result.status == 'partial'
    assert result.unknown_after == [Square(file=8, rank=0)]
    observed[(8, 1)] = 'black'
    assert actual.check_transition(before, after, cells(before), observed).status == 'conflict'


def test_hidden_target_cannot_pass_whole_board_audit():
    before = opening_position()
    after = moved(before, 'red_rook_0', 0, 8)
    observed = cells(after)
    del observed[(0, 8)]
    assert actual.check_transition(before, after, cells(before), observed).status == 'conflict'


def test_two_legal_moves_in_one_snapshot_require_video_backfill():
    before = opening_position()
    after = moved(moved(before, 'red_rook_0', 0, 8), 'black_rook_0', 0, 1)
    result = actual.check_transition(before, after, cells(before), cells(after))
    assert result.status == 'conflict'
    assert '中间' in result.reason


def test_selected_events_finish_tail_slide_but_exclude_next_piece():
    before = opening_position()
    first = moved(before, 'red_rook_0', 0, 8)
    final = moved(first, 'red_rook_0', 0, 7)
    reply = moved(final, 'black_rook_0', 0, 1)
    events = [VideoEvent(0, [board_image(before)], position=before),
              VideoEvent(3, [board_image(before), board_image(first)], position=first, action_start_s=1),
              VideoEvent(4, [board_image(first), board_image(final)], position=final, action_start_s=3.5),
              VideoEvent(6, [board_image(final), board_image(reply)], position=reply, action_start_s=5)]
    selected = list(actual.selected_events(iter(events), before, 2))
    assert [e.position for e in selected if e.position] == [before, first, final]


def test_tail_does_not_include_move_that_started_after_selection():
    before = opening_position()
    after = moved(before, 'red_rook_0', 0, 8)
    events = [VideoEvent(0, [board_image(before)], position=before),
              VideoEvent(3, [board_image(before), board_image(after)], position=after, action_start_s=2.5)]
    assert len(list(actual.selected_events(iter(events), before, 2))) == 1


def test_tail_ignores_unchanged_board_with_different_turn_metadata():
    before = opening_position()
    after = moved(before, 'red_rook_0', 0, 8).model_copy(update={'side_to_move': 'black'})
    same = after.model_copy(update={'side_to_move': 'red'})
    events = [VideoEvent(1, [board_image(after)], position=after, action_start_s=.5),
              VideoEvent(3, [board_image(same)], position=same)]
    selected = list(actual.selected_events(iter(events), before, 2))
    assert len(selected) == 1
    assert selected[0].position == after


def test_selected_capture_does_not_end_at_victim_lift():
    before = opening_position()
    next(p for p in before.pieces if p.id == 'black_pawn_0').square = Square(file=0, rank=8)
    after = moved(before, 'red_rook_0', 0, 8)
    after.pieces = [p for p in after.pieces if p.id != 'black_pawn_0']
    events = [VideoEvent(0, [board_image(before)], position=before),
              VideoEvent(2.2, [board_image(before)], moving_colors=('black',), departure_only=True),
              VideoEvent(3, [board_image(before), board_image(after)], position=after, action_start_s=1)]
    selected = list(actual.selected_events(iter(events), before, 2))
    assert selected[-1].position == after


def test_manual_tail_correction_accepts_time_beyond_selection():
    before = opening_position()
    task = Task(id='a' * 32, filename='clip.mp4', end_s=2,
                metadata=VideoMetadata(width=901, height=1001, duration_s=10, codec='h264'),
                initial_position=before, rules_enabled=True)
    replay = TurnReplay(task, before)
    replay.observe(moved(before, 'red_rook_0', 0, 8), 3, ['before.jpg', 'after.jpg'], 1)
    replay.end(5)
    corrected = correct_turn(task, 1, [task.moves[0].move], 3)
    assert corrected.moves[0].action_end_s == 3


def test_actual_pipeline_extends_decoder_and_keeps_only_selected_tail(tmp_path, monkeypatch):
    before = opening_position()
    first = moved(before, 'red_rook_0', 0, 8)
    final = moved(first, 'red_rook_0', 0, 7)
    reply = moved(final, 'black_rook_0', 0, 1)
    frames = [(0, before), (1, before.model_copy(deep=True)), (2.5, first), (2.6, first),
              (3.5, final), (3.6, final), (4.5, reply), (4.6, reply)]
    next(p for p in frames[1][1].pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    image_frames = [(t, board_image(p)) for t, p in frames]
    cv2.rectangle(image_frames[1][1], (0, 750), (100, 1000), (20, 20, 20), -1)
    requested = []
    def decode(path, start, end):
        requested.append(end)
        return iter((t, image) for t, image in image_frames if start <= t <= end)
    monkeypatch.setattr(actual, 'iter_frames', decode)
    monkeypatch.setattr(actual, 'rectify_board', lambda frame, *args, **kwargs: frame)
    monkeypatch.setattr('app.scanning.scan_video', lambda task, store: task)
    store = TaskStore(tmp_path)
    task = store.create('tail.mp4', VideoMetadata(width=901, height=1001, duration_s=8, codec='h264'))
    task.end_s, task.replay_mode, task.rules_enabled = 2, 'actual', True
    task.calibration = Calibration(corners=[{'x':50,'y':50},{'x':850,'y':50},{'x':850,'y':950},{'x':50,'y':950}])
    task.calibration_confirmed, task.initial_position = True, before
    store.save(task, 0)
    analyze_task(task.id, store, None)
    result = store.get(task.id)
    assert result.status != 'failed', result.error
    assert max(requested) == 8
    assert len(result.moves) == 1
    assert result.moves[0].move.target == Square(file=0, rank=7)
    assert result.moves[0].action_end_s == 3.5
    assert result.moves[0].board_check is not None


def test_backfill_recovers_real_intermediate_landings_without_rule_inference(monkeypatch):
    before = opening_position()
    red = moved(before, 'red_rook_0', 0, 8)
    final = moved(red, 'black_rook_0', 0, 1)
    frames = [(0, before), (1, red), (1.1, red), (2, final), (2.1, final)]
    monkeypatch.setattr(actual, 'iter_frames', lambda *args: iter((t, board_image(p)) for t, p in frames))
    monkeypatch.setattr(actual, 'rectify_board', lambda image, *args, **kwargs: image)
    result = actual.backfill_interval('clip.mp4', 0, 2.1, None, before, final)
    assert [board_signature(e.position) for e in result] == [board_signature(red), board_signature(final)]
    assert result[0].action_start_s == 1
    assert result[1].action_start_s == 2
    assert all(e.backfilled for e in result)


def test_backfill_without_intermediate_frame_does_not_invent_moves(monkeypatch):
    before = opening_position()
    final = moved(moved(before, 'red_rook_0', 0, 8), 'black_rook_0', 0, 1)
    frames = [(0, before), (2, final), (2.1, final)]
    monkeypatch.setattr(actual, 'iter_frames', lambda *args: iter((t, board_image(p)) for t, p in frames))
    monkeypatch.setattr(actual, 'rectify_board', lambda image, *args, **kwargs: image)
    assert actual.backfill_interval('clip.mp4', 0, 2.1, None, before, final) == []


def test_eof_unfinished_move_stays_review_with_no_invented_target():
    before = opening_position()
    hidden = board_image(before)
    cv2.rectangle(hidden, (0, 750), (100, 1000), (20, 20, 20), -1)
    events = list(actual.selected_events(actual.actual_events(iter([(0, board_image(before)), (1, hidden), (4, hidden)]),
                              before, turn_boundaries=True), before, 2))
    assert events[-1].review_reason
    assert events[-1].position is None


def test_same_board_long_thinking_gap_is_not_missing_move():
    before = opening_position()
    events = list(actual.actual_events(iter([(0, board_image(before)), (24, board_image(before))]),
                                      before, turn_boundaries=True))
    assert len(events) == 1


def test_before_source_occlusion_can_use_earlier_direct_frame_evidence(tmp_path, monkeypatch):
    before = opening_position()
    red = moved(before, 'red_rook_0', 0, 8)
    black = moved(red, 'black_rook_0', 0, 1)
    blocked = board_image(red)
    cv2.rectangle(blocked, (0, 0), (100, 200), (20, 20, 20), -1)
    monkeypatch.setattr(actual, 'actual_events', lambda *a, **k: iter([
        VideoEvent(0, [board_image(before)], position=before),
        VideoEvent(1, [board_image(before), blocked], position=red, action_start_s=.5),
        VideoEvent(2, [blocked, board_image(black)], position=black, action_start_s=1.5)]))
    monkeypatch.setattr('app.scanning.scan_video', lambda task, store: task)
    store = TaskStore(tmp_path)
    task = store.create('occluded-before.mp4', VideoMetadata(width=901, height=1001, duration_s=3, codec='h264'))
    task.replay_mode, task.rules_enabled, task.initial_position = 'actual', True, before
    task.calibration = Calibration(corners=[{'x':50,'y':50},{'x':850,'y':50},{'x':850,'y':950},{'x':50,'y':950}])
    store.save(task, 0)
    analyze_task(task.id, store, None)
    result = store.get(task.id)
    assert [r.status for r in result.moves] == ['auto_verified', 'auto_verified']
    assert len(result.moves[1].evidence_ids) >= 3


def test_stale_whole_board_clear_time_cannot_precede_actual_landing():
    before = opening_position()
    task = Task(id='b' * 32, filename='clip.mp4', end_s=5,
                metadata=VideoMetadata(width=901, height=1001, duration_s=5, codec='h264'),
                initial_position=before, rules_enabled=True)
    replay = TurnReplay(task, before)
    red = moved(before, 'red_rook_0', 0, 8)
    replay.observe(red, 2.4, [], 1.2)
    replay.movement(('black',), .8)
    replay.observe(moved(red, 'black_rook_0', 0, 1), 3.5, [], 3)
    assert task.moves[0].video_end_s >= task.moves[0].action_end_s


def test_next_piece_lift_outside_selection_does_not_pollute_last_selected_turn():
    before = opening_position()
    red = moved(before, 'red_rook_0', 0, 8)
    hidden = board_image(red)
    cv2.rectangle(hidden, (0, 0), (100, 200), (20, 20, 20), -1)
    frames = [(0, board_image(before)), (.5, board_image(red)), (.6, board_image(red)),
              (3, hidden), (4, hidden)]
    result = list(actual.selected_events(actual.actual_events(iter(frames), before, turn_boundaries=True), before, 2))
    assert not any(e.review_reason for e in result)


def test_eof_merged_evidence_keeps_timestamp_alignment():
    before = opening_position()
    task = Task(id='c' * 32, filename='clip.mp4', end_s=3,
                metadata=VideoMetadata(width=901, height=1001, duration_s=3, codec='h264'),
                initial_position=before, rules_enabled=True)
    replay = TurnReplay(task, before)
    replay.observe(moved(before, 'red_rook_0', 0, 8), 1, ['before', 'after'], .5)
    replay.attach_evidence(None, ['before', 'after'], [0, 1])
    replay.uncertain(3, [], ['after', 'hidden'], '落点遮挡')
    replay.attach_evidence(None, ['after', 'hidden'], [1, 3])
    replay.end(3)
    assert task.moves[0].evidence_ids == ['before', 'after', 'hidden']
    assert task.moves[0].evidence_times_s == [0, 1, 3]


def test_backfill_uses_brief_landing_not_following_action_picture(monkeypatch):
    before = opening_position()
    red = moved(before, 'red_rook_0', 0, 8)
    final = moved(red, 'black_rook_0', 0, 1)
    frames = [(0, before), (1, red), (2, final), (2.1, final)]
    monkeypatch.setattr(actual, 'iter_frames', lambda *args: iter((t, board_image(p)) for t, p in frames))
    monkeypatch.setattr(actual, 'rectify_board', lambda image, *args, **kwargs: image)
    result = actual.backfill_interval('clip.mp4', 0, 2.1, None, before, final)
    assert len(result) == 2
    assert result[0].image_times_s[-1] == 1


def test_backfilled_tail_preserves_already_selected_piece_identity():
    before = opening_position()
    first = moved(before, 'red_rook_0', 0, 8)
    final = moved(first, 'red_rook_0', 0, 7)
    event = VideoEvent(3, [board_image(first), board_image(final)], position=final, action_start_s=2.5, backfilled=True)
    result = list(actual.selected_events(iter([event]), first, 2, last_piece='red_rook_0'))
    assert result == [event]


def test_next_piece_visible_lift_outside_selection_is_excluded():
    before = opening_position()
    red = moved(before, 'red_rook_0', 0, 8)
    lifted = red.model_copy(deep=True)
    lifted.pieces = [p for p in lifted.pieces if p.id != 'black_rook_0']
    frames = [(0, board_image(before)), (.5, board_image(red)), (.6, board_image(red)),
              (3, board_image(lifted)), (4.1, board_image(lifted))]
    result = list(actual.selected_events(actual.actual_events(iter(frames), before, turn_boundaries=True), before, 2))
    assert not any(e.review_reason for e in result)


def test_changing_range_discards_old_tail_evidence_boundary(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    store = TaskStore(tmp_path)
    task = store.create('range.mp4', VideoMetadata(width=901, height=1001, duration_s=10, codec='h264'))
    task.end_s, task.evidence_end_s = 2, 6
    task = store.save(task, 0)
    with TestClient(create_app(tmp_path)) as client:
        response = client.patch(f'/api/tasks/{task.id}/setup', json={'revision':task.revision, 'start_s':0, 'end_s':3})
    assert response.status_code == 200
    assert response.json()['evidence_end_s'] is None
