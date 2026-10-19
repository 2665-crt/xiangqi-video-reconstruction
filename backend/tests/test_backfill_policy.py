import cv2
import app.actual as actual
from app.analysis import VideoEvent, analyze_task
from app.schemas import AnalysisStats, Calibration, Square, VideoMetadata, opening_position
from app.storage import TaskStore
from test_move_evidence import moved
from test_tracking import board_image


def test_old_runs_do_not_report_unmeasured_backfills_as_zero():
    old = AnalysisStats.model_validate({'scanned_frames': 12107, 'elapsed_seconds': 755.4})
    assert old.backfill_intervals is None


def run_events(tmp_path, monkeypatch, events, recover=None, end=10):
    calls = []

    def backfill(source, start, stop, calibration, before, expected, on_frame=None, motion_active=False, motion_reference_s=None):
        calls.append((start, stop))
        if on_frame is not None:
            on_frame(start)
        return recover or []

    monkeypatch.setattr(actual, 'actual_events', lambda *a, **k: iter(events))
    monkeypatch.setattr(actual, 'backfill_interval', backfill)
    monkeypatch.setattr('app.scanning.scan_video', lambda task, store: task)
    store = TaskStore(tmp_path)
    task = store.create('policy.mp4', VideoMetadata(width=901, height=1001, duration_s=end, codec='h264'))
    task.replay_mode, task.rules_enabled = 'actual', True
    task.calibration_confirmed = True
    task.initial_position, task.initial_turn_known = opening_position(), True
    task.calibration = Calibration(corners=[{'x':50,'y':50},{'x':850,'y':50},{'x':850,'y':950},{'x':50,'y':950}])
    store.save(task, 0)
    (store.task_dir(task.id) / 'source.mp4').touch()
    analyze_task(task.id, store, None)
    result = store.get(task.id)
    assert result.status != 'failed', result.error
    return result, calls


def test_unrelated_board_noise_waits_forward_and_keeps_correct_moves(tmp_path, monkeypatch):
    before = opening_position()
    red = moved(before, 'red_rook_0', 0, 8)
    black = moved(red, 'black_rook_0', 0, 1)
    noisy = moved(red, 'black_rook_8', 8, 1)
    events = [VideoEvent(0, [board_image(before)], position=before),
              VideoEvent(1, [board_image(before), board_image(noisy)], position=red, action_start_s=.5),
              VideoEvent(2, [board_image(before), board_image(red)], position=red, action_start_s=.5),
              VideoEvent(3, [board_image(red), board_image(black)], position=black, action_start_s=2.5)]
    result, calls = run_events(tmp_path, monkeypatch, events)
    assert calls == []
    assert result.status == 'ready'
    assert [r.move.piece_id for r in result.moves] == ['red_rook_0', 'black_rook_0']
    assert result.confirmed_prefix_length == 2


def test_missing_endpoint_waits_for_visible_evidence_without_rewind(tmp_path, monkeypatch):
    before = opening_position()
    after = moved(before, 'red_rook_0', 0, 8)
    hidden = board_image(after)
    cv2.rectangle(hidden, (0, 800), (100, 900), (20, 20, 20), -1)
    result, calls = run_events(tmp_path, monkeypatch, [
        VideoEvent(0, [board_image(before)], position=before),
        VideoEvent(1, [board_image(before), hidden], position=after, action_start_s=.5),
        VideoEvent(2, [board_image(before), board_image(after)], position=after, action_start_s=.5)])
    assert calls == []
    assert result.status == 'ready'
    assert len(result.moves) == 1
    assert result.moves[0].move.target == Square(file=0, rank=8)


def test_confirmed_board_is_not_replayed_for_noisy_unchanged_observation(tmp_path, monkeypatch):
    before = opening_position()
    after = moved(before, 'red_rook_0', 0, 8)
    noisy = moved(after, 'black_rook_8', 8, 1)
    result, calls = run_events(tmp_path, monkeypatch, [
        VideoEvent(0, [board_image(before)], position=before),
        VideoEvent(1, [board_image(before), board_image(after)], position=after, action_start_s=.5),
        VideoEvent(2, [board_image(noisy)], position=after),
        VideoEvent(3, [board_image(after)], position=after)])
    assert calls == []
    assert result.status == 'ready'
    assert len(result.moves) == 1
    assert result.moves[0].action_end_s == 1


def test_different_candidates_share_one_unresolved_interval_budget(tmp_path, monkeypatch):
    before = opening_position()
    first = moved(moved(before, 'red_rook_0', 0, 8), 'black_rook_0', 0, 1)
    different = moved(first, 'red_rook_8', 8, 8)
    result, calls = run_events(tmp_path, monkeypatch, [
        VideoEvent(0, [board_image(before)], position=before),
        VideoEvent(2, [board_image(before), board_image(first)], position=first),
        VideoEvent(3, [board_image(before), board_image(different)], position=different)])
    assert calls == [(0, 2)]
    assert result.status == 'needs_review'
    assert result.confirmed_prefix_length == 0
    assert all(r.move is None for r in result.moves)
    assert result.run_stats.backfill_intervals == 1
    assert result.run_stats.backfill_frames == 1


def test_long_unresolved_gap_is_preserved_without_scanning_from_old_boundary(tmp_path, monkeypatch):
    before = opening_position()
    final = moved(moved(before, 'red_rook_0', 0, 8), 'black_rook_0', 0, 1)
    result, calls = run_events(tmp_path, monkeypatch, [
        VideoEvent(0, [board_image(before)], position=before),
        VideoEvent(30, [board_image(before), board_image(final)], position=final)], end=35)
    assert calls == []
    assert result.status == 'needs_review'
    assert result.confirmed_prefix_length == 0
    assert result.moves[0].board_check.status == 'conflict'


def test_real_missing_intermediate_moves_can_still_be_recovered(tmp_path, monkeypatch):
    before = opening_position()
    red = moved(before, 'red_rook_0', 0, 8)
    black = moved(red, 'black_rook_0', 0, 1)
    recovered = [VideoEvent(1, [board_image(before), board_image(red)], position=red, action_start_s=.5, backfilled=True),
                 VideoEvent(2, [board_image(red), board_image(black)], position=black, action_start_s=1.5, backfilled=True)]
    result, calls = run_events(tmp_path, monkeypatch, [
        VideoEvent(0, [board_image(before)], position=before),
        VideoEvent(2, [board_image(before), board_image(black)], position=black)], recover=recovered)
    assert calls == [(0, 2)]
    assert result.status == 'ready'
    assert [r.move.piece_id for r in result.moves] == ['red_rook_0', 'black_rook_0']
    assert result.run_stats.backfill_intervals == 1


def test_new_confirmed_boundary_allows_a_separate_local_recovery(tmp_path, monkeypatch):
    before = opening_position()
    red = moved(before, 'red_rook_0', 0, 8)
    black = moved(red, 'black_rook_0', 0, 1)
    following = moved(black, 'red_rook_8', 8, 8)
    result, calls = run_events(tmp_path, monkeypatch, [
        VideoEvent(0, [board_image(before)], position=before),
        VideoEvent(2, [board_image(before), board_image(black)], position=black),
        VideoEvent(3, [board_image(before), board_image(red)], position=red, action_start_s=2.5),
        VideoEvent(5, [board_image(red), board_image(following)], position=following)])
    assert calls == [(0, 2), (3, 5)]
    assert result.status == 'needs_review'
    assert result.run_stats.backfill_intervals == 2


def test_multi_piece_candidate_with_hidden_changes_does_not_trigger_backfill(tmp_path, monkeypatch):
    before = opening_position()
    final = moved(moved(before, 'red_rook_0', 0, 8), 'black_rook_0', 0, 1)
    hidden = board_image(final)
    cv2.rectangle(hidden, (0, 800), (100, 900), (20, 20, 20), -1)
    result, calls = run_events(tmp_path, monkeypatch, [
        VideoEvent(0, [board_image(before)], position=before),
        VideoEvent(2, [board_image(before), hidden], position=final)])
    assert calls == []
    assert result.status == 'needs_review'
    assert result.confirmed_prefix_length == 0
