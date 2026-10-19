import cv2
import numpy as np
import pytest

from app.schemas import Calibration, Square, opening_position
from test_tracking import board_image


def test_border_selection_is_refined_to_intersections():
    from app.calibration import refine_calibration, rectify_board
    image = np.full((1060, 960, 3), (100, 175, 200), np.uint8)
    for x in range(80, 881, 100):
        cv2.line(image, (x, 80), (x, 980), (25, 25, 25), 3)
    for y in range(80, 981, 100):
        cv2.line(image, (80, y), (880, y), (25, 25, 25), 3)
    cal = Calibration(corners=[{'x': 40, 'y': 40}, {'x': 920, 'y': 40},
                               {'x': 920, 'y': 1020}, {'x': 40, 'y': 1020}])
    refined = refine_calibration(image, cal)
    assert np.allclose([(p.x, p.y) for p in refined.corners],
                       [(80, 80), (880, 80), (880, 980), (80, 980)], atol=5)
    assert rectify_board(image, refined, margin=50).shape == (1001, 901, 3)


def test_unique_initial_candidate_cannot_hide_missing_piece_or_wrong_color():
    from app.actual import verified_initial_candidates
    initial = opening_position()
    missing = initial.model_copy(deep=True)
    missing.pieces = [p for p in missing.pieces if p.id != 'red_rook_0']
    wrong = initial.model_copy(deep=True)
    next(p for p in wrong.pieces if p.id == 'red_rook_0').color = 'black'
    assert verified_initial_candidates([missing, wrong], [board_image(initial)]) == []
    assert verified_initial_candidates([initial], [board_image(initial)]) == [initial]


def test_fallback_does_not_send_stationary_hand_or_pickup_to_model():
    from app.actual import actual_events
    initial = opening_position()
    clear = board_image(initial)
    hidden = clear.copy()
    cv2.rectangle(hidden, (5, 890), (95, 995), (90, 130, 190), -1)
    # The first frame is occluded, which used to select permanent model fallback.
    frames = [(0, hidden), (.3, hidden), (1, clear), (1.2, clear),
              (2, hidden), (3, hidden), (4, clear), (4.2, clear)]
    events = list(actual_events(iter(frames), initial))
    assert not [e for e in events if e.position is None and not e.review_reason]
    assert not [e for e in events if e.position is not None and e.position != initial]


def test_long_pickup_pause_then_rook_capture_is_one_operation():
    from app.actual import actual_events, snapshot_record
    initial = opening_position()
    after = initial.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=0)
    after.pieces = [p for p in after.pieces if p.id != 'black_rook_0']
    lifted = initial.model_copy(deep=True)
    lifted.pieces = [p for p in lifted.pieces if p.id != 'red_rook_0']
    frames = [(0, board_image(initial)), (.1, board_image(lifted)),
              (3, board_image(lifted)), (6, board_image(lifted)),
              (7, board_image(after)), (7.2, board_image(after))]
    events = list(actual_events(iter(frames), initial))
    confirmed = [e for e in events if e.position is not None and not e.review_reason]
    assert len(confirmed) == 2
    assert not [e for e in events if e.position is None and not e.review_reason]
    record = snapshot_record(initial, confirmed[-1].position, 1, 0, 7)
    assert record.move.piece_id == 'red_rook_0'
    assert record.move.captured_piece_id == 'black_rook_0'


def test_initial_hidden_piece_cannot_be_confirmed_as_missing():
    from app.actual import verified_initial_candidates
    initial = opening_position()
    missing = initial.model_copy(deep=True)
    missing.pieces = [p for p in missing.pieces if p.id != 'red_rook_0']
    image = board_image(initial)
    cv2.rectangle(image, (5, 905), (95, 995), (90, 130, 190), -1)
    assert verified_initial_candidates([missing], [image]) == []


def test_model_cannot_remove_unrelated_hidden_piece():
    from app.actual import verified_observed_candidates
    initial = opening_position()
    after = initial.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    hidden = board_image(after)
    cv2.rectangle(hidden, (705, 905), (795, 995), (90, 130, 190), -1)
    missing = after.model_copy(deep=True)
    missing.pieces = [p for p in missing.pieces if p.id != 'red_horse_7']
    assert verified_observed_candidates(initial, [missing], board_image(initial), hidden) == []
    assert verified_observed_candidates(initial, [after], board_image(initial), hidden) == [after]


def test_occluded_start_preserves_two_rapid_clear_moves():
    from app.actual import actual_events
    initial = opening_position()
    first = initial.model_copy(deep=True)
    next(p for p in first.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    second = first.model_copy(deep=True)
    next(p for p in second.pieces if p.id == 'black_rook_0').square = Square(file=0, rank=1)
    hidden = board_image(initial)
    cv2.rectangle(hidden, (5, 905), (95, 995), (90, 130, 190), -1)
    frames = [(0, hidden), (.1, board_image(first)), (.18, board_image(first)),
              (.2, board_image(second)), (.28, board_image(second))]
    positions = [e.position for e in actual_events(iter(frames), initial) if e.position and not e.review_reason]
    from app.chess import board_signature
    assert list(map(board_signature, positions)) == list(map(board_signature, [initial, first, second]))


def test_short_visible_move_then_return_keeps_review_evidence():
    from app.actual import actual_events
    initial = opening_position()
    after = initial.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    events = list(actual_events(iter([(0, board_image(initial)), (.1, board_image(after)),
                                    (.14, board_image(after)), (.16, board_image(initial)),
                                    (.3, board_image(initial))]), initial))
    from app.chess import board_signature
    assert any(e.review_reason and e.position and board_signature(e.position) == board_signature(after) for e in events)


def test_pure_removal_keeps_full_candidate_for_manual_confirmation():
    from app.actual import actual_events
    initial = opening_position()
    after = initial.model_copy(deep=True)
    after.pieces = [p for p in after.pieces if p.id != 'red_rook_0']
    events = list(actual_events(iter([(0, board_image(initial)), (.1, board_image(after)),
                                    (1.2, board_image(after)), (2, board_image(after))]), initial))
    assert any(e.review_reason and e.position == after for e in events)
    assert not any(e.position == after and not e.review_reason for e in events)


@pytest.mark.parametrize('outcome', ['capture', 'return', 'removal'])
def test_pipeline_resolves_long_pickup_but_keeps_real_removal_review(tmp_path, monkeypatch, outcome):
    import app.actual as actual
    from app.schemas import VideoMetadata
    from app.storage import TaskStore
    initial = opening_position()
    lifted = initial.model_copy(deep=True)
    lifted.pieces = [p for p in lifted.pieces if p.id != 'red_rook_0']
    after = initial.model_copy(deep=True)
    if outcome == 'capture':
        next(p for p in after.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=0)
        after.pieces = [p for p in after.pieces if p.id != 'black_rook_0']
    elif outcome == 'removal':
        after = lifted
    frames = [(0, board_image(initial)), (.1, board_image(lifted)), (1.2, board_image(lifted)),
              (2, board_image(after)), (2.2, board_image(after))]
    monkeypatch.setattr(actual, 'iter_frames', lambda *args: iter(frames))
    store = TaskStore(tmp_path)
    task = store.create('pickup.mp4', VideoMetadata(width=901, height=1001, duration_s=3, codec='h264'))
    task.initial_position, task.replay_mode = initial, 'actual'
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50},
                                           {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    task = store.save(task, 0)
    actual.analyze_actual(task, store, None)
    saved = store.get(task.id)
    assert saved.run_stats.model_requests == 0
    if outcome == 'removal':
        assert saved.status == 'needs_review' and len(saved.moves) == 1
        assert saved.moves[0].observed_positions[0] == lifted
        corrected = actual.correct_actual(saved, 1, [], position=lifted)
        assert corrected.status == 'ready' and len(corrected.moves[0].position_after.pieces) == 31
    else:
        assert saved.status == 'ready'
        assert len(saved.moves) == (1 if outcome == 'capture' else 0)


def test_model_cannot_overwrite_unchanged_piece_identity_or_omit_visible_piece():
    from app.actual import verified_observed_candidates
    initial = opening_position()
    after = initial.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    wrong = after.model_copy(deep=True)
    next(p for p in wrong.pieces if p.id == 'red_pawn_6').square = Square(file=6, rank=5)
    assert verified_observed_candidates(initial, [wrong], board_image(initial), board_image(after)) == []
    assert verified_observed_candidates(initial, [after], board_image(initial), board_image(after)) == [after]


def test_partial_change_response_preserves_unrelated_pieces_and_capture_identity():
    import json
    from app.vision import parse_positions
    initial = opening_position()
    result = parse_positions(json.dumps({'changes': [
        {'file': 0, 'rank': 9, 'color': None, 'kind': None},
        {'file': 0, 'rank': 0, 'color': 'red', 'kind': 'rook'}]}), reference=initial, actual=True)
    assert len(result) == 1 and len(result[0].pieces) == 31
    assert next(p for p in result[0].pieces if p.id == 'red_rook_0').square == Square(file=0, rank=0)
    assert all(p.id != 'black_rook_0' for p in result[0].pieces)
    assert next(p for p in result[0].pieces if p.id == 'red_pawn_6').square == Square(file=6, rank=6)


def test_initial_piece_sheet_enumerates_visible_pieces_with_coordinates():
    from app.vision import piece_sheet
    image, cells = piece_sheet(board_image(opening_position()))
    assert image.shape[0] >= 400 and image.shape[1] >= 600
    assert len(cells) == 32
    assert cells[(0, 9)] == 'red' and cells[(0, 0)] == 'black'


def test_pipeline_reports_current_run_without_erasing_historical_usage(tmp_path, monkeypatch):
    import app.actual as actual
    from app.analysis import analyze_task
    from app.schemas import VideoMetadata
    from app.storage import TaskStore
    initial = opening_position()
    monkeypatch.setattr(actual, 'iter_frames', lambda *args: iter([(0, board_image(initial)), (.2, board_image(initial))]))
    store = TaskStore(tmp_path)
    task = store.create('timing.mp4', VideoMetadata(width=901, height=1001, duration_s=1, codec='h264'))
    task.initial_position, task.replay_mode = initial, 'actual'
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50}, {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    task.analysis_stats.model_requests = 19
    task.usage = [{'total_tokens': 99}]
    store.save(task, 0)
    analyze_task(task.id, store, None)
    saved = store.get(task.id)
    assert saved.run_stats.model_requests == 0 and saved.run_stats.scanned_frames == 2
    assert saved.run_stats.elapsed_seconds > 0
    assert saved.run_stats.decode_seconds >= 0 and saved.run_stats.opencv_seconds > 0
    assert saved.analysis_stats.model_requests == 19 and saved.usage[0]['total_tokens'] == 99


def test_existing_calibration_supported_by_complete_board_is_preserved(tmp_path, monkeypatch):
    import app.actual as actual
    from app.analysis import analyze_task
    from app.schemas import VideoMetadata
    from app.storage import TaskStore
    initial = opening_position()
    monkeypatch.setattr(actual, 'iter_frames', lambda *args: ((t, board_image(initial)) for t in [0, .2]))
    store = TaskStore(tmp_path)
    task = store.create('calibration.mp4', VideoMetadata(width=901, height=1001, duration_s=1, codec='h264'))
    task.initial_position, task.replay_mode = initial, 'actual'
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50}, {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    (store.task_dir(task.id) / 'source.mp4').touch()
    monkeypatch.setattr(__import__('app.calibration', fromlist=['']), 'refine_calibration',
                        lambda *args: (_ for _ in ()).throw(AssertionError('Must preserve independently verified calibration')))
    store.save(task, 0)
    analyze_task(task.id, store, None)
    saved = store.get(task.id)
    assert saved.status == 'ready' and saved.calibration == task.calibration
