import cv2
import pytest

from app.schemas import Calibration, Position, Square, VideoMetadata, opening_position
from app.storage import TaskStore
from test_tracking import board_image


def relocated(position, changes):
    result = position.model_copy(deep=True)
    for piece_id, file, rank in changes:
        next(p for p in result.pieces if p.id == piece_id).square = Square(file=file, rank=rank)
    return result


def test_local_only_initial_classification_counts_zero_cloud_requests(tmp_path, monkeypatch):
    import app.actual as actual
    from app.analysis import analyze_task
    monkeypatch.setattr(actual, 'find_initial_frame', lambda *args: (0, board_image(opening_position())))
    monkeypatch.setattr(actual, 'initial_frame_images', lambda *args: [args[-1]])
    class Local:
        usage = []
        last_request_count = last_image_count = 0
        last_response = {'source': 'local_piece_classifier', 'readings': []}
        def recognize_position(self, *args, **kwargs): return []
    monkeypatch.setattr('app.piece_classifier.local_piece_classifier', lambda: Local())
    store = TaskStore(tmp_path)
    task = store.create('local.mp4', VideoMetadata(width=901, height=1001, duration_s=3, codec='h264'))
    task.replay_mode = 'actual'
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50}, {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    store.save(task, 0)
    analyze_task(task.id, store, None)
    result = store.get(task.id)
    assert result.status == 'needs_review'
    assert result.analysis_stats.model_requests == result.run_stats.model_requests == 0
    assert result.analysis_stats.images_sent == result.run_stats.images_sent == 0
    assert (store.task_dir(task.id) / 'vision-last-response.json').is_file()


def test_actual_initial_request_failure_saves_diagnostic(tmp_path, monkeypatch):
    import json
    import app.actual as actual
    from app.analysis import analyze_task
    monkeypatch.setattr(actual, 'find_initial_frame', lambda *args: (0, board_image(opening_position())))
    monkeypatch.setattr(actual, 'initial_frame_images', lambda *args: [args[-1]])
    class Vision:
        usage = []
        last_response = {'request_error': 'ReadTimeout', 'elapsed_s': 90.5}
        def recognize_position(self, *args, **kwargs):
            raise ValueError('等待模型响应超时')
        def close(self): pass
    store = TaskStore(tmp_path)
    task = store.create('timeout.mp4', VideoMetadata(width=901, height=1001, duration_s=3, codec='h264'))
    task.replay_mode = 'actual'
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50}, {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    store.save(task, 0)
    analyze_task(task.id, store, Vision())
    result = store.get(task.id)
    assert result.status == 'failed'
    assert result.error == '等待模型响应超时'
    assert result.analysis_stats.model_requests == 1
    assert result.run_stats.model_seconds > 0
    assert json.loads((store.task_dir(task.id) / 'vision-last-response.json').read_text(encoding='utf-8'))['request_error'] == 'ReadTimeout'


def test_local_worker_returns_editable_draft_instead_of_discarding_initial_pieces(tmp_path, monkeypatch):
    import app.actual as actual
    from app.analysis import analyze_task
    draft = opening_position()
    review = {'square': {'file': 0, 'rank': 0}, 'color': 'black', 'prediction': 'rook', 'score': .7, 'reason': '棋种置信度不足'}
    monkeypatch.setattr(actual, 'find_initial_frame', lambda *args: (0, board_image(draft)))
    monkeypatch.setattr(actual, 'initial_frame_images', lambda *args: [args[-1]])
    class Local:
        usage = []
        last_request_count = last_image_count = 0
        last_response = {'draft_position': draft.model_dump(), 'piece_reviews': [review], 'error': '棋种置信度不足'}
        def recognize_position(self, *args, **kwargs): return []
    monkeypatch.setattr('app.piece_classifier.local_piece_classifier', lambda: Local())
    store = TaskStore(tmp_path)
    task = store.create('draft.mp4', VideoMetadata(width=901, height=1001, duration_s=3, codec='h264'))
    task.replay_mode = 'actual'
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50}, {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    store.save(task, 0)
    analyze_task(task.id, store, None)
    result = store.get(task.id)
    assert result.initial_draft == draft
    assert result.initial_piece_reviews[0].square == Square(file=0, rank=0)
    assert result.initial_position is None and result.initial_candidates == []
    assert result.status == 'needs_review' and result.moves == []
    assert result.run_stats.model_requests == result.run_stats.images_sent == 0


def test_cannon_diagonal_and_same_side_changes_keep_identity():
    from app.actual import reconcile_position, snapshot_record
    initial = opening_position()
    after = relocated(initial, [('red_cannon_1', 3, 5)])
    observed = after.model_copy(deep=True)
    for p in observed.pieces:
        p.id = 'observed_' + p.id
    restored = reconcile_position(initial, observed, [initial])
    record = snapshot_record(initial, restored, 1, 40, 41)
    assert record.status == 'auto_verified'
    assert record.move.piece_id == 'red_cannon_1'
    assert (record.move.target.file, record.move.target.rank) == (3, 5)
    second = reconcile_position(restored, relocated(restored, [('red_cannon_7', 7, 4)]), [initial])
    assert next(p for p in second.pieces if p.id == 'red_cannon_7').square.rank == 4


def test_capture_then_undo_restores_captured_piece_and_identity():
    from app.actual import reconcile_position, snapshot_record
    initial = opening_position()
    after = relocated(initial, [('red_cannon_1', 0, 3)])
    after.pieces = [p for p in after.pieces if p.id != 'black_pawn_0']
    captured = reconcile_position(initial, after, [initial])
    assert len(captured.pieces) == 31
    record = snapshot_record(initial, captured, 1, 0, 1)
    assert record.move.captured_piece_id == 'black_pawn_0'
    undo = reconcile_position(captured, initial, [initial, captured])
    assert {p.id for p in undo.pieces} == {p.id for p in initial.pieces}
    assert snapshot_record(captured, undo, 2, 1, 2).position_after.pieces == initial.pieces


def test_batch_relocation_has_one_snapshot_without_invented_order():
    from app.actual import reconcile_position, snapshot_record
    initial = opening_position()
    after = relocated(initial, [('red_cannon_1', 3, 5), ('black_horse_1', 2, 4)])
    restored = reconcile_position(initial, after, [initial])
    record = snapshot_record(initial, restored, 1, 30, 31)
    assert record.move is None
    assert record.status == 'auto_verified'
    assert record.change_type == 'setup'
    assert len(record.position_after.pieces) == 32


def test_same_kind_identity_ambiguity_still_reproduces_unique_board():
    from app.actual import reconcile_position
    initial = opening_position()
    observed = relocated(initial, [('red_cannon_1', 2, 5), ('red_cannon_7', 6, 5)])
    for p in observed.pieces:
        p.id = 'new_' + p.id
    after = reconcile_position(initial, observed, [initial])
    assert {(p.kind, p.square.file, p.square.rank) for p in after.pieces if p.color == 'red' and p.kind == 'cannon'} == {
        ('cannon', 2, 5), ('cannon', 6, 5)}


def test_clear_setup_can_remove_king_and_place_it_outside_palace():
    p = opening_position().model_dump()
    p['pieces'] = [piece for piece in p['pieces'] if piece['id'] != 'black_king_4']
    next(piece for piece in p['pieces'] if piece['id'] == 'red_king_4')['square'] = {'file': 0, 'rank': 4}
    observed = Position.model_validate(p)
    assert len(observed.pieces) == 31


def test_same_color_piece_swap_is_not_erased_by_reference_memory():
    import json
    from app.vision import parse_positions
    initial = opening_position()
    observed = initial.model_copy(deep=True)
    cannon = next(p for p in observed.pieces if p.id == 'red_cannon_1')
    rook = next(p for p in observed.pieces if p.id == 'red_rook_0')
    cannon.square, rook.square = rook.square, cannon.square
    payload = {'candidates': [{'pieces': [{'color': p.color, 'kind': p.kind, 'file': p.square.file, 'rank': p.square.rank}
                                         for p in observed.pieces], 'side_to_move': None}]}
    result = parse_positions(json.dumps(payload), reference=initial, actual=True)
    assert next(p for p in result[0].pieces if p.id == 'red_cannon_1').square == Square(file=0, rank=9)


def test_start_at_zero_can_initialize_from_first_clear_frame(monkeypatch):
    import app.actual as actual
    monkeypatch.setattr(actual, 'iter_frames', lambda *args: iter([(0, board_image(opening_position()))]))
    calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50}, {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    time, image = actual.find_initial_frame('zero.mp4', 0, calibration)
    assert time == 0 and image.shape == (1001, 901, 3)


def test_piece_back_held_over_board_cannot_be_a_confirmed_landing():
    from app.actual import actual_events
    initial = opening_position()
    after = relocated(initial, [('black_horse_1', 4, 4)])
    held = board_image(after)
    # 棋子背面没有正面文字，旁边的手指颜色不能被当成红字。
    cv2.circle(held, (450, 450), 43, (225, 225, 225), -1)
    cv2.rectangle(held, (500, 405), (600, 440), (90, 130, 190), -1)
    frames = [(0, board_image(initial)), (.1, held), (.3, held),
              (1, board_image(after)), (1.1, board_image(after))]
    events = list(actual_events(iter(frames), initial))
    assert len([e for e in events if e.position is not None]) == 2
    assert events[-1].time_s == 1


def test_clear_visible_lift_at_range_start_uses_before_lift_position(monkeypatch):
    import app.actual as actual
    initial = opening_position()
    lifted = initial.model_copy(deep=True)
    lifted.pieces = [p for p in lifted.pieces if p.id != 'red_cannon_1']
    frames = [(8, board_image(initial)), (8.2, board_image(initial)),
              (9, board_image(lifted)), (9.2, board_image(lifted)), (10, board_image(lifted))]
    monkeypatch.setattr(actual, 'iter_frames', lambda path, start, end: iter((t, f) for t, f in frames if start <= t <= end))
    calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50}, {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    time, image = actual.find_initial_frame('lift.mp4', 10, calibration)
    assert time == 8.2


def test_model_confirmed_last_frame_does_not_add_false_eof_review():
    from app.actual import actual_events
    initial = opening_position()
    after = initial.model_copy(deep=True)
    from app.schemas import Piece
    after.pieces.append(Piece(id='added', color='red', kind='cannon', square=Square(file=4, rank=4)))
    reference = [initial]
    events = []
    for event in actual_events(iter([(0, board_image(initial)), (.1, board_image(after)), (1.2, board_image(after))]),
                               initial, lambda: reference[0]):
        events.append(event)
        if event.position is None:
            reference[0] = after
    assert len(events) == 2
    assert not events[-1].review_reason


def test_unrecognized_complete_snapshot_does_not_disappear_after_later_clear_board(tmp_path, monkeypatch):
    import app.actual as actual
    from app.analysis import VideoEvent, analyze_task
    initial = opening_position()
    middle = relocated(initial, [('red_cannon_1', 3, 5)])
    after = relocated(middle, [('black_horse_1', 2, 4)])
    monkeypatch.setattr(actual, 'actual_events', lambda *args, **kwargs: iter([
        VideoEvent(0, [board_image(initial)], position=initial),
        VideoEvent(1, [board_image(initial), board_image(middle)]),
        VideoEvent(2, [board_image(middle), board_image(after)], position=after)]))
    store = TaskStore(tmp_path)
    task = store.create('gap.mp4', VideoMetadata(width=901, height=1001, duration_s=3, codec='h264'))
    task.replay_mode, task.initial_position = 'actual', initial
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50}, {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    store.save(task, 0)
    analyze_task(task.id, store, None)
    result = store.get(task.id)
    assert result.status == 'needs_review'
    assert [r.status for r in result.moves] == ['needs_review']
    assert result.moves[0].position_after is None
    assert result.moves[0].board_check.status == 'conflict'


def test_same_color_same_occupancy_glyph_change_still_reaches_model():
    from app.actual import actual_events
    initial = opening_position()
    image = board_image(initial)
    changed = image.copy()
    cv2.circle(changed, (50, 50), 43, (220, 220, 220), -1)
    cv2.putText(changed, 'W', (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (30, 30, 30), 6)
    events = list(actual_events(iter([(0, image), (.1, changed), (.3, changed), (.5, changed), (1.2, changed)]), initial))
    assert any(e.position is None and not e.review_reason for e in events)


def test_visible_piece_face_still_held_by_fingers_waits_for_release():
    from app.actual import actual_events
    initial = opening_position()
    after = relocated(initial, [('red_cannon_1', 4, 4)])
    held = board_image(after)
    cv2.rectangle(held, (400, 390), (520, 530), (90, 130, 190), -1)
    cv2.circle(held, (450, 450), 43, (220, 220, 220), -1)
    cv2.putText(held, 'P', (435, 465), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (50, 30, 180), 5)
    events = list(actual_events(iter([(0, board_image(initial)), (.1, held), (.3, held),
                                    (1, board_image(after)), (1.1, board_image(after))]), initial))
    assert len(events) == 2
    assert events[-1].time_s == 1


def test_hand_over_cannon_waits_for_landing_without_review():
    from app.actual import actual_events
    initial = opening_position()
    after = relocated(initial, [('red_cannon_1', 3, 5)])
    hidden = board_image(initial)
    cv2.rectangle(hidden, (90, 690), (310, 800), (90, 130, 190), -1)
    frames = [(0, board_image(initial)), (.1, hidden), (.7, hidden),
              (1, board_image(after)), (1.1, board_image(after)), (1.2, board_image(after))]
    events = list(actual_events(iter(frames), initial))
    assert [e.position for e in events] == [initial, after.model_copy(update={'side_to_move': 'black'})]
    assert all(not e.uncertain for e in events)
    assert events[-1].time_s == 1


def test_pick_up_and_put_back_does_not_create_review_or_move():
    from app.actual import actual_events
    initial = opening_position()
    hidden = board_image(initial)
    cv2.rectangle(hidden, (90, 690), (210, 800), (90, 130, 190), -1)
    frames = [(0, board_image(initial)), (.1, hidden), (.8, hidden),
              (1, board_image(initial)), (1.2, board_image(initial))]
    events = list(actual_events(iter(frames), initial))
    assert len(events) == 1 and events[0].position == initial


def test_boundary_backward_search_is_adaptive_not_fixed_seconds(monkeypatch):
    import app.actual as actual
    initial = opening_position()
    hidden = board_image(initial)
    cv2.rectangle(hidden, (90, 690), (210, 800), (90, 130, 190), -1)
    clear = board_image(initial)
    frames = [(5.6, clear), (5.7, clear), (5.8, clear)] + [(t / 10, hidden) for t in range(59, 123)]
    calls = []
    def read(path, start, end):
        calls.append((start, end))
        return iter((t, image) for t, image in frames if start <= t <= end)
    monkeypatch.setattr(actual, 'iter_frames', read)
    calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50}, {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    time, image = actual.find_initial_frame('fixture.mp4', 12.2, calibration)
    assert time == pytest.approx(5.8)
    assert min(start for start, _ in calls) <= 5.6


def test_pipeline_filters_preroll_but_keeps_move_crossing_range_start(tmp_path, monkeypatch):
    import app.actual as actual
    from app.analysis import analyze_task
    initial = opening_position()
    before = relocated(initial, [('black_horse_1', 2, 4)])
    after = relocated(before, [('red_cannon_1', 3, 5)])
    frames = [(8, board_image(initial)), (8.1, board_image(initial)),
              (9, board_image(before)), (9.1, board_image(before)),
              (10.1, board_image(after)), (10.2, board_image(after))]
    monkeypatch.setattr(actual, 'iter_frames', lambda path, start, end: iter((t, f) for t, f in frames if start <= t <= end))
    store = TaskStore(tmp_path)
    task = store.create('range.mp4', VideoMetadata(width=901, height=1001, duration_s=12, codec='h264'))
    task.start_s, task.end_s = 10, 11
    task.replay_mode = 'actual'
    task.initial_position, task.initial_frame_s = initial, 8
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50}, {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    task.status = 'analyzing'
    store.save(task, 0)
    analyze_task(task.id, store, None)
    saved = store.get(task.id)
    assert saved.status == 'ready'
    assert len(saved.moves) == saved.confirmed_prefix_length == 1
    assert saved.moves[0].video_end_s == pytest.approx(10.1)
    assert saved.moves[0].video_start_s >= 10
    assert next(p for p in saved.initial_position.pieces if p.id == 'black_horse_1').square.rank == 4
    assert saved.moves[0].move.piece_id == 'red_cannon_1'
    assert saved.analysis_stats.model_requests == 0
    saved.moves, saved.confirmed_prefix_length = [], 0
    saved = store.save(saved, saved.revision)
    analyze_task(saved.id, store, None)
    repeated = store.get(saved.id)
    assert repeated.status == 'ready'
    assert len(repeated.moves) == 1
    assert repeated.moves[0].move.piece_id == 'red_cannon_1'


def test_actual_vision_uses_known_identity_for_nonlegal_move_and_final_image():
    import httpx
    import json
    from app.vision import VisionClient
    initial = opening_position()
    after = relocated(initial, [('red_cannon_1', 3, 5)])
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        pieces = [{'color': p.color, 'kind': p.kind, 'file': p.square.file, 'rank': p.square.rank} for p in after.pieces]
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({'candidates': [
            {'pieces': pieces, 'side_to_move': None}]})}}], 'usage': {'total_tokens': 8}})
    client = httpx.Client(transport=httpx.MockTransport(respond))
    vision = VisionClient('https://example.test/v1', 'placeholder', 'fixture', client)
    result = vision.recognize_position([board_image(initial), board_image(after)], reference=initial, actual=True)
    assert len(result) == 1
    assert next(p for p in result[0].pieces if p.id == 'red_cannon_1').square == Square(file=3, rank=5)
    prompt = requests[0]['messages'][0]['content'][0]['text']
    assert 'red_cannon_1' in prompt
    assert '最后' in prompt
    vision.close()


def test_worker_automatically_initializes_without_confirming_turn(tmp_path, monkeypatch):
    import app.actual as actual
    from app.analysis import analyze_task
    initial = opening_position()
    after = relocated(initial, [('black_horse_1', 2, 4)])
    frames = [(9.5, board_image(initial)), (9.7, board_image(initial)),
              (10.1, board_image(after)), (10.2, board_image(after))]
    monkeypatch.setattr(actual, 'iter_frames', lambda path, start, end: iter((t, f) for t, f in frames if start <= t <= end))
    store = TaskStore(tmp_path)
    task = store.create('auto.mp4', VideoMetadata(width=901, height=1001, duration_s=12, codec='h264'))
    task.start_s, task.end_s, task.replay_mode = 10, 11, 'actual'
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50}, {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    store.save(task, 0)
    class Vision:
        usage = []
        def recognize_position(self, images, reference=None, actual=False):
            assert actual and len(images) == 2
            return [initial, initial.model_copy(update={'side_to_move': 'black'})]
        def close(self):
            pass
    analyze_task(task.id, store, Vision())
    saved = store.get(task.id)
    assert saved.status == 'ready'
    assert saved.initial_position is not None
    assert saved.initial_frame_s == 9.7
    assert saved.moves[0].move.piece_id == 'black_horse_1'
    assert saved.analysis_stats.model_requests == 1 and saved.analysis_stats.images_sent == 2


def test_api_analysis_without_initial_runs_in_cancellable_worker(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    import app.main as main
    app = main.create_app(tmp_path)
    task = app.state.store.create('new-range.mp4', VideoMetadata(width=901, height=1001, duration_s=900, codec='h264'))
    task.start_s, task.end_s = 500, 700
    app.state.store.save(task, 0)
    class Vision:
        configured = True
        def close(self): pass
    class Process:
        alive = False
        def __init__(self, **kwargs): pass
        def start(self): self.alive = True
        def is_alive(self): return self.alive
        def terminate(self): self.alive = False
        def join(self, timeout=None): pass
    monkeypatch.setattr(main, 'VisionClient', Vision)
    monkeypatch.setattr(main.multiprocessing, 'get_context', lambda method: SimpleNamespace(Process=Process))
    with TestClient(app) as client:
        response = client.post(f'/api/tasks/{task.id}/analyze')
        assert response.status_code == 202
        assert response.json()['replay_mode'] == 'actual'
        assert response.json()['initial_position'] is None
        stopped = client.post(f'/api/tasks/{task.id}/stop').json()
        assert stopped['status'] == 'interrupted'
        assert stopped['analysis_stats']['model_requests'] == 0


def test_revalidate_and_correct_actual_snapshot_do_not_apply_chess_rules(tmp_path):
    from app.actual import snapshot_record
    from app.analysis import revalidate_suffix, correct_interval
    from app.schemas import Move
    initial = opening_position()
    after = relocated(initial, [('red_cannon_1', 3, 5)])
    task = TaskStore(tmp_path).create('setup.mp4', VideoMetadata(width=901, height=1001, duration_s=10, codec='h264'))
    task.replay_mode, task.initial_position = 'actual', initial
    task.moves = [snapshot_record(initial, after, 1, 0, 1)]
    rebuilt = revalidate_suffix(task, 1)
    assert rebuilt.status == 'ready'
    action = Move(index=1, color='red', piece_id='red_cannon_1', source=Square(file=1, rank=7), target=Square(file=4, rank=4))
    corrected = correct_interval(task, 1, [action], 1)
    assert corrected.moves[0].status == 'corrected'
    assert next(p for p in corrected.moves[0].position_after.pieces if p.id == 'red_cannon_1').square.rank == 4


def test_api_can_confirm_full_setup_snapshot_with_added_piece(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.schemas import MoveRecord, Piece
    app = create_app(tmp_path)
    task = app.state.store.create('setup.mp4', VideoMetadata(width=901, height=1001, duration_s=10, codec='h264'))
    task.replay_mode, task.initial_position = 'actual', opening_position()
    task.moves = [MoveRecord(index=1, video_start_s=0, video_end_s=1, status='needs_review')]
    after = task.initial_position.model_copy(deep=True)
    after.pieces.append(Piece(id='extra_cannon', color='red', kind='cannon', square=Square(file=4, rank=4)))
    task = app.state.store.save(task, 0)
    with TestClient(app) as client:
        response = client.patch(f'/api/tasks/{task.id}/moves/1', json={
            'revision': task.revision, 'position': after.model_dump(by_alias=True), 'video_time_s': 1})
        assert response.status_code == 200
        assert response.json()['moves'][0]['status'] == 'corrected'
        assert len(response.json()['moves'][0]['position_after']['pieces']) == 33


def test_saving_unchanged_settings_preserves_preroll_context(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    app = create_app(tmp_path)
    task = app.state.store.create('repeat.mp4', VideoMetadata(width=901, height=1001, duration_s=20, codec='h264'))
    task.initial_position = relocated(opening_position(), [('black_horse_1', 2, 4)])
    task.context_position, task.initial_frame_s, task.start_s = opening_position(), 8, 10
    task = app.state.store.save(task, 0)
    with TestClient(app) as client:
        response = client.patch(f'/api/tasks/{task.id}/setup', json={'revision': task.revision,
            'start_s': task.start_s, 'end_s': task.end_s, 'initial_position': task.initial_position.model_dump()})
        assert response.status_code == 200
        assert response.json()['initial_frame_s'] == 8
        assert response.json()['context_position'] == task.context_position.model_dump()


def test_occluded_start_does_not_drop_unfinished_event_at_end():
    import app.actual as actual
    initial = opening_position()
    hidden = board_image(initial)
    cv2.rectangle(hidden, (90, 690), (210, 800), (90, 130, 190), -1)
    events = list(actual.actual_events(iter([(0, hidden), (1, hidden)]), initial))
    assert len(events) == 2 and events[-1].review_reason


def test_temporary_glyph_movement_does_not_trigger_confirmation():
    from app.actual import actual_events
    initial = opening_position()
    image = board_image(initial)
    changed = image.copy()
    cv2.circle(changed, (50, 50), 43, (220, 220, 220), -1)
    cv2.putText(changed, 'W', (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (30, 30, 30), 6)
    events = list(actual_events(iter([(0, image), (.1, changed), (.3, changed), (.5, image), (1.2, image)]), initial))
    assert len(events) == 1


def test_unknown_completed_change_returning_to_original_stays_reviewable(tmp_path, monkeypatch):
    import app.actual as actual
    from app.analysis import VideoEvent, analyze_task
    initial = opening_position()
    changed = relocated(initial, [('red_cannon_1', 3, 5)])
    monkeypatch.setattr(actual, 'actual_events', lambda *args, **kwargs: iter([
        VideoEvent(0, [board_image(initial)], position=initial),
        VideoEvent(1, [board_image(changed)]),
        VideoEvent(2, [board_image(initial)], position=initial)]))
    store = TaskStore(tmp_path)
    task = store.create('returned.mp4', VideoMetadata(width=901, height=1001, duration_s=3, codec='h264'))
    task.replay_mode, task.initial_position = 'actual', initial
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50}, {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    store.save(task, 0)
    analyze_task(task.id, store, None)
    assert store.get(task.id).moves[0].status == 'needs_review'


def test_full_snapshot_correction_after_gap_uses_last_known_position(tmp_path):
    from app.actual import correct_actual, snapshot_record
    from app.schemas import MoveRecord
    initial = opening_position()
    after = relocated(initial, [('red_cannon_1', 3, 5)])
    task = TaskStore(tmp_path).create('gap-edit.mp4', VideoMetadata(width=901, height=1001, duration_s=3, codec='h264'))
    task.replay_mode, task.initial_position = 'actual', initial
    task.moves = [MoveRecord(index=1,video_start_s=0,video_end_s=1,status='needs_review'), snapshot_record(initial,after,2,1,2)]
    result = correct_actual(task, 2, [], 2, position=after)
    assert [r.status for r in result.moves] == ['needs_review','corrected']
    assert {p.id: p.square for p in result.moves[1].position_after.pieces} == {p.id: p.square for p in after.pieces}
