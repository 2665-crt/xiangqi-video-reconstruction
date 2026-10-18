from fractions import Fraction

import av
import pytest
from fastapi.testclient import TestClient

from app.actual import snapshot_record
from app.chess import apply_move
from app.main import create_app
from app.schemas import Calibration, Move, Square, VideoMetadata, opening_position
from app.storage import TaskStore
from test_tracking import board_image


def fixture_task(store):
    task = store.create('range.mp4', VideoMetadata(width=902, height=1002, duration_s=8, codec='h264'))
    initial = opening_position()
    positions, actions = [initial], []
    for identity, target in [('red_pawn_0', (0, 5)), ('black_pawn_0', (0, 4)),
                             ('red_pawn_2', (2, 5)), ('black_pawn_2', (2, 4))]:
        piece = next(p for p in positions[-1].pieces if p.id == identity)
        action = Move(index=len(actions) + 1, color=piece.color, piece_id=identity,
                      source=piece.square, target=Square(file=target[0], rank=target[1]))
        actions.append(action)
        positions.append(apply_move(positions[-1], action))
    task.initial_position, task.replay_mode, task.rules_enabled = initial, 'actual', True
    task.calibration = Calibration(corners=[{'x': 50, 'y': 50}, {'x': 850, 'y': 50},
                                          {'x': 850, 'y': 950}, {'x': 50, 'y': 950}])
    task.calibration_confirmed = True
    task.moves = [snapshot_record(initial, positions[1], 1, 0, 1),
                  snapshot_record(positions[3], positions[4], 2, 1, 6.5)]
    for r, start, end in [(task.moves[0], .5, 1), (task.moves[1], 6, 6.5)]:
        r.action_start_s, r.action_end_s, r.timing_status = start, end, 'observed'
    task.status, task.confirmed_prefix_length = 'ready', 2
    task.processed_video_s = task.end_s
    return task, positions, actions


def test_insertion_keeps_old_step_and_replays_shifted_suffix(tmp_path):
    from app.turns import insert_turn
    task, positions, actions = fixture_task(TaskStore(tmp_path))
    first = insert_turn(task, 2, actions[1], 3, 2.5)
    fixed = insert_turn(first, 3, actions[2], 4, 3.5)
    assert [r.move.piece_id for r in fixed.moves] == [
        'red_pawn_0', 'black_pawn_0', 'red_pawn_2', 'black_pawn_2']
    assert [r.index for r in fixed.moves] == [1, 2, 3, 4]
    assert fixed.moves[3].move.index == 4
    assert fixed.moves[3].position_before == positions[3]
    assert fixed.moves[3].position_after == positions[4]
    assert fixed.moves[3].action_start_s == 6
    assert task.moves[1].index == 2


def test_insertion_rejects_timestamp_outside_neighbor_actions(tmp_path):
    from app.turns import insert_turn
    task, _, actions = fixture_task(TaskStore(tmp_path))
    with pytest.raises(ValueError, match='时间'):
        insert_turn(task, 2, actions[1], 6.1, 5.9)


def test_range_sorts_click_order_and_requires_observed_boundaries(tmp_path):
    from app.recheck import range_bounds
    task, _, _ = fixture_task(TaskStore(tmp_path))
    assert range_bounds(task, 2, 1) == (1, 2, 1, 6)
    with pytest.raises(ValueError):
        range_bounds(task, 1, 1)
    task.moves[1].action_start_s = None
    with pytest.raises(ValueError, match='动作时间'):
        range_bounds(task, 1, 2)


def test_replace_only_interior_preserves_boundary_and_remaps_related_steps(tmp_path):
    from app.recheck import replace_range
    task, positions, _ = fixture_task(TaskStore(tmp_path))
    task.moves[1].related_steps = [1, 2]
    recovered = [snapshot_record(positions[1], positions[2], 1, 1, 3),
                 snapshot_record(positions[2], positions[3], 2, 3, 4)]
    for r in recovered:
        r.action_start_s, r.action_end_s, r.timing_status = r.video_end_s - .5, r.video_end_s, 'observed'
    fixed = replace_range(task, 1, 2, recovered)
    assert len(fixed.moves) == 4
    assert fixed.moves[0] == task.moves[0]
    assert fixed.moves[-1].related_steps == [1, 4]
    assert fixed.moves[-1].position_after == positions[4]
    assert fixed.moves[-1].move.piece_id == 'black_pawn_2'


def encode_clip(path, positions):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=10)
        stream.width, stream.height, stream.pix_fmt = 902, 1002, 'yuv420p'
        import cv2
        for i in range(80):
            t = i / 10
            p = positions[0 if t < .5 else 1 if t < 2.5 else 2 if t < 3.5 else 3 if t < 6 else 4]
            image = cv2.copyMakeBorder(board_image(p), 0, 1, 0, 1, cv2.BORDER_REPLICATE)
            frame = av.VideoFrame.from_ndarray(image, format='bgr24')
            frame.pts, frame.time_base = i, Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_real_decoded_video_recovers_both_missing_moves_in_order(tmp_path):
    from app.recheck import trace_range
    store = TaskStore(tmp_path)
    task, positions, _ = fixture_task(store)
    path = store.task_dir(task.id) / 'source.mp4'
    encode_clip(path, positions)
    progress = []
    records, endpoint_matches = trace_range(path, task.calibration, positions[1], 1, 6,
        store.task_dir(task.id) / 'evidence', lambda t, n, records: progress.append((t, n)))
    moves = [r for r in records if r.move is not None]
    assert [r.move.piece_id for r in moves] == ['black_pawn_0', 'red_pawn_2']
    assert endpoint_matches is True
    assert all(r.status == 'auto_verified' for r in moves)
    assert 2.4 <= moves[0].action_start_s <= 2.6
    assert 3.4 <= moves[1].action_start_s <= 3.6
    assert progress[-1][1] == 50  # 1.0 <= t < 6.0 at ten frames/second
    assert all(r.evidence_ids for r in moves)


def test_insert_route_does_not_replace_selected_step(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        task, _, actions = fixture_task(app.state.store)
        task = app.state.store.save(task, 0)
        response = client.post(f'/api/tasks/{task.id}/moves/2/insert', json={
            'revision': task.revision, 'move': actions[1].model_dump(),
            'action_start_s': 2.5, 'video_time_s': 3})
        assert response.status_code == 200
        assert [r['move']['piece_id'] for r in response.json()['moves']] == [
            'red_pawn_0', 'black_pawn_0', 'black_pawn_2']
        stale = client.post(f'/api/tasks/{task.id}/moves/2/insert', json={
            'revision': task.revision, 'move': actions[1].model_dump(), 'video_time_s': 3})
        assert stale.status_code == 409


def test_manual_position_edit_is_reported_even_when_move_text_matches(tmp_path):
    from app.recheck import comparison
    task, positions, _ = fixture_task(TaskStore(tmp_path))
    original = snapshot_record(positions[1], positions[2], 2, 1, 3).model_copy(deep=True)
    original.manual_override = True
    next(p for p in original.position_after.pieces if p.id == 'black_horse_1').kind = 'cannon'
    task.moves.insert(1, original)
    task.moves[-1].index = 3
    candidate = snapshot_record(positions[1], positions[2], 1, 1, 3)
    assert comparison(task, 1, 3, [candidate])[3] == [2]


def test_identical_manual_correction_survives_recheck_replacement(tmp_path):
    from app.recheck import replace_range
    task, positions, _ = fixture_task(TaskStore(tmp_path))
    original = snapshot_record(positions[1], positions[2], 2, 1, 3)
    original.manual_override, original.status = True, 'corrected'
    task.moves.insert(1, original)
    task.moves[-1].index = 3
    candidate = snapshot_record(positions[1], positions[2], 1, 1, 3)
    fixed = replace_range(task, 1, 3, [candidate])
    assert fixed.moves[1].manual_override is True
    assert fixed.moves[1].status == 'corrected'


def test_api_recheck_keeps_original_until_apply_and_updates_export(tmp_path):
    import time
    from app.recording import build_recording_playlist
    app = create_app(tmp_path)
    with TestClient(app) as client:
        task, positions, _ = fixture_task(app.state.store)
        encode_clip(app.state.store.task_dir(task.id) / 'source.mp4', positions)
        task = app.state.store.save(task, 0)
        started = client.post(f'/api/tasks/{task.id}/recheck', json={
            'revision': task.revision, 'first_index': 2, 'second_index': 1})
        assert started.status_code == 202
        response = started.json()
        assert response['recheck']['start_s'] == 1 and response['recheck']['end_s'] == 6
        blocked = client.post(f'/api/tasks/{task.id}/moves/2/insert', json={
            'revision': response['revision'], 'move': task.moves[1].move.model_dump(), 'video_time_s': 3})
        assert blocked.status_code == 409
        for _ in range(150):
            response = client.get(f'/api/tasks/{task.id}').json()
            if response['recheck']['status'] != 'running':
                break
            time.sleep(.05)
        assert response['recheck']['status'] == 'applied'
        assert len(response['moves']) == 4
        assert response['recheck']['added'] == 2
        applied = client.post(f'/api/tasks/{task.id}/recheck/apply', json={
            'revision': response['revision'], 'recheck_id': response['recheck']['id']})
        assert applied.status_code == 200
        saved = app.state.store.get(task.id)
        assert len(saved.moves) == 4
        assert saved.recheck.right_index == 4
        assert len(build_recording_playlist(saved, 1)['records']) == 4
        assert list((app.state.store.task_dir(task.id) / 'recheck-history').glob('*-before.json'))


def test_early_decoder_eof_is_not_reported_as_complete_empty_range(tmp_path, monkeypatch):
    import app.recheck as module
    store = TaskStore(tmp_path)
    task, positions, _ = fixture_task(store)
    path = store.task_dir(task.id) / 'source.mp4'
    encode_clip(path, positions)
    monkeypatch.setattr(module, 'iter_frames', lambda *args: iter([(1.0, board_image(positions[1]))]))
    records, matches = module.trace_range(path, task.calibration, positions[1], 1, 6,
        store.task_dir(task.id) / 'evidence', lambda *args: None)
    assert matches is False
    assert any(r.status == 'needs_review' and '完整' in r.reason for r in records)


def test_rechecked_timing_replaces_bad_manual_time_without_reordering_moves(tmp_path):
    from app.recheck import replace_range
    task, positions, _ = fixture_task(TaskStore(tmp_path))
    original = snapshot_record(positions[1], positions[2], 2, 1, 5)
    original.manual_override, original.status = True, 'corrected'
    original.action_start_s, original.action_end_s, original.timing_status = 4.5, 5, 'observed'
    task.moves.insert(1, original)
    task.moves[-1].index = 3
    records = [snapshot_record(positions[1], positions[2], 1, 1, 3),
               snapshot_record(positions[2], positions[3], 2, 3, 4)]
    for r in records:
        r.action_start_s, r.action_end_s, r.timing_status = r.video_end_s - .5, r.video_end_s, 'observed'
    fixed = replace_range(task, 1, 3, records)
    assert [r.action_end_s for r in fixed.moves] == [1, 3, 4, 6.5]


def test_cancelled_worker_cannot_save_into_a_new_recheck(tmp_path):
    from app.storage import RevisionConflict
    app = create_app(tmp_path)
    with TestClient(app) as client:
        task, positions, _ = fixture_task(app.state.store)
        encode_clip(app.state.store.task_dir(task.id) / 'source.mp4', positions)
        task = app.state.store.save(task, 0)
        started = client.post(f'/api/tasks/{task.id}/recheck', json={
            'revision': task.revision, 'first_index': 1, 'second_index': 2}).json()
        old_job = started['recheck']['id']
        cancelled = client.post(f'/api/tasks/{task.id}/recheck/cancel', json={
            'revision': started['revision'], 'recheck_id': old_job})
        assert cancelled.status_code == 200
        assert cancelled.json()['recheck']['status'] == 'cancelled'
        restarted = client.post(f'/api/tasks/{task.id}/recheck', json={
            'revision': cancelled.json()['revision'], 'first_index': 1, 'second_index': 2}).json()
        assert restarted['recheck']['id'] != old_job
        with pytest.raises(RevisionConflict):
            app.state.store.update_recheck(task.id, old_job, lambda saved: saved.moves.clear())
        assert len(app.state.store.get(task.id).moves) == 2
        stopped = client.post(f'/api/tasks/{task.id}/recheck/cancel', json={
            'revision': restarted['revision'], 'recheck_id': restarted['recheck']['id']})
        assert stopped.status_code == 200


def test_partial_recovery_retains_moves_and_marks_covered_endpoint_unknown(tmp_path, monkeypatch):
    import cv2
    import app.recheck as module
    store = TaskStore(tmp_path)
    task, positions, _ = fixture_task(store)
    path = store.task_dir(task.id) / 'source.mp4'
    encode_clip(path, positions)
    hidden = board_image(positions[3])
    cv2.rectangle(hidden, (0, 0), (500, 500), (20, 20, 20), -1)
    frames = [(1, board_image(positions[1])), (2.5, board_image(positions[2])),
              (2.6, board_image(positions[2])), (3.5, board_image(positions[3])),
              (3.6, board_image(positions[3])), (5.9, hidden)]
    monkeypatch.setattr(module, 'iter_frames', lambda *args: iter(frames))
    records, matches = module.trace_range(path, task.calibration, positions[1], 1, 6,
        store.task_dir(task.id) / 'evidence', lambda *args: None)
    recovered = [r.move or (r.candidates[0] if len(r.candidates) == 1 and r.board_check is not None
                           and r.board_check.status in ('verified', 'partial') else None) for r in records]
    assert [m.piece_id for m in recovered if m] == ['black_pawn_0', 'red_pawn_2']
    assert any(r.candidates and r.status == 'needs_review' for r in records)
    assert matches is False
    assert any(r.status == 'needs_review' for r in records)
