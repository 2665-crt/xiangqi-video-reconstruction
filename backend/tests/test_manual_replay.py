from fastapi.testclient import TestClient

from app.actual import correct_actual, snapshot_record
from app.main import create_app
from app.recording import build_recording_playlist
from app.schemas import Move, MoveRecord, Square, VideoMetadata, opening_position
from app.storage import TaskStore
from test_actual import relocated


def fixture_task(store):
    task = store.create('manual.mp4', VideoMetadata(width=900, height=1000, duration_s=5, codec='h264'))
    task.initial_position, task.replay_mode, task.rules_enabled = opening_position(), 'actual', True
    red = relocated(task.initial_position, [('red_rook_0', 0, 8)])
    red.side_to_move = 'black'
    black = relocated(red, [('black_rook_0', 0, 1)])
    black.side_to_move = 'red'
    task.moves = [snapshot_record(task.initial_position, red, 1, 0, 1), snapshot_record(red, black, 2, 1, 2)]
    task.processed_video_s, task.status, task.confirmed_prefix_length = 5, 'ready', 2
    return task


def test_correction_uses_saved_previous_board_and_propagates_without_changing_moves(tmp_path):
    store = TaskStore(tmp_path)
    task = fixture_task(store)
    task.moves[0].position_before = relocated(task.initial_position, [('black_advisor_3', 4, 1)])
    original_reply = task.moves[1].move.model_dump()
    action = Move(index=1, piece_id='red_horse_1', color='red', source=Square(file=1, rank=9), target=Square(file=2, rank=7))
    fixed = correct_actual(task, 1, [action], 1)
    assert len(fixed.moves) == 2
    assert next(p for p in fixed.moves[0].position_after.pieces if p.id == 'black_advisor_3').square == Square(file=3, rank=0)
    assert next(p for p in fixed.moves[1].position_after.pieces if p.id == 'red_horse_1').square == Square(file=2, rank=7)
    assert next(p for p in fixed.moves[1].position_after.pieces if p.id == 'red_rook_0').square == Square(file=0, rank=9)
    assert fixed.moves[1].move.model_dump() == original_reply
    saved = store.save(fixed, 0)
    assert store.get(saved.id).moves[1].position_after == fixed.moves[1].position_after


def test_illegal_move_is_saved_with_warning_and_suffix_stays_playable(tmp_path):
    task = fixture_task(TaskStore(tmp_path))
    action = Move(index=1, piece_id='black_horse_1', color='black', source=Square(file=1, rank=0), target=Square(file=1, rank=3))
    fixed = correct_actual(task, 1, [action], 1)
    assert fixed.moves[0].status == 'corrected'
    assert fixed.moves[0].violation_note
    assert fixed.moves[1].position_after is not None
    assert fixed.moves[1].violation_note
    assert len(build_recording_playlist(fixed, 1)['records']) == 2


def test_position_add_replace_delete_survives_later_moves(tmp_path):
    task = fixture_task(TaskStore(tmp_path))
    edited = task.initial_position.model_copy(deep=True)
    edited.pieces = [p for p in edited.pieces if p.id != 'black_advisor_3']
    horse = next(p for p in edited.pieces if p.id == 'red_horse_1')
    horse.kind = 'cannon'
    horse.square = Square(file=2, rank=7)
    fixed = correct_actual(task, 1, [], 1, edited)
    assert fixed.moves[0].change_type == 'setup'
    assert not any(p.id == 'black_advisor_3' for p in fixed.moves[1].position_after.pieces)
    assert next(p for p in fixed.moves[1].position_after.pieces if p.id == horse.id).kind == 'cannon'


def test_ignored_record_does_not_change_board_or_export(tmp_path):
    task = fixture_task(TaskStore(tmp_path))
    fixed = correct_actual(task, 1, [], 1)
    assert fixed.moves[0].change_type == 'review'
    assert fixed.moves[0].position_after == task.initial_position
    assert next(p for p in fixed.moves[1].position_after.pieces if p.id == 'red_rook_0').square == Square(file=0, rank=9)
    playlist = build_recording_playlist(fixed, 1)
    assert len(playlist['records']) == 1
    assert playlist['duration_s'] == 2


def test_api_accepts_illegal_snapshot_and_counts_only_visible_records(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        task = fixture_task(app.state.store)
        task = app.state.store.save(task, 0)
        edited = relocated(task.initial_position, [('red_rook_0', 1, 8)])
        response = client.patch(f'/api/tasks/{task.id}/moves/1', json={
            'revision': task.revision, 'position': edited.model_dump(), 'video_time_s': 1})
        assert response.status_code == 200
        assert response.json()['moves'][0]['violation_note']
        response = client.patch(f'/api/tasks/{task.id}/moves/1', json={
            'revision': response.json()['revision'], 'moves': [], 'video_time_s': 1})
        assert response.status_code == 200
        assert client.get('/api/tasks').json()[0]['steps'] == 1


def test_missing_piece_in_retained_move_is_warned_without_restoring_deleted_piece(tmp_path):
    task = fixture_task(TaskStore(tmp_path))
    edited = task.initial_position.model_copy(deep=True)
    edited.pieces = [p for p in edited.pieces if p.id != 'black_rook_0']
    fixed = correct_actual(task, 1, [], 1, edited)
    assert not any(p.id == 'black_rook_0' for p in fixed.moves[1].position_after.pieces)
    assert fixed.moves[1].move.piece_id == 'black_rook_0'
    assert fixed.moves[1].violation_note
    assert len(build_recording_playlist(fixed, 1)['records']) == 2


def test_worker_save_cannot_restore_old_suffix_after_manual_correction(tmp_path):
    store = TaskStore(tmp_path)
    task = fixture_task(store)
    task = store.save(task, 0)
    worker = task.model_copy(deep=True)
    action = Move(index=1, piece_id='red_horse_1', color='red', source=Square(file=1, rank=9), target=Square(file=2, rank=7))
    fixed = correct_actual(task, 1, [action], 1)
    store.save(fixed, task.revision)
    saved = store.save_analysis(worker)
    assert next(p for p in saved.moves[1].position_after.pieces if p.id == 'red_horse_1').square == Square(file=2, rank=7)


def test_later_manual_position_edits_do_not_restore_unrelated_old_pieces(tmp_path):
    task = fixture_task(TaskStore(tmp_path))
    edited = task.moves[1].position_after.model_copy(deep=True)
    next(p for p in edited.pieces if p.id == 'black_horse_1').kind = 'cannon'
    fixed = correct_actual(task, 2, [], 2, edited)
    action = Move(index=1, piece_id='red_horse_1', color='red', source=Square(file=1, rank=9), target=Square(file=2, rank=7))
    fixed = correct_actual(fixed, 1, [action], 1)
    assert next(p for p in fixed.moves[1].position_after.pieces if p.id == 'red_horse_1').square == Square(file=2, rank=7)
    assert next(p for p in fixed.moves[1].position_after.pieces if p.id == 'black_horse_1').kind == 'cannon'


def test_horse_leg_warning_does_not_block_manual_move(tmp_path):
    task = fixture_task(TaskStore(tmp_path))
    action = Move(index=1, piece_id='red_horse_1', color='red', source=Square(file=1, rank=9), target=Square(file=3, rank=8))
    fixed = correct_actual(task, 1, [action], 1)
    assert fixed.moves[0].status == 'corrected'
    assert '蹩马腿' in fixed.moves[0].violation_note


def test_later_kind_edit_preserves_new_upstream_square(tmp_path):
    task = fixture_task(TaskStore(tmp_path))
    edited = task.moves[1].position_after.model_copy(deep=True)
    next(p for p in edited.pieces if p.id == 'black_horse_1').kind = 'cannon'
    fixed = correct_actual(task, 2, [], 2, edited)
    action = Move(index=1, piece_id='black_horse_1', color='black', source=Square(file=1, rank=0), target=Square(file=1, rank=4))
    fixed = correct_actual(fixed, 1, [action], 1)
    piece = next(p for p in fixed.moves[1].position_after.pieces if p.id == 'black_horse_1')
    assert piece.kind == 'cannon'
    assert piece.square == Square(file=1, rank=4)


def test_later_deletion_remains_after_upstream_capture_is_revised(tmp_path):
    task = fixture_task(TaskStore(tmp_path))
    edited = task.moves[1].position_after.model_copy(deep=True)
    edited.pieces = [p for p in edited.pieces if p.id != 'black_horse_1']
    fixed = correct_actual(task, 2, [], 2, edited)
    action = Move(index=1, piece_id='red_rook_0', color='red', source=Square(file=0, rank=9), target=Square(file=1, rank=0), captured_piece_id='black_horse_1')
    fixed = correct_actual(fixed, 1, [action], 1)
    action.target, action.captured_piece_id = Square(file=0, rank=8), None
    fixed = correct_actual(fixed, 1, [action], 1)
    assert not any(p.id == 'black_horse_1' for p in fixed.moves[1].position_after.pieces)


def test_known_suffix_with_old_rule_failure_remains_playable(tmp_path):
    task = fixture_task(TaskStore(tmp_path))
    task.moves[1].status, task.moves[1].position_after = 'needs_review', None
    action = Move(index=1, piece_id='red_horse_1', color='red', source=Square(file=1, rank=9), target=Square(file=2, rank=7))
    fixed = correct_actual(task, 1, [action], 1)
    assert fixed.moves[1].status == 'auto_verified'
    assert len(build_recording_playlist(fixed, 1)['records']) == 2


def test_old_legal_task_accepts_same_correction_api(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        task = fixture_task(app.state.store)
        task.replay_mode, task.rules_enabled = 'legal', False
        task = app.state.store.save(task, 0)
        edited = relocated(task.initial_position, [('red_rook_0', 1, 8)])
        response = client.patch(f'/api/tasks/{task.id}/moves/1', json={
            'revision': task.revision, 'position': edited.model_dump(), 'video_time_s': 1})
        assert response.status_code == 200
        response = client.patch(f'/api/tasks/{task.id}/moves/1', json={
            'revision': response.json()['revision'], 'moves': [], 'video_time_s': 1})
        assert response.status_code == 200
        assert len(response.json()['moves']) == 2
        assert response.json()['moves'][0]['change_type'] == 'review'


def test_position_swap_preserves_both_piece_identities(tmp_path):
    task = fixture_task(TaskStore(tmp_path))
    edited = task.moves[0].position_after.model_copy(deep=True)
    next(p for p in edited.pieces if p.id == 'red_rook_0').square = Square(file=8, rank=9)
    next(p for p in edited.pieces if p.id == 'red_rook_8').square = Square(file=0, rank=8)
    fixed = correct_actual(task, 1, [], 1, edited)
    pieces = {p.id: p for p in fixed.moves[0].position_after.pieces}
    assert len(pieces) == 32
    assert pieces['red_rook_0'].square == Square(file=8, rank=9)
    assert pieces['red_rook_8'].square == Square(file=0, rank=8)
