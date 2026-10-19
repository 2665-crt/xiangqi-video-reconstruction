from fastapi.testclient import TestClient

from app.main import create_app
from app.recording import build_recording_playlist
from app.schemas import Move, MoveRecord, Square, Task, VideoMetadata, opening_position


def make_task():
    before = opening_position()
    after = before.model_copy(deep=True)
    next(p for p in after.pieces if p.id == 'red_rook_0').square = Square(file=0, rank=8)
    action = Move(index=1, piece_id='red_rook_0', color='red', source=Square(file=0, rank=9), target=Square(file=0, rank=8))
    record = MoveRecord(index=1, move=action, position_before=before, position_after=after,
                        status='auto_verified', video_start_s=0, video_end_s=1)
    return Task(id='c' * 32, filename='long.mp4', metadata=VideoMetadata(width=901, height=1001, duration_s=10, codec='h264'),
                initial_position=before, moves=[record], end_s=10, processed_video_s=10, status='ready')


def test_task_list_does_not_count_empty_reviews_as_moves(tmp_path):
    app = create_app(tmp_path)
    stored = app.state.store.create('long.mp4', make_task().metadata)
    task = make_task()
    task.id = stored.id
    task.moves.extend(MoveRecord(index=i, status='needs_review', video_start_s=1, video_end_s=9) for i in range(2, 1002))
    app.state.store.save(task, 0)
    with TestClient(app) as client:
        assert client.get('/api/tasks').json()[0]['steps'] == 1


def test_confirmed_stationary_record_does_not_add_export_frame():
    task = make_task()
    record = task.moves[0].model_copy(deep=True)
    record.index = 2
    record.move.target = record.move.source
    record.position_after = record.position_before
    task.moves.append(record)
    assert len(build_recording_playlist(task, 1)['records']) == 1


def test_retained_command_without_any_piece_movement_is_not_a_step():
    from app.records import movement_records
    task = make_task()
    task.moves[0].position_after = task.moves[0].position_before.model_copy(deep=True)
    assert movement_records(task) == []


def test_unresolved_gap_does_not_block_current_steps_export():
    task = make_task()
    task.moves.append(MoveRecord(index=2, status='needs_review', video_start_s=1, video_end_s=9))
    payload = build_recording_playlist(task, 1)
    assert len(payload['records']) == 1
    assert payload['records'][0]['index'] == 1
    assert task.moves[1].status == 'needs_review'
