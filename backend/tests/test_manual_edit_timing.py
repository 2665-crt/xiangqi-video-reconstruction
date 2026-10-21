import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.recording import build_recording_playlist
from app.schemas import Move, Square, Task
from test_manual_replay import fixture_task


def horse_move():
    return Move(index=1, piece_id='red_horse_1', color='red',
                source=Square(file=1, rank=9), target=Square(file=2, rank=7))


@pytest.mark.parametrize('index', [2, 3])
def test_insert_rebases_existing_piece_from_its_saved_square(tmp_path, index):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        task = app.state.store.save(fixture_task(app.state.store), 0)
        action = Move(index=index, piece_id='red_rook_0', color='red',
                      source=Square(file=0, rank=9), target=Square(file=0, rank=7))
        response = client.post(f'/api/tasks/{task.id}/moves/{index}/insert', json={
            'revision': task.revision, 'move': action.model_dump(by_alias=True), 'video_time_s': 4})
        assert response.status_code == 200, response.text
        saved = Task.model_validate(response.json())
        record = saved.moves[index - 1]
        assert record.move.source == Square(file=0, rank=8)
        assert next(p for p in record.position_before.pieces if p.id == action.piece_id).square == record.move.source
        assert next(p for p in record.position_after.pieces if p.id == action.piece_id).square == action.target
        assert app.state.store.get(task.id).moves == saved.moves
        assert len(build_recording_playlist(saved, 1)['records']) == 3


@pytest.mark.parametrize('index,time_s,start_s', [(1, 4, 3.5), (2, .25, .1), (3, 0, None), (2, 1, 2)])
def test_insert_at_selected_index_accepts_overlapping_or_reversed_video_times(tmp_path, index, time_s, start_s):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        task = app.state.store.save(fixture_task(app.state.store), 0)
        original_ids = [r.move.piece_id for r in task.moves]
        response = client.post(f'/api/tasks/{task.id}/moves/{index}/insert', json={
            'revision': task.revision, 'move': horse_move().model_dump(by_alias=True),
            'video_time_s': time_s, 'action_start_s': start_s})
        assert response.status_code == 200, response.text
        saved = Task.model_validate(response.json())
        assert [r.move.piece_id for r in saved.moves] == original_ids[:index - 1] + ['red_horse_1'] + original_ids[index - 1:]
        assert [r.index for r in saved.moves] == [1, 2, 3]
        assert all(r.video_start_s <= r.video_end_s for r in saved.moves)
        assert saved.moves[index - 1].action_end_s == time_s
        assert app.state.store.get(task.id).moves == saved.moves
        assert len(build_recording_playlist(saved, 1)['records']) == 3
        for record in saved.moves[index - 1:]:
            horse = next(p for p in record.position_after.pieces if p.id == 'red_horse_1')
            assert horse.square == Square(file=2, rank=7)


@pytest.mark.parametrize('index,time_s,start_s', [(1, 4, 3.5), (2, .25, .1), (1, 1, 2)])
@pytest.mark.parametrize('mode', ['move', 'position', 'remove'])
def test_edit_or_remove_is_not_blocked_by_neighbor_or_start_time(tmp_path, index, time_s, start_s, mode):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        task = app.state.store.save(fixture_task(app.state.store), 0)
        request = {'revision': task.revision, 'video_time_s': time_s, 'action_start_s': start_s}
        if mode == 'position':
            edited = task.moves[index - 1].position_after.model_copy(deep=True)
            next(p for p in edited.pieces if p.id == 'red_horse_1').square = Square(file=2, rank=7)
            request['position'] = edited.model_dump()
        else:
            request['moves'] = [] if mode == 'remove' else [horse_move().model_dump(by_alias=True)]
        response = client.patch(f'/api/tasks/{task.id}/moves/{index}', json=request)
        assert response.status_code == 200, response.text
        saved = Task.model_validate(response.json())
        record = saved.moves[index - 1]
        assert record.status == 'corrected'
        assert len(saved.moves) == len(task.moves)
        assert all(r.video_start_s <= r.video_end_s for r in saved.moves)
        assert app.state.store.get(task.id).moves == saved.moves
        if mode == 'remove':
            assert record.change_type == 'review' and record.move is None
            assert record.video_end_s == task.moves[index - 1].video_end_s
            assert len(build_recording_playlist(saved, 1)['records']) == 1
        else:
            assert record.action_end_s == time_s
            if start_s > time_s:
                assert record.action_start_s is None
                assert record.timing_status == 'uncertain'
            else:
                assert record.action_start_s == start_s
        for old, current in zip(task.moves[index:], saved.moves[index:]):
            assert current.move == old.move


@pytest.mark.parametrize('replay_mode', ['legal', 'actual'])
def test_legacy_interval_edit_also_accepts_time_after_following_step(tmp_path, replay_mode):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        task = fixture_task(app.state.store)
        task.replay_mode, task.rules_enabled = replay_mode, False
        task = app.state.store.save(task, 0)
        reply = Move(index=2, piece_id='black_horse_1', color='black',
                     source=Square(file=1, rank=0), target=Square(file=2, rank=2))
        response = client.patch(f'/api/tasks/{task.id}/moves/1', json={
            'revision': task.revision, 'video_time_s': 4,
            'moves': [horse_move().model_dump(by_alias=True), reply.model_dump(by_alias=True)]})
        assert response.status_code == 200, response.text
        saved = Task.model_validate(response.json())
        assert [r.move.piece_id for r in saved.moves[:2]] == ['red_horse_1', 'black_horse_1']
        assert all(r.video_start_s <= r.video_end_s for r in saved.moves)

