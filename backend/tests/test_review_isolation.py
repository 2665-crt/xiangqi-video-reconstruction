import pytest
from fastapi.testclient import TestClient

from app.actual import correct_actual, snapshot_record
from app.main import create_app
from app.schemas import Move, MoveRecord, Square, VideoMetadata, opening_position
from app.storage import TaskStore
from app.turns import TurnReplay
from test_actual import relocated


def test_clear_reply_is_confirmed_after_illegal_step(tmp_path):
    task = TaskStore(tmp_path).create('isolated.mp4', VideoMetadata(width=900, height=1000, duration_s=4, codec='h264'))
    task.rules_enabled, task.replay_mode, task.initial_position = True, 'actual', opening_position()
    replay = TurnReplay(task, task.initial_position)
    illegal = relocated(task.initial_position, [('red_rook_0', 1, 8)])
    reply = relocated(illegal, [('black_rook_0', 0, 1)])
    replay.observe(illegal, 1, [])
    replay.observe(reply, 2, [])
    replay.end(4)
    assert [r.status for r in task.moves] == ['needs_review', 'auto_verified']
    assert task.moves[0].position_after is None
    assert task.moves[1].move.piece_id == 'black_rook_0'
    assert task.moves[1].position_before.pieces == illegal.pieces


def test_mixed_color_observation_splits_unique_legal_turns(tmp_path):
    initial = opening_position()
    task = TaskStore(tmp_path).create('sliding.mp4', VideoMetadata(width=900, height=1000, duration_s=4, codec='h264'))
    task.rules_enabled, task.replay_mode, task.initial_position = True, 'actual', initial
    replay = TurnReplay(task, initial)
    red = relocated(initial, [('red_horse_1', 2, 7)])
    mixed = relocated(red, [('black_horse_1', 2, 2), ('red_rook_0', 1, 9)])
    replay.observe(red, 1, [])
    replay.observe(mixed, 2, [])
    replay.end(4)
    assert len(task.moves) == 3
    assert all(r.status == 'auto_verified' for r in task.moves)
    assert [r.move.piece_id for r in task.moves] == ['red_horse_1', 'black_horse_1', 'red_rook_0']
    assert all(r.violation is None for r in task.moves)


def test_review_has_system_candidate_and_can_be_edited_after_earlier_gap(tmp_path):
    task = TaskStore(tmp_path).create('review.mp4', VideoMetadata(width=900, height=1000, duration_s=4, codec='h264'))
    task.rules_enabled, task.replay_mode, task.initial_position = True, 'actual', opening_position()
    replay = TurnReplay(task, task.initial_position)
    illegal = relocated(task.initial_position, [('red_rook_0', 1, 8)])
    reply = relocated(illegal, [('black_rook_0', 0, 1)])
    replay.observe(illegal, 1, [])
    replay.observe(reply, 2, [])
    replay.end(4)
    assert task.moves[0].candidates
    assert task.moves[0].candidates[0].piece_id == 'red_rook_0'
    action = Move(index=2, piece_id='black_rook_0', color='black', source=Square(file=0, rank=0), target=Square(file=0, rank=2))
    corrected = correct_actual(task, 2, [action], 3)
    assert corrected.moves[0].status == 'needs_review'
    assert corrected.moves[1].status == 'corrected'
    assert corrected.moves[1].move.target == Square(file=0, rank=2)


def test_revalidation_keeps_clear_independent_reply(tmp_path):
    initial = opening_position()
    illegal = relocated(initial, [('red_rook_0', 1, 8)])
    illegal.side_to_move = 'black'
    reply = relocated(illegal, [('black_rook_0', 0, 1)])
    task = TaskStore(tmp_path).create('review.mp4', VideoMetadata(width=900, height=1000, duration_s=4, codec='h264'))
    task.rules_enabled, task.replay_mode, task.initial_position = True, 'actual', initial
    task.moves = [MoveRecord(index=1, video_start_s=0, video_end_s=1, status='needs_review', observed_positions=[illegal]),
                  snapshot_record(illegal, reply, 2, 1, 2).model_copy(update={'position_before': illegal})]
    legal = Move(index=1, piece_id='red_rook_0', color='red', source=Square(file=0, rank=9), target=Square(file=0, rank=8))
    corrected = correct_actual(task, 1, [legal], 1)
    assert corrected.moves[1].status == 'auto_verified'
    assert corrected.moves[1].move.piece_id == 'black_rook_0'


def test_live_review_is_saved_and_worker_progress_cannot_overwrite_it(tmp_path):
    app = create_app(tmp_path)
    store = app.state.store
    initial = opening_position()
    after = relocated(initial, [('red_rook_0', 0, 8)])
    task = store.create('live.mp4', VideoMetadata(width=900, height=1000, duration_s=4, codec='h264'))
    task.rules_enabled, task.replay_mode, task.initial_position = True, 'actual', initial
    task.moves = [snapshot_record(initial, after, 1, 0, 1)]
    with TestClient(app) as client:
        task.status = 'analyzing'
        task = store.save(task, 0)
        worker_copy = task.model_copy(deep=True)
        move = Move(index=1, piece_id='red_rook_0', color='red', source=Square(file=0, rank=9), target=Square(file=0, rank=7))
        response = client.patch(f'/api/tasks/{task.id}/moves/1', json={
            'revision': task.revision, 'move': move.model_dump(by_alias=True), 'video_time_s': 1})
        assert response.status_code == 200
        assert response.json()['status'] == 'analyzing'
        saved = store.save_analysis(worker_copy)
        assert saved.moves[0].status == 'corrected'
        assert saved.moves[0].move.target == Square(file=0, rank=7)


def test_slide_final_and_reply_in_one_observation_do_not_insert_middle_step(tmp_path):
    initial = opening_position()
    task = TaskStore(tmp_path).create('slide.mp4', VideoMetadata(width=900, height=1000, duration_s=4, codec='h264'))
    task.rules_enabled, task.replay_mode, task.initial_position = True, 'actual', initial
    replay = TurnReplay(task, initial)
    middle = relocated(initial, [('red_rook_0', 0, 8)])
    final = relocated(initial, [('red_rook_0', 0, 7), ('black_rook_0', 0, 1)])
    replay.observe(middle, 1, [])
    replay.observe(final, 2, [])
    replay.end(4)
    assert len(task.moves) == 2
    assert task.moves[0].move.source == Square(file=0, rank=9)
    assert task.moves[0].move.target == Square(file=0, rank=7)


def test_unknown_initial_turn_does_not_invent_order_for_both_colors(tmp_path):
    initial = opening_position()
    task = TaskStore(tmp_path).create('unknown.mp4', VideoMetadata(width=900, height=1000, duration_s=4, codec='h264'))
    task.rules_enabled, task.replay_mode, task.initial_position, task.initial_turn_known = True, 'actual', initial, False
    replay = TurnReplay(task, initial)
    final = relocated(initial, [('red_rook_0', 0, 8), ('black_rook_0', 0, 1)])
    replay.observe(final, 1, [])
    replay.end(4)
    assert task.moves[0].status == 'needs_review'
    assert not any(r.status == 'auto_verified' for r in task.moves)


def test_no_move_review_keeps_its_independent_baseline(tmp_path):
    initial = opening_position()
    task = TaskStore(tmp_path).create('baseline.mp4', VideoMetadata(width=900, height=1000, duration_s=4, codec='h264'))
    task.rules_enabled, task.replay_mode, task.initial_position = True, 'actual', initial
    replay = TurnReplay(task, initial)
    illegal = relocated(initial, [('red_rook_0', 1, 8)])
    replay.observe(illegal, 1, [])
    replay.observe(relocated(illegal, [('black_rook_0', 0, 1)]), 2, [])
    replay.end(4)
    corrected = correct_actual(task, 2, [], 4)
    assert corrected.moves[1].position_before.side_to_move == 'black'
    assert corrected.moves[1].position_after.side_to_move == 'black'


def test_final_worker_save_recomputes_status_after_last_manual_review(tmp_path):
    store = TaskStore(tmp_path)
    task = store.create('late-review.mp4', VideoMetadata(width=900, height=1000, duration_s=4, codec='h264'))
    task.rules_enabled, task.replay_mode, task.initial_position = True, 'actual', opening_position()
    task.status = 'needs_review'
    task.moves = [MoveRecord(index=1, video_start_s=0, video_end_s=1, status='needs_review')]
    task = store.save(task, 0)
    worker = task.model_copy(deep=True)
    action = Move(index=1, color='red', piece_id='red_rook_0', source=Square(file=0, rank=9), target=Square(file=0, rank=8))
    corrected = correct_actual(task, 1, [action], 1)
    store.save(corrected, task.revision)
    saved = store.save_analysis(worker)
    assert saved.status == 'ready'
    assert saved.confirmed_prefix_length == 1
